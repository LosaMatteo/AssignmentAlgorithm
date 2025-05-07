"""
Applicazione Flask per la schedulazione ottimizzata dei dipendenti.

Aggiornamenti principali rispetto alla versione precedente
---------------------------------------------------------
1. **Valore pausa dinamico** – La colonna "PAUSE" della matrice `s` è ora pari al 60 % della
   media delle tre colonne di reparto (Produzione, Forni, Confezionamento) calcolata per
   ciascun dipendente.
2. **Mapping reparti dinamico** – Gli idReparto che arrivano dai servizi REST non sono
   più assunti come 1, 2, 3.  Il codice ora costruisce dinamicamente il mapping
   *nome‑reparto → idReparto* e lo riutilizza per scrivere le colonne della matrice.
3. **Pulizia dei default** – Il fallback quando mancano dati di stress rimane a 50 (valore
   neutro) ma non finisce più nelle colonne se i dati corretti sono presenti.
"""

from flask import Flask, jsonify
import numpy as np
from amplpy import AMPL
from enum import Enum
from datetime import datetime
import json
import requests
import argparse
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# ENUM e costanti di dominio
# ---------------------------------------------------------------------------

class Task(Enum):
    PAUSE = 1
    PRODUZIONE = 2
    FORNI = 3
    CONFEZIONAMENTO = 4

class Service(Enum):
    REPARTI = 0
    MEAN_STR = 1
    CURR_SCHED = 2
    CURR_STR = 3
    PRED_STR = 4

# ---------------------------------------------------------------------------
# Modello di dominio
# ---------------------------------------------------------------------------

class Employee:
    """Rappresenta un dipendente con i dati del turno e le misure di stress."""

    def __init__(self, id_smartwatch: int, turno: Dict[str, Any], stress: float | None = None):
        self.id_smartwatch = id_smartwatch
        self.stress = stress  # ultima misura reale dal device

        # info turno (direttamente dal JSON dei servizi)
        self.id_turno: int | None = turno.get("idTurno")
        self.giorno: str | None = turno.get("giorno")
        self.id_reparto: int | None = turno.get("idReparto")
        self.reparto: str | None = turno.get("reparto")  # nome (es. "PRODUZIONE")
        self.id_area_lavoro: int | None = turno.get("idAreaLavoro")
        self.area_lavoro: str | None = turno.get("areaLavoro")
        self.fascia_oraria: str | None = turno.get("fasciaOraria")
        self.id_settimana: int | None = turno.get("idSettimana")
        self.tm_inizio: str | None = turno.get("tmInizio")
        self.tm_fine: str | None = turno.get("tmFine")

        # previsioni di stress per reparto {idReparto: valore}
        self.predicted_stresses: Dict[int, float] = {}

    # ---------------------------------------------------------------------
    # helper
    # ---------------------------------------------------------------------

    def set_pausa(self) -> None:
        self.reparto = "PAUSE"

    # ---------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Employee(id_smartwatch={self.id_smartwatch}, id_turno={self.id_turno}, "
            f"giorno={self.giorno}, reparto={self.reparto}, stress={self.stress}, "
            f"predicted_stresses={self.predicted_stresses})"
        )

# ---------------------------------------------------------------------------
# Flask app e variabili globali
# ---------------------------------------------------------------------------

app = Flask(__name__)

# variabili di stato (mantengono contesto tra le chiamate)
curr_assignment: List[int] | None = None  # vettore j0
api: bool | None = None                   # true se si usano i servizi remoti
location = "CAPANNONE NUOVO"              # location default per le assegnazioni

# ---------------------------------------------------------------------------
# Utilità locali – quando NON si usano le API
# ---------------------------------------------------------------------------

def generate_stress_matrix(rows: int, cols: int, assignment: List[int] | None) -> np.ndarray:
    """Crea una matrice random per test locali (senza API)."""
    if assignment is not None:
        matrix = np.empty((0, cols))
        for elem in assignment:
            if elem == 0:
                row = np.full((1, cols), 60)          # partiamo da valori medio‑bassi in pausa
            else:
                row = np.random.randint(50, 100, (1, cols))
            matrix = np.vstack((matrix, row))
        matrix[:, 0] = 20                              # colonna pausa costante nei test
        return matrix

    # primo giro: colonna pausa = 10, altre random
    return np.hstack((np.full((rows, 1), 10), np.random.randint(50, 100, size=(rows, cols - 1))))

# ---------------------------------------------------------------------------
# Integrazione con i servizi REST
# ---------------------------------------------------------------------------

def launch_service() -> Tuple[Any, Any, Any, Any]:
    """Chiama i servizi remoti e ritorna i JSON già decodificati."""
    file_name = "service.json"
    with open(file_name, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    url_dict = json_data["url"]
    url_list = list(url_dict.items())

    headers = {
        "Cdclient": json_data["auth"]["cdclient"],
        "Origin": json_data["auth"]["origin"],
        "Authorization": f"Bearer {json_data['auth']['token']}"
    }

    now = datetime.now()
    params_base = {
        "day": now.day,
        "month": now.month,
        "year": now.year,
        "h": now.hour,
        "min": now.minute,
    }

    for elem in Service:
        _, url = url_list[elem.value]
        if elem is Service.REPARTI:
            response_reparti = requests.get(url, headers=headers)
        elif elem is Service.MEAN_STR:
            response_meanstr = requests.get(url, params=params_base, headers=headers)
        elif elem is Service.CURR_SCHED:
            response_currsched = requests.get(url, params=params_base, headers=headers)
        elif elem is Service.CURR_STR:
            response_currstr = requests.post(url, params=params_base | {"numValues": 1}, headers=headers)
        elif elem is Service.PRED_STR:
            response_predstr = requests.post(url, params=params_base | {"numValues": 1}, headers=headers)

    return (
        response_currsched.json(),
        response_currstr.json(),
        response_meanstr.json(),
        response_predstr.json(),
    )

# ---------------------------------------------------------------------------
# Creazione lista Employee e previsioni di stress
# ---------------------------------------------------------------------------

def build_employee_list(service_responses: Tuple[Any, Any, Any, Any]) -> List[Employee]:
    """Crea la lista di Employee popolando le previsioni di stress per reparto."""

    curr_sched, curr_str, mean_str, pred_str = service_responses

    # {idReparto: stress_medio}
    department_stress_dict: Dict[int, float] = {
        dep["idReparto"]: dep["stress"] for dep in mean_str
    }

    employee_list: List[Employee] = []
    for item in curr_sched:
        id_smartwatch = item.get("idSmartwatch")
        turno = item.get("turno", {})

        key = f"dev-sim-{id_smartwatch}"
        stress: float | None = None
        if key in curr_str:
            measurements = pred_str.get(key, [])
            if measurements:
                stress = measurements[-1].get("value")

        employee = Employee(id_smartwatch, turno, stress)
        if turno.get("riposo"):
            employee.set_pausa()
        employee_list.append(employee)

    # -----------------------------------------------------------
    # Calcolo stress previsto per ciascun reparto
    # -----------------------------------------------------------
    for emp in employee_list:
        current_avg = department_stress_dict.get(emp.id_reparto)
        if emp.stress is not None and current_avg:
            relative_diff = (emp.stress - current_avg) / current_avg
        else:
            relative_diff = None

        for dept_id, dept_avg in department_stress_dict.items():
            if relative_diff is not None:
                predicted = dept_avg * (1 + relative_diff)
            else:
                predicted = 50  # fallback
            emp.predicted_stresses[dept_id] = predicted

    return employee_list

# ---------------------------------------------------------------------------
# Matrice di previsione e posizioni iniziali (con pausa = 0.6 * media)
# ---------------------------------------------------------------------------

def build_prediction_matrix_and_initial_positions(
    employees: List[Employee],
) -> Tuple[np.ndarray, List[int]]:
    """Restituisce la matrice `s` e il vettore j0 per AMPL.

    - Colonna 0: Pausa (60 % della media delle altre tre).
    - Colonne 1‑3: Produzione, Forni, Confezionamento nell'ordine logico.
    """

    # Mapping nome‑reparto → idReparto (dinamico)
    dept_name_to_id: Dict[str, int] = {}
    for e in employees:
        if e.reparto and e.id_reparto:  # ci aspettiamo queste chiavi sul JSON
            dept_name_to_id.setdefault(e.reparto.upper(), e.id_reparto)

    # Ordine colonne fisso (nome logico); pescando l'id dinamico dal mapping
    ordered_names = ["PRODUZIONE", "FORNI", "CONFEZIONAMENTO"]
    desired_dept_order: List[int] = [dept_name_to_id.get(name) for name in ordered_names]

    prediction_matrix: List[List[float]] = []
    for emp in employees:
        dept_vals: List[float] = []
        for dept_id in desired_dept_order:
            if dept_id is None:  # reparto non presente nei dati
                dept_vals.append(50.0)  # valore neutro
            else:
                dept_vals.append(emp.predicted_stresses.get(dept_id, 50.0))

        pause_val = 0.6 * float(np.mean(dept_vals))
        prediction_matrix.append([pause_val] + dept_vals)

    # vettore j0  (Task enum è 1‑indexed; AMPL si aspetta 0‑indexed, quindi -1)
    initial_positions = [Task[emp.reparto].value - 1 for emp in employees]

    return np.array(prediction_matrix, dtype=float), initial_positions

# ---------------------------------------------------------------------------
# Formattazione della risposta /schedule
# ---------------------------------------------------------------------------

def build_schedule_response(matrix: np.ndarray):
    """Costruisce l'oggetto JSON di risposta per l'endpoint /schedule."""
    global location

    data: List[List[str]] = [["" for _ in range(matrix.shape[1])] for _ in range(matrix.shape[0])]

    for i in range(matrix.shape[0]):
        id_str = str(i + 1)
        row = matrix[i].tolist()
        col = row.index(1) + 1  # colonna vincente (1‑indexed)
        task_name = Task(col).name  # E.g. "PAUSE" / "PRODUZIONE" / ...

        if task_name == "PAUSE":
            data[i] = [id_str, task_name, ""]
        else:
            data[i] = [id_str, location, task_name]

    assignments = [
        {"id": row[0], "location": row[1], "task": row[2]}
        for row in data if row[1] != "PAUSE"
    ]
    pauses = [{"id": row[0]} for row in data if row[1] == "PAUSE"]

    return jsonify({"assignments": assignments, "pauses": pauses})

# ---------------------------------------------------------------------------
# Risoluzione ottimizzazione
# ---------------------------------------------------------------------------

def solve_optimization(
    rows: int | None = None,
    cols: int | None = None,
    matrix: np.ndarray | None = None,
    assignment: List[int] | None = None,
):
    """Risoluzione del modello AMPL + CPLEX e costruzione della risposta."""

    global curr_assignment

    ampl = AMPL()
    ampl.read("model.mod")
    ampl.readData("param_values.dat")

    # -------------------------------------------------------
    # Caso senza API (demo locale)
    # -------------------------------------------------------
    if matrix is None:
        rows, cols = 10, 4
        matrix = generate_stress_matrix(rows, cols, curr_assignment)
        if curr_assignment is None:
            curr_assignment = [1 for _ in range(rows)]

    # -------------------------------------------------------
    # Caso con API – prima chiamata
    # -------------------------------------------------------
    elif curr_assignment is None:
        curr_assignment = assignment

    # -------------------------------------------------------
    # Caricamento parametri in AMPL
    # -------------------------------------------------------
    employees_idx = list(range(1, rows + 1))
    depts_idx = list(range(cols))
    ampl.getSet("EMPLOYEES").setValues(employees_idx)
    ampl.getSet("DEPARTMENTS").setValues(depts_idx)

    alpha = ampl.getParameter("alpha")
    for i in employees_idx:
        for j in depts_idx:
            alpha.set((i, j), 1)

    smax = ampl.getParameter("Smax")
    for i in employees_idx:
        smax.set((i,), 90)

    s_param = ampl.getParameter("s")
    for i, emp_idx in enumerate(employees_idx, start=0):
        for j, dept_idx in enumerate(depts_idx, start=0):
            s_param.set((emp_idx, dept_idx), float(matrix[i, j]))

    j0_param = ampl.getParameter("j0")
    for i, val in enumerate(curr_assignment, start=1):
        j0_param.set((i,), val)

    ampl.option["solver"] = "cplex"
    ampl.solve()

    # -------------------------------------------------------
    # Lettura soluzione
    # -------------------------------------------------------
    solution = np.zeros_like(matrix)
    for i in employees_idx:
        for j in depts_idx:
            solution[i - 1, j] = ampl.getVariable("x").get(i, j).value()

    response = build_schedule_response(solution)

    # aggiorna j0 per la prossima iterazione
    new_j0 = np.where((solution == 1).any(axis=1), np.argmax(solution == 1, axis=1), 0)
    curr_assignment = new_j0.tolist()
    return response

# ---------------------------------------------------------------------------
# Flask endpoint
# ---------------------------------------------------------------------------

@app.route("/schedule", methods=["GET"])
def schedule_endpoint():  # pragma: no cover
    global api, employees, prediction_matrix, initial_positions
    if api:
        return solve_optimization(
            rows=len(employees),
            cols=prediction_matrix.shape[1],
            matrix=prediction_matrix,
            assignment=initial_positions,
        )
    return solve_optimization()

# ---------------------------------------------------------------------------
# Entry‑point CLI / inizializzazione
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avvia il servizio di scheduling")
    parser.add_argument("--api", action="store_true", help="Usa i servizi remoti invece dei dati random")
    args = parser.parse_args()

    api = args.api

    if api:
        srv_resp = launch_service()
        employees = build_employee_list(srv_resp)
        prediction_matrix, initial_positions = build_prediction_matrix_and_initial_positions(employees)
    app.run(debug=True)
