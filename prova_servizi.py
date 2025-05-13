"""
Applicazione Flask per la schedulazione ottimizzata dei dipendenti.

Caratteristiche chiave
----------------------
* **Valore pausa dinamico** – La colonna «PAUSE» della matrice `s` vale il 60 % della
  media dello stress previsto negli altri tre reparti (Produzione, Forni, Confezionamento)
  per ogni singolo dipendente.
* **Mapping reparti dinamico** – Gli `idReparto` provenienti dalle API non sono assunti
  come 1 / 2 / 3: il mapping *nome‑reparto ➜ idReparto* è ricavato runtime dal JSON.
* **Gestione flag `riposo`** – Se `turno["riposo"]` è `true` il dipendente viene
  inizialmente messo in pausa; se è `false` rimane nel reparto indicato.
* **Fallback neutri** – Se mancano dati di stress un valore di 50 viene usato come
  default.
"""

from __future__ import annotations

from datetime import datetime
import argparse
import json
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from amplpy import AMPL
from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# ENUM e costanti dominio
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
# Modello dominio
# ---------------------------------------------------------------------------

class Employee:
    """Info turno + dati stress per un dipendente."""

    def __init__(self, id_smartwatch: int, turno: Dict[str, Any], stress: float | None = None):
        self.id_smartwatch = id_smartwatch
        self.stress = stress  # ultima misura reale

        # ---- dati turno ----
        self.id_turno: int | None = turno.get("idTurno")
        self.giorno: str | None = turno.get("giorno")
        self.id_reparto: int | None = turno.get("idReparto")
        self.reparto: str | None = turno.get("reparto")
        self.id_area_lavoro: int | None = turno.get("idAreaLavoro")
        self.area_lavoro: str | None = turno.get("areaLavoro")
        self.fascia_oraria: str | None = turno.get("fasciaOraria")
        self.id_settimana: int | None = turno.get("idSettimana")
        self.tm_inizio: str | None = turno.get("tmInizio")
        self.tm_fine: str | None = turno.get("tmFine")

        # Flag riposo: true = pausa
        self.rest: bool = bool(turno.get("riposo", False))

        # Stress previsti per reparto {idReparto: valore}
        self.predicted_stresses: Dict[int, float] = {}

        # Forza reparto «PAUSE» se riposo
        if self.rest:
            self.set_pausa()

    # ------------------------------------------------------------------
    def set_pausa(self) -> None:
        self.reparto = "PAUSE"
        self.id_reparto = None

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Employee(sw={self.id_smartwatch}, rep={self.reparto}, rest={self.rest}, "
            f"stress={self.stress})"
        )

# ---------------------------------------------------------------------------
# Flask app & stato globale
# ---------------------------------------------------------------------------

app = Flask(__name__)

curr_assignment: List[int] | None = None  # vettore j0 AMPL
api_mode: bool = False                    # True se si usano le API
location_default = "CAPANNONE NUOVO"

# ---------------------------------------------------------------------------
# Utility (versione senza API)
# ---------------------------------------------------------------------------

def generate_stress_matrix(rows: int, cols: int, assignment: List[int] | None) -> np.ndarray:
    if assignment is not None:
        matrix = np.vstack([
            np.full(cols, 60) if a == 0 else np.random.randint(50, 100, cols)
            for a in assignment
        ])
        matrix[:, 0] = 20  # pausa fissa nel test
        return matrix.astype(float)

    return np.hstack((
        np.full((rows, 1), 10),
        np.random.randint(50, 100, size=(rows, cols - 1))
    )).astype(float)

# ---------------------------------------------------------------------------
# Integrazione servizi REST
# ---------------------------------------------------------------------------

def launch_service() -> Tuple[Any, Any, Any, Any]:
    with open("service.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    headers = {
        "Cdclient": cfg["auth"]["cdclient"],
        "Origin": cfg["auth"]["origin"],
        "Authorization": f"Bearer {cfg['auth']['token']}",
    }

    now = datetime.now()
    params = {
        "day": now.day,
        "month": now.month,
        "year": now.year,
        "h": now.hour,
        "min": now.minute,
    }
    params_num = params | {"numValues": 1}

    urls = list(cfg["url"].items())
    response_map = {}
    for svc in Service:
        _, url = urls[svc.value]
        if svc in {Service.REPARTI, Service.MEAN_STR, Service.CURR_SCHED}:
            resp = requests.get(url, params=params if svc != Service.REPARTI else None, headers=headers)
        else:
            resp = requests.post(url, params=params_num, headers=headers)
        response_map[svc] = resp.json()

    return (
        response_map[Service.CURR_SCHED],
        response_map[Service.CURR_STR],
        response_map[Service.MEAN_STR],
        response_map[Service.PRED_STR],
    )

# ---------------------------------------------------------------------------
# Costruzione Employee + previsioni stress
# ---------------------------------------------------------------------------

def build_employee_list(resps: Tuple[Any, Any, Any, Any]) -> List[Employee]:
    curr_sched, curr_str, mean_str, pred_str = resps

    dept_avg: Dict[int, float] = {d["idReparto"]: d["stress"] for d in mean_str}

    emp_list: List[Employee] = []
    for rec in curr_sched:
        sw_id = rec.get("idSmartwatch")
        turno = rec.get("turno", {})

        key = f"dev-sim-{sw_id}"
        stress_val: float | None = None
        if key in curr_str:
            readings = pred_str.get(key, [])
            if readings:
                stress_val = readings[-1]["value"]

        emp = Employee(sw_id, turno, stress_val)
        emp_list.append(emp)

    # ----- calcolo stress previsti per reparto -----
    for emp in emp_list:
        curr_avg = dept_avg.get(emp.id_reparto)
        rel_diff: float | None = None
        if emp.stress is not None and curr_avg:
            rel_diff = (emp.stress - curr_avg) / curr_avg

        for rep_id, rep_avg in dept_avg.items():
            emp.predicted_stresses[rep_id] = rep_avg * (1 + rel_diff) if rel_diff is not None else 50.0

    return emp_list

# ---------------------------------------------------------------------------
# Matrice `s` e j0
# ---------------------------------------------------------------------------

def build_prediction_matrix_and_initial_positions(emp_list: List[Employee]) -> Tuple[np.ndarray, List[int]]:
    # mapping nome ➜ id reperto
    name_to_id: Dict[str, int] = {
        e.reparto.upper(): e.id_reparto for e in emp_list if e.reparto and e.id_reparto
    }

    ordered_names = ["PRODUZIONE", "FORNI", "CONFEZIONAMENTO"]
    ordered_ids = [name_to_id.get(n) for n in ordered_names]

    matrix: List[List[float]] = []
    for emp in emp_list:
        rep_vals = [emp.predicted_stresses.get(rid, 50.0) if rid is not None else 50.0 for rid in ordered_ids]
        pause_val = 0.6 * float(np.mean(rep_vals))
        matrix.append([pause_val] + rep_vals)

    j0 = [Task[emp.reparto].value - 1 for emp in emp_list]
    print(j0)
    return np.array(matrix, dtype=float), j0

# ---------------------------------------------------------------------------
# Costruzione risposta JSON
# ---------------------------------------------------------------------------

def build_schedule_response(sol: np.ndarray):
    res_data: List[Dict[str, Any]] = []
    pauses: List[Dict[str, Any]] = []

    for idx, row in enumerate(sol, start=1):
        task_idx = int(np.argmax(row)) + 1  # 1‑indexed
        task_name = Task(task_idx).name
        if task_name == "PAUSE":
            pauses.append({"id": str(idx)})
        else:
            res_data.append({
                "id": str(idx),
                "location": location_default,
                "task": task_name,
            })

    return jsonify({"assignments": res_data, "pauses": pauses})

# ---------------------------------------------------------------------------
# Solutore AMPL + CPLEX
# ---------------------------------------------------------------------------

def solve_optimization(rows: int | None = None, cols: int | None = None, matrix: np.ndarray | None = None, assignment: List[int] | None = None):
    global curr_assignment

    ampl = AMPL()
    ampl.read("model.mod")
    ampl.readData("param_values.dat")

    if matrix is None:
        rows, cols = 10, 4
        matrix = generate_stress_matrix(rows, cols, curr_assignment)
        if curr_assignment is None:
            curr_assignment = [1] * rows
    elif curr_assignment is None:
        curr_assignment = assignment

    emp_idx = list(range(1, rows + 1))
    dept_idx = list(range(cols))

    ampl.getSet("EMPLOYEES").setValues(emp_idx)
    ampl.getSet("DEPARTMENTS").setValues(dept_idx)

    alpha = ampl.getParameter("alpha")
    for i in emp_idx:
        for j in dept_idx:
            alpha.set((i, j), 1)

    smax = ampl.getParameter("Smax")
    for i in emp_idx:
        smax.set((i,), 90)

    s_param = ampl.getParameter("s")
    for i, r in enumerate(emp_idx, start=0):
        for j, c in enumerate(dept_idx, start=0):
            s_param.set((r, c), float(matrix[i, j]))

    j0_param = ampl.getParameter("j0")
    for i, v in enumerate(curr_assignment, start=1):
        j0_param.set((i,), v)

    ampl.option["solver"] = "cplex"
    ampl.solve()

    sol = np.zeros_like(matrix)
    for i in emp_idx:
        for j in dept_idx:
            sol[i-1, j] = ampl.getVariable("x").get(i, j).value()

    # update j0
    curr_assignment = np.where(sol.argmax(axis=1) >= 0, sol.argmax(axis=1), 0).tolist()

    return build_schedule_response(sol)

# ---------------------------------------------------------------------------
# Flask endpoint
# ---------------------------------------------------------------------------

@app.route("/schedule", methods=["GET"])
def schedule_api():
    global api_mode, emp_list, pred_matrix, init_pos
    if api_mode:
        return solve_optimization(len(emp_list), pred_matrix.shape[1], pred_matrix, init_pos)
    return solve_optimization()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scheduler dipendenti")
    parser.add_argument("--api", action="store_true", help="Usa i servizi remoti")
    args = parser.parse_args()

    api_mode = args.api

    if api_mode:
        json_resps = launch_service()
        emp_list = build_employee_list(json_resps)
        pred_matrix, init_pos = build_prediction_matrix_and_initial_positions(emp_list)

    app.run(debug=True)
