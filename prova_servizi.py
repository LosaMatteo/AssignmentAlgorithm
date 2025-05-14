from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from amplpy import AMPL
from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# CONFIGURATION & LOGGER
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENUMS per task e servizi e orari di inizio turno
# ---------------------------------------------------------------------------

shift_hours = [0, 8, 16]

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
# MODELLO DOMINIO: Employee
# ---------------------------------------------------------------------------

class Employee:
    """Rappresenta un dipendente con turno e dati di stress."""
    # contatore di classe
    _count: int = 0

    def __init__(
        self,
        id_smartwatch: int,
        turno: Dict[str, Any],
        stress: float | None = None,
    ):
        # incrementa il contatore ad ogni nuova istanza
        Employee._count += 1

        self.id_smartwatch = id_smartwatch
        self.stress = stress
        self.id_turno = turno.get("idTurno")
        self.giorno = turno.get("giorno")
        self.id_reparto = turno.get("idReparto")
        self.reparto = turno.get("reparto")
        self.id_area_lavoro = turno.get("idAreaLavoro")
        self.area_lavoro = turno.get("areaLavoro")
        self.fascia_oraria = turno.get("fasciaOraria")
        self.id_settimana = turno.get("idSettimana")
        self.tm_inizio = turno.get("tmInizio")
        self.tm_fine = turno.get("tmFine")
        self.rest = bool(turno.get("riposo", False))
        self.predicted_stresses: Dict[int, float] = {}
        if self.rest:
            self._set_pausa()

    def _set_pausa(self) -> None:
        self.reparto = "PAUSE"
        self.id_reparto = None

    def __repr__(self) -> str:
        return (
            f"Employee(sw={self.id_smartwatch}, rep={self.reparto}, rest={self.rest}, stress={self.stress})"
        )

    @classmethod
    def total_employees(cls) -> int:
        """Restituisce il numero totale di Employee istanziati."""
        return cls._count

# ---------------------------------------------------------------------------
# APP FLASK e variabili globali
# ---------------------------------------------------------------------------

app = Flask(__name__)
api_mode = False
location_default = "CAPANNONE NUOVO"
curr_assignment: List[int] | None = None
pred_matrix: np.ndarray | None = None
init_pos: List[int] | None = None

# ---------------------------------------------------------------------------
# UTILITY per generare matrice di test
# ---------------------------------------------------------------------------

def generate_stress_matrix(rows: int, cols: int, assignment: List[int] | None) -> np.ndarray:
    if assignment is not None:
        matrix = np.vstack([
            np.full(cols, 60) if a == 0 else np.random.randint(50, 100, cols)
            for a in assignment
        ])
        matrix[:, 0] = 20
        return matrix.astype(float)
    return np.hstack((
        np.full((rows, 1), 10),
        np.random.randint(50, 100, size=(rows, cols - 1)),
    )).astype(float)

# ---------------------------------------------------------------------------
# INTEGRAZIONE SERVIZI REST
# ---------------------------------------------------------------------------

def launch_service() -> Tuple[Any, Any, Any, Any]:
    try:
        with open("service.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.error("Errore lettura service.json: %s", e)
        raise

    headers = {
        "Cdclient": cfg["auth"]["cdclient"],
        "Origin": cfg["auth"]["origin"],
        "Authorization": f"Bearer {cfg['auth']['token']}"
    }
    now = datetime.now()
    params = {"day": now.day, "month": now.month, "year": now.year,
              "h": now.hour, "min": now.minute}
    params_num = {**params, "numValues": 1}
    responses: Dict[Service, Any] = {}
    for svc in Service:
        _, url = list(cfg["url"].items())[svc.value]
        try:
            if svc in {Service.REPARTI, Service.MEAN_STR, Service.CURR_SCHED}:
                resp = requests.get(
                    url,
                    params=None if svc == Service.REPARTI else params,
                    headers=headers,
                    timeout=10
                )
            else:
                resp = requests.post(
                    url,
                    params=params_num,
                    headers=headers,
                    timeout=10
                )
            resp.raise_for_status()
            responses[svc] = resp.json()

        except Exception as e:
            logger.error("Errore API %s a %s: %s", svc.name, url, e)

    return (
        responses[Service.CURR_SCHED],
        responses[Service.CURR_STR],
        responses[Service.MEAN_STR],
        responses[Service.PRED_STR]
    )

# ---------------------------------------------------------------------------
# COSTRUZIONE LISTA DIPENDENTI e previsione stress
# ---------------------------------------------------------------------------

def build_employee_list(resps: Tuple[Any, Any, Any, Any]) -> List[Employee]:
    curr_sched, curr_str, mean_str, pred_str = resps
    dept_avg = {d["idReparto"]: d["stress"] for d in mean_str}
    emp_list: List[Employee] = []
    for rec in curr_sched:
        sw_id = rec.get("idSmartwatch")
        turno = rec.get("turno", {})
        key = f"dev-sim-{sw_id}"
        stress_val = None
        if key in curr_str:
            readings = pred_str.get(key, [])
            if readings:
                stress_val = readings[-1].get("value")
        emp = Employee(sw_id, turno, stress_val)
        emp_list.append(emp)
    for emp in emp_list:
        curr_avg = dept_avg.get(emp.id_reparto)
        rel_diff = None
        if emp.stress is not None and curr_avg:
            rel_diff = (emp.stress - curr_avg) / curr_avg
        for rid, avg in dept_avg.items():
            emp.predicted_stresses[rid] = avg * (1 + (rel_diff or 0))
    return emp_list

# ---------------------------------------------------------------------------
# COSTRUZIONE MATRICE PREVISIONALE e posizioni iniziali
# ---------------------------------------------------------------------------

def build_prediction_matrix_and_initial_positions(
    emp_list: List[Employee]
) -> Tuple[np.ndarray, List[int]]:
    name_to_id = {e.reparto.upper(): e.id_reparto for e in emp_list if e.reparto}
    ordered_names = ["PRODUZIONE", "FORNI", "CONFEZIONAMENTO"]
    ordered_ids = [name_to_id.get(n) for n in ordered_names]
    matrix: List[List[float]] = []
    for emp in emp_list:
        rep_values = [emp.predicted_stresses.get(rid, 50.0) for rid in ordered_ids]
        pause_val = 0.6 * float(np.mean(rep_values))
        matrix.append([pause_val] + rep_values)
    j0 = [Task[emp.reparto].value - 1 if emp.reparto in Task.__members__ else 0 for emp in emp_list]
    return np.array(matrix, dtype=float), j0

# ---------------------------------------------------------------------------
# COSTRUZIONE RISPOSTA JSON
# ---------------------------------------------------------------------------

def build_schedule_response(sol: np.ndarray, simulated: bool = False, reason: str = "") -> Any:
    assignments: List[Dict[str, Any]] = []
    pauses: List[Dict[str, Any]] = []
    for idx, row in enumerate(sol, start=1):
        task_idx = int(np.argmax(row)) + 1
        task_name = Task(task_idx).name
        if task_name == "PAUSE":
            pauses.append({"id": str(idx)})
        else:
            assignments.append({
                "id": str(idx),
                "location": location_default,
                "task": task_name
            })
    status_msg = "success"
    if simulated:
        status_msg = "warning: generati valori randomici"
    return jsonify({
        "assignments": assignments,
        "pauses": pauses,
        "status": status_msg,
        "reason": reason
    })

# ---------------------------------------------------------------------------
# SOLVER AMPL + CPLEX con gestione errori
# ---------------------------------------------------------------------------

def getTimestamps(shift_hours):
    now_secs = datetime.now().hour * 3600 + datetime.now().minute * 60 + datetime.now().second
    for idx in range(len(shift_hours)):
        start_h = shift_hours[idx] * 3600
        end_h = (shift_hours[idx+1] * 3600) if idx+1 < len(shift_hours) else 24*3600
        if start_h <= now_secs < end_h:
            shift_start_secs = start_h
            shift_end_secs = end_h
            break
    return now_secs - shift_start_secs, shift_end_secs - shift_start_secs

def getK(tStart, tEnd):
    epsilon = 0.0001
    if tEnd != 0: return 1 / (1 - (tStart / tEnd) + epsilon)
    return 1

def solve_optimization(
    rows: int | None = None,
    cols: int | None = None,
    matrix: np.ndarray | None = None,
    assignment: List[int] | None = None,
) -> Any:
    global curr_assignment
    simulated = False
    reason = ""
    try:
        try:
            ampl = AMPL()
            ampl.read("model.mod")
            # Leggi solo parametri specifici dal file DAT
            ampl.readData("param_values.dat")
            logger.info("Parametri selettivi caricati da param_values.dat")
        except Exception as exc:
            last_status = f"error: caricamento parametri AMPL - {exc}"
            logger.exception(last_status)
            sys.exit(1)
        # Se non ho matrice passata, uso quella globale
        if matrix is None:
            matrix = pred_matrix
        # Se non ho assignment passata, uso globale
        if assignment is None:
            assignment = init_pos
        # Se ancora mancanti, segnalo simulazione
        if matrix is None or assignment is None:
            simulated = True
            reason = "mancano dati reali, uso valori simulati"
            rows, cols = len(Employee.total_employees()), len(Task.__members__)
            matrix = generate_stress_matrix(rows, cols, curr_assignment)
            if curr_assignment is None:
                curr_assignment = [1] * rows
        elif rows is None or cols is None:
            rows, cols = matrix.shape
        # Imposto j0 da assignment
        curr_assignment = assignment
        # Imposto set di AMPL
        emp_idx = list(range(1, rows + 1))
        dept_idx = list(range(cols))
        ampl.getSet("EMPLOYEES").setValues(emp_idx)
        ampl.getSet("DEPARTMENTS").setValues(dept_idx)
        ampl.getParameter("s").setValues(matrix)
        tStart_val, tEnd_val = getTimestamps(shift_hours)
        ampl.getParameter("k").set(getK(tStart_val, tEnd_val))
        ampl.getParameter("j0").setValues(assignment)
        ampl.option["solver"] = "cplex"
        try:
            # risolvo
            ampl.solve()
            code = int(ampl.getValue("solve_result_num"))
            text = ampl.getValue("solve_result")
            if code != 0:
                # se il solver non è OK, interrompo subito
                raise RuntimeError(f"Solver failed with status {code}: {text}")
            obj_value = ampl.getObjective("TotalCost").value()
            if obj_value == 0:
                # se l’obiettivo è zero, è un caso anomalo
                raise RuntimeError(f"Objective value is zero: {obj_value}")
            sol = np.zeros_like(matrix)
            for i in emp_idx:
                for j in dept_idx:
                    sol[i-1, j] = ampl.getVariable("x").get(i, j).value()
            curr_assignment = sol.argmax(axis=1).tolist()
            print(matrix)
            print(assignment)
            print(curr_assignment)
            print(getK(tStart_val, tEnd_val))
            print(tStart_val)
            print(tEnd_val)

            return build_schedule_response(sol, simulated, reason)

        except Exception as e:
            logger.exception("Errore generico AMPL")
            return jsonify({
                "assignments": [],
                "pauses": [],
                "status": f"error: {e}",
                "reason": "Errore generico AMPL"
            }), 500
    except ValueError as e:
        logger.exception("Errore in solve_optimization")
        return jsonify({
            "assignments": [],
            "pauses": [],
            "status": f"error: {e}",
            "reason": "Non è stato possibile generare randomicamente la matrice: nessun dato nell'API"
        }), 500
    except Exception as e:
        logger.exception("Errore in solve_optimization")
        return jsonify({
            "assignments": [],
            "pauses": [],
            "status": f"error: {e}",
            "reason": "Errore generico nella risoluzione"
        }), 500

# ---------------------------------------------------------------------------
# ENDPOINT FLASK
# ---------------------------------------------------------------------------

@app.route("/schedule", methods=["GET"])
def schedule_api():
    try:
        resps = launch_service()
        emp_list = build_employee_list(resps)
        pred_matrix, init_pos = build_prediction_matrix_and_initial_positions(emp_list)
        return solve_optimization(matrix=pred_matrix, assignment=init_pos)
    except Exception as e:
        logger.exception("Errore avvio servizio")
        sys.exit(1)
   

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
