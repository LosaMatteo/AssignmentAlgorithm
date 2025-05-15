"""
Questo modulo implementa un'applicazione Flask per l'ottimizzazione della pianificazione
dei turni dei dipendenti. Interagisce con servizi API esterni per recuperare dati
sui dipendenti, i loro turni e i livelli di stress attuali e previsti.
Utilizza AMPL per risolvere un problema di ottimizzazione al fine di assegnare
i dipendenti ai reparti minimizzando lo stress complessivo, sotto certi vincoli.

NOTA: Il programma necessita di un file esterno, 'service.json', in cui sono presenti
      le informazioni necessarie per l'utilizzo dei servizi API.

Funzionalita' principali:
- Recupero dati da API esterne (turni, stress corrente, stress medio per reparto, stress predetto).
- Modellazione dei dati dei dipendenti (stress predetto, reparto lavorativo attuale).
- Risoluzione del problema di ottimizzazione tramite AMPL per trovare l'assegnamento dei dipendenti
  ai reparti che minimizza lo stress totale.
- Endpoint Flask per ottenere la pianificazione ottimizzata.
"""

from __future__ import annotations

import os
import json
import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import requests
from amplpy import AMPL
from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENUMS per task e servizi e costanti
# ---------------------------------------------------------------------------

DEFAULT_LOCATION = "CAPANNONE NUOVO"
SHIFT_HOURS = [0, 8, 16]
ORDERED_DEPT_NAMES = ["PRODUZIONE", "FORNI", "CONFEZIONAMENTO"]
MODEL_FILE = "model.mod"
DATA_FILE = "param_values.dat"
SERVICE_FILE = "service.json"

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
    """Rappresenta un dipendente con informazioni sul turno, reparto e stress attuale e previsti"""
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
        """Imposta la pausa a un dipendente"""
        self.reparto = Task.PAUSE.name
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
used_extimated_stress = False # flag per segnalare l'utilizzo di valori di stress approssimativi

def check_files_exist() -> None:
    """Funzione che verifica la presenza dei file necessari per il funzionamento
       del programma, in particolare i file del modello (.mod), valori dei paramwtri (.dat)
       e il file per l'utilizzo dei servizi (.json). Termina l'applicazione se i file non
       vengono trovati.
    """
    files_to_check = [MODEL_FILE, DATA_FILE, SERVICE_FILE]
    missing_files = []
    for f_path in files_to_check:
        if not os.path.isfile(f_path):
            missing_files.append(f_path)

    if missing_files:
        logger.critical(f"File critici mancanti: {', '.join(missing_files)}. L'applicazione non puo' avviarsi.")
        sys.exit(1)

def launch_service() -> Tuple[Any, Any, Any, Any]:
    """Gestisce la chiamata ai servizi API e la lettura delle rispettive risposte,
       insieme alla gestione degli errori.
        
    La funzione restituisce la tupla contenente le risposte dei singoli servizi.
    Se non e' possibile ricevere una risposta dai servizi, viene segnalato l'errore.

    :return: Insieme delle risposte dei servizi API.
    :rtype: Tuple[Dict[Service]]
    """
    try:
        with open(SERVICE_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.error("Error while reading service.json: %s", e)
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
            logger.error("API error %s in %s: %s", svc.name, url, e)

    return (
        responses[Service.CURR_SCHED],
        responses[Service.CURR_STR],
        responses[Service.MEAN_STR],
        responses[Service.PRED_STR]
    )

def build_employee_list(resps: Tuple[Any, Any, Any, Any]) -> List[Employee]:
    """Restituisce la lista degli oggetti Employee in base ai dati restituiti dai servizi.

    Se la risposta riguardante lo stress medio per reparto (mean_str) è vuota,
    il programma termina. Il programma terminera' anche se la lista finale
    dei Se sono mancanti dei dati riguardanti lo stress, questi verranno stimati
    per poter proseguire con la soluzione del problema.
    
    NOTA: Non viene gestito il caso in cui, nell'assegnazioen corrente, un cliente sia
          in pausa perche' non ancora previsto dai servizi API utilizzati.

    :param resps: Risposte dei servizi API.
    :type resps: Tuple[Dict[Service]]
    :raises ValueError: Se la lista 'mean_str' o la lista 'emp_list' e' vuota
                        o se c'è' un'inconsistenza nell'id del device.
    :return: Insieme dei dipendenti presenti.
    :rtype: List[Employee]
    """
    global used_extimated_stress
    curr_sched, curr_str, mean_str, pred_str = resps
    if not mean_str:
        raise ValueError("Empty average stress list. Check API response body")
    dept_avg = {d["idReparto"]: d["stress"] for d in mean_str}
    emp_list: List[Employee] = []
    for rec in curr_sched:
        sw_id = rec.get("idSmartwatch")
        turno = rec.get("turno", {})
        key = f"dev-sim-{sw_id}"
        stress_val = None
        if key in curr_str:
            readings = curr_str.get(key, []) # legge lo stress corrente del dipendente
            if readings:
                stress_val = readings[-1].get("value")
            else:
                # se non e' stato possibile leggere lo stress corrente,
                # viene assegnato lo stress medio del reparto del dipendente
                used_extimated_stress = True # attiva la flag per segnalare l'utilizzo di dati non completi
                stress_val = dept_avg.get(turno.get("idReparto"))
        else:
            raise ValueError("Unexpected mismatch in device ids. Check API response.")
        emp = Employee(sw_id, turno, stress_val) # crea l'istanza del dipendente
        emp_list.append(emp)
    if not emp_list:
        raise ValueError("Empty employee list. Check API response body")
    for emp in emp_list:
        # riempe il campo dello stress predetto, relativo a ongi reparto, per ogni dipendete
        fill_predicted_stress(emp, pred_str, dept_avg)
    return emp_list

def fill_predicted_stress(emp: Employee, pred_str: dict, dept_str: dict) -> None:
    """Ottiene o calcola lo stress predetto relativo a un dipendente per ogni reparto.

    Se c'e' un'incosistenza nei valori del device id il programma terinera'. 

    :param emp: Dipendente di cui va calcolato lo stress predetto.
    :type emp: Employee
    :param pred_str: Dati sullo stress predetto restituiti dal servizio API.
    :type pred_str: dict
    :param dept_str: Dati sullo stress medio dei reparti.
    :type dept_str: dict
    :raises ValueError: Se la chiave del device non viene trovata nel dizionario
                        dello stress predetto o se lo stress medio del reparto corrente
                        non ha un valore.
    """
    global used_extimated_stress
    curr_avg = dept_str.get(emp.id_reparto) # stress medio del reparto del dipendente
    if not curr_avg or curr_avg == 0:
        raise ValueError(f"Invalid average stress value for department {emp.id_reparto}")
    # scarto dello stress del dipendete rispetto alla media del reparto
    rel_diff = (emp.stress - curr_avg) / curr_avg 
    for rid, avg in dept_str.items():
        # Se il reparto e' quello in cui si trova attualmente il dipendente
        # provo a utilizzare lo stress predetto fornito dal servizio API.
        # Altrimenti, se il reparto d' diverso o non e' possibile ottenere
        # lo stress predetto, questo viene stimato imputando lo scarto relativo dello stress
        # anche agli altri reparti
        if rid == emp.id_reparto:
            key = f"dev-sim-{emp.id_smartwatch}"
            if key in pred_str:
                readings = pred_str.get(key, [])
                if readings:
                    stress = readings[-1].get("value")
                    emp.predicted_stresses[rid] = stress
                else:
                    used_extimated_stress = True # attiva la flag per segnalare l'utilizzo di dati non completi
                    emp.predicted_stresses[rid] = avg * (1 + (rel_diff or 0))
            else:
                raise ValueError("Unexpected mismatch in device ids. Check API response.")
        else:
            emp.predicted_stresses[rid] = avg * (1 + (rel_diff or 0))       
    return  

def build_prediction_matrix_and_initial_positions(
    emp_list: List[Employee]
) -> Tuple[np.ndarray, List[int]]:
    """Costruisce la matrice dello stress predetto e la lista rappresentante 
       l'assegnamento corrente ai reparti a partire dalla lista dei dipendenti.

    :param emp_list: Lista dei dipendenti presenti.
    :type emp_list: List[Employee]
    :return: Matrice contenente lo stress previsto per ogni dipendente per ogni reparto
             insieme alla lista contenente gli assegnamenti correnti ai reparti.
    :rtype Tuple[np.ndarray, List[int]
    """
    name_to_id = {e.reparto.upper(): e.id_reparto for e in emp_list if e.reparto}
    ordered_ids: List[Optional[int]] = [name_to_id.get(n) for n in ORDERED_DEPT_NAMES]
    matrix: List[List[float]] = []
    for emp in emp_list:
        # legge lo stress predetto del dipendente per ogni reparto
        rep_values = [emp.predicted_stresses.get(rid, emp.stress) if rid is not None else emp.stress for rid in ordered_ids]
        # lo stress predetto per la pausa e' il 60% dello stress medio del dipendente
        pause_val = 0.6 * float(np.mean(rep_values))
        matrix.append([pause_val] + rep_values)
    j0: List[int] = [] # vettore degli assegnamenti attuali
    j0 = [Task[emp.reparto].value - 1 if emp.reparto and emp.reparto in Task.__members__ else 0 for emp in emp_list]
    return np.array(matrix, dtype=float), j0

def build_schedule_response(sol: np.ndarray, reason: str | None) -> Any:
    """Costruisce la struttura JSON da restituire al client come risultato
       del problema. Riporta gli assegnamenti dei dipendenti e messaggi di 
       status.

    :param sol: Soluzione del problema di ottimizzazione.
    :type sol: np.darray
    :param reason: Messaggio con eventuali warning da segnalare al client.
    :type reason: str | None
    :return: Oggetto json contenente la risposta.
    :rtype: Any
    """    
    global used_extimated_stress
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
                "location": DEFAULT_LOCATION,
                "task": task_name
            })
    status_msg = "success"
    if used_extimated_stress:
        status_msg = "solved - WARNING: utilizzati valori di stress meno accurati"
        used_extimated_stress = False # reset della flag per la chiamata successiva
    return jsonify({
        "assignments": assignments,
        "pauses": pauses,
        "status": status_msg,
        "reason": reason
    })

def get_timestamps() -> Tuple[int, int]:
    """Ottiene il timestamp corrente e di fine turno da utilizzare per la soluzione del problema.

    :return: Timestamp relativo all'istante della chiamata e timestamp relativo alla fine
             del turno corrente.
    :rtype: Tuple[int, int]
    """
    now_secs = datetime.now().hour * 3600 + datetime.now().minute * 60 + datetime.now().second
    for idx in range(len(SHIFT_HOURS)):
        start_h = SHIFT_HOURS[idx] * 3600
        end_h = (SHIFT_HOURS[idx+1] * 3600) if idx + 1 < len(SHIFT_HOURS) else 24*3600
        if start_h <= now_secs < end_h:
            shift_start_secs = start_h
            shift_end_secs = end_h
            break
    return now_secs - shift_start_secs, shift_end_secs - shift_start_secs # type: ignore

def get_k(tStart: int, tEnd: int) -> float:
    """Calcola il valore del parametro del modello che tiene conto dell'istante
       in cui e' stata effettuata la chiamata al servizio rispetto alla fine del turno.

    Restituisce il valore 1 se il parametro tEnd e' uguale a 0.

    :param tStart: Timestamp relativo all'istante della chiamata al servizio.
    :type tStart: int
    :param tEnd: Timestamp relativo alla fine del turno attuale.
    :type tEnd: int
    :return: Valore della funzione 1 / (1 - (tStart / tEnd) + epsilon)
    :rtype: float
    """
    epsilon = 0.0001
    if tEnd != 0: return 1 / (1 - (tStart / tEnd) + epsilon)
    raise ValueError("Invalid value for variable tEnd")

def solve_optimization(
    matrix: np.ndarray,
    assignment: List[int],
) -> Any:
    """Risolve il problema di ottimizzazione attraverso la libreria AMPL e
       il risolutore cplex.

    Il programma termina se viene rilevata un'inconsistenza nei dati peraparati
    per il modello o se viene rilevato un problema nella risoluzione del problema.

    :param matrix: Matrice relativa allo stress predetto per ogni dipendente in ogni reparto.
    :type matrix: np.darray
    :param assignment: Lista contenente l'assegnazione corrente ai reparti.
    :type assignment: List[int]
    :raises ValueError: Se La matrice di stress o il vettore degli assegnamenti e' vuoto.
    :raises RuntimeError: Se c'e' stato un problema nella risoluzione del problema di ottimizzazione.
    :return: Struttura JSON da restituire al client.
    :rtype: Any
    """    
    global used_extimated_stress
    msg = None
    if not matrix.size or not assignment:
        logger.error("Unexpected error. Either sress matrix or current assignment is none")
        return jsonify({
            "assignments": [],
            "pauses": [],
            "status": "error: Dati di input non validi",
            "reason": "La matrice di stress o l'assegnamento corrente sono vuoti."
        }), 500
    try:
        ampl = AMPL()
        ampl.read(MODEL_FILE)
        # lettura dei parametri necessari dal file DAT
        ampl.readData(DATA_FILE)
        if used_extimated_stress:
            msg = "Missing values"
            logger.info(msg)
        rows, cols = matrix.shape
        # caricamento parametri del modello
        emp_idx = list(range(1, rows + 1))
        dept_idx = list(range(cols))
        ampl.getSet("EMPLOYEES").setValues(emp_idx)
        ampl.getSet("DEPARTMENTS").setValues(dept_idx)            
        ampl.getParameter("s").setValues(matrix)
        tStart_val, tEnd_val = get_timestamps()
        ampl.getParameter("k").set(get_k(tStart_val, tEnd_val))
        ampl.getParameter("j0").setValues(assignment)
        ampl.option["solver"] = "cplex"
        try:
            # risolvo
            ampl.solve()
            code = int(ampl.getValue("solve_result_num"))
            text = ampl.getValue("solve_result")
            if code != 0:
                # se il solver non e' OK, interrompo subito
                raise RuntimeError(f"Solver failed with status {code}: {text}")
            obj_value = ampl.getObjective("TotalCost").value()
            if obj_value == 0:
                # se l’obiettivo e' zero, e' un caso anomalo
                logger.warning(f"Objective value is zero: {obj_value}")
            sol = np.zeros_like(matrix) # prepara la soluzione da mostrare
            for i in emp_idx:
                for j in dept_idx:
                    sol[i-1, j] = ampl.getVariable("x").get(i, j).value()

            return build_schedule_response(sol, msg)

        except Exception as e:
            logger.exception("Error while resolving")
            return jsonify({
                "assignments": [],
                "pauses": [],
                "status": f"error: {e}",
                "reason": "Errore durante la risoluzione"
            }), 500
        
    except Exception as e:
        logger.exception("Unexpected error in solve_optimization")
        return jsonify({
            "assignments": [],
            "pauses": [],
            "status": f"error: {e}",
            "reason": "Errore imprevisto durante la risoluzione"
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
        return solve_optimization(pred_matrix, init_pos)
    
    except ValueError as ve:
        logger.error(f"Input data error: {ve}")

        return jsonify({
            "assignments": [],
            "pauses": [],
            "status": "error: Dati di input invalidi o problema con API esterne",
            "reason": str(ve)
        }), 500
        
    except RuntimeError as re:
        logger.error(f"Runtime error while resolving: {re}")
        
        return jsonify({
            "assignments": [],
            "pauses": [],
            "status": "error: Problema durante la risoluzione del problema",
            "reason": str(re)
        }), 500
    except Exception as e:
        logger.exception(f"Unhandled error in endpoint /schedule: {e}")

        return jsonify({
            "assignments": [],
            "pauses": [],
            "status": "error: Errore interno del server",
            "reason": "Si è verificato un errore imprevisto durante l'elaborazione della richiesta."
        }), 500

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_files_exist()
    app.run(debug=True)
