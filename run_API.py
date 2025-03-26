"""
Questo codice implementa un'applicazione Flask che gestisce la schedulazione ottimizzata dei dipendenti in diversi reparti
di un'azienda. Utilizza un modello di ottimizzazione (risolto tramite AMPL e CPLEX) per assegnare i dipendenti ai reparti
(PRODUZIONE, FORNI, CONFEZIONAMENTO) o a una pausa, minimizzando lo stress complessivo.

Funzionalità principali:
1. Genera una matrice casuale che rappresenta lo stress dei dipendenti se assegnati a un reparto.
2. Risolve un problema di ottimizzazione per determinare l'assegnazione ottimale.
3. Fornisce un endpoint API (`/schedule`) che restituisce l'assegnazione aggiornata in formato JSON.

L'assegnazione corrente viene mantenuta come partenza per la chiamata successiva.
"""

from flask import Flask, jsonify
import numpy as np
from amplpy import AMPL
from enum import Enum
from datetime import datetime
import json
import requests
import argparse

# variabile globale per tenere traccia dell'ultimo risultato tra chiamate
curr_assignment = None # param j0
api = None # flag per usare chiamate api
location = "CAPANNONE NUOVO" # location dipendenti

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

# classe per rappresentare il dipendente (con i dati del turno e lo stress rilevato)
class Employee:
    def __init__(self, id_smartwatch, turno, stress=None):
        self.id_smartwatch = id_smartwatch
        self.stress = stress  # valore di stress rilevato dal device
        # dati relativi al turno
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
        # dizionario per previsioni dello stress per ogni reparto
        self.predicted_stresses = {}
    
    def __repr__(self):
        return (f"Employee(id_smartwatch={self.id_smartwatch}, id_turno={self.id_turno}, "
                f"giorno={self.giorno}, reparto={self.reparto}, stress={self.stress}, "
                f"predicted_stresses={self.predicted_stresses})")

app = Flask(__name__)

# generazione della matrice di stress senza API
def generate_stress_matrix(rows, cols, assignment):
    if assignment is not None:
        matrix = np.empty((0, cols))
        for elem in assignment:
            if elem == 0:
                row = np.full((1, cols), 60)
            else:
                row = np.random.randint(50, 100, (1, cols)) 
            matrix = np.vstack((matrix, row))
        matrix[:, 0] = 20 # colonna pausa
        return matrix
    else:
        return np.hstack((np.full((rows, 1), 10), np.random.randint(50, 100, size=(rows, cols - 1))))

# Funzione per creare la lista di Employee e calcolare la matrice di previsione dello stress
def build_employee_list(response):
    """
    Crea una lista di oggetti Employee a partire dalla risposta dei servzi API,
    e calcola la matrice s di previsione dello stress per ogni dipendente in ogni reparto.
    
    Parametri:
      api_response (list): Lista di dizionari ottenuti dalla chiamata API dello scheduling corrente.
      stress_data (dict): Dizionario con i dati di stress, con chiavi tipo "dev-sim-<id>".
      department_stress (list): Lista di dizionari contenenti lo stress medio per ciascun reparto.
    
    Ritorna:
      list: Lista di istanze Employee con l'attributo 'predicted_stresses' valorizzato.
    """
    # Creazione di un dizionario {idReparto: stress_medio}
    department_stress_dict = {dep["idReparto"]: dep["stress"] for dep in response[2]}
    
    employee_list = []
    for item in response[0]:
        id_smartwatch = item.get("idSmartwatch")
        turno = item.get("turno", {})
        # Recuperiamo lo stress misurato dal device
        key = f"dev-sim-{id_smartwatch}"
        stress = None
        if key in response[1]:
            measurements = response[1][key]
            if measurements:
                stress = measurements[-1].get("value")
        employee = Employee(id_smartwatch, turno, stress)
        employee_list.append(employee)
    
    # Calcolo della matrice di previsione
    # Per ciascun dipendente, calcoliamo il relativo scarto percentuale rispetto al reparto corrente
    # e lo applichiamo a ciascuna media dei reparti per stimare il livello di stress previsto.
    for emp in employee_list:
        current_avg = department_stress_dict.get(emp.id_reparto, None)
        if emp.stress is not None and current_avg is not None and current_avg != 0:
            relative_diff = (emp.stress - current_avg) / current_avg
        else:
            # Se manca qualche dato, non è possibile calcolare la variazione
            relative_diff = None
        
        # Per ogni reparto, se la variazione è calcolabile, stimiamo:
        # stress_previsto = media_reparto * (1 + relative_diff)
        # altrimenti, assegnamo il valore 1
        for dept_id, dept_avg in department_stress_dict.items():
            if relative_diff is not None:
                predicted = dept_avg * (1 + relative_diff)
            else:
                predicted = 50
            emp.predicted_stresses[dept_id] = predicted

    # Costruiamo e stampiamo la matrice s: per ogni dipendente mostriamo lo stress previsto in ogni reparto
    """
    print("Matrice di previsione dello stress per dipendente (chiave: idSmartwatch):")
    for emp in employee_list:
        print(f"Employee {emp.id_smartwatch} (Reparto corrente: {emp.reparto}): {emp.predicted_stresses}")
    """
    
    return employee_list

def launch_service():
    file_name = "service.json"
    # lettura dati
    with open(file_name, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    url_dict = json_data["url"]
    url_list = list(url_dict.items())

    # costruzione header
    cdclient = json_data["auth"]["cdclient"]
    origin = json_data["auth"]["origin"]
    token = json_data["auth"]["token"]

    # header richiesta
    headers = {
        "Cdclient": f"{cdclient}",
        "Origin": f"{origin}",
        "Authorization": f"Bearer {token}"
    }

    # orario corrente
    ora_corrente = datetime.now()

    day = ora_corrente.day
    month = ora_corrente.month
    year = ora_corrente.year
    hour = ora_corrente.hour
    minute = ora_corrente.minute

    # parametri richieste
    params_1_2 = {
        "day": f"{day}",
        "month": f"{month}",
        "year": f"{year}",
        "h": f"{hour}",
        "min": f"{minute}",
        "min": 36
    }

    params_3 = {
        "day": f"{day}",
        "month": f"{month}",
        "year": f"{year}",
        "h": f"{hour}",
        "min": f"{minute}",
        "numValues": 1
    }

    params_4 = {
        "day": f"{day}",
        "month": f"{month}",
        "year": f"{year}",
        "h": f"{hour}",
        "min": f"{minute}",
        "numValues": 1
    }

    for elem in Service:
        _, url = url_list[elem.value]
        match elem.value:
            case 0:
                response_reparti = requests.get(url, headers=headers)
            case 1:
                response_meanstr = requests.get(url, params=params_1_2, headers=headers)
               # print(response_meanstr.json())
            case 2:
                response_currsched = requests.get(url, params=params_1_2, headers=headers)
              # print(response_currsched.json())
            case 3:
                response_currstr = requests.post(url, params=params_3, headers=headers)
              # print(response_currstr.json())
            case 4:
                response_predstr = requests.post(url, params=params_4, headers=headers)

    return response_currsched.json(), response_currstr.json(), response_meanstr.json()

def build_prediction_matrix_and_initial_positions(employees):
    """
    Costruisce la matrice di previsione dello stress e il vettore delle posizioni iniziali.
    
    La matrice ha come colonne:
      - Colonna 0: Pausa (valore fisso 20)
      - Colonna 1: Produzione (reparto 1)
      - Colonna 2: Forni (reparto 2)
      - Colonna 3: Confezionamento (reparto 3)
      
    Il vettore delle posizioni iniziali contiene, per ogni dipendente, il codice del reparto corrente:
      1 per Pausa, 2 per Produzione, 3 per Forni, 4 per Confezionamento.
    
    Ritorna:
      - prediction_matrix: lista di liste (righe: dipendenti, colonne: reparti)
      - initial_positions: vettore delle posizioni iniziali per ciascun dipendente
    """
    # Ordine dei reparti come da mapping: 
    # Il reparto "pausa" viene aggiunto in prima posizione (colonna 0)
    # Le altre colonne seguono: produzione (id 1), forni (id 2), confezionamento (id 3)
    desired_dept_order = [1, 2, 3]
    
    prediction_matrix = []
    for emp in employees:
        # La colonna della pausa è fissa a 20
        row = [20]
        # Per ciascun reparto, prendi lo stress previsto; se non presente, assegna 1
        for dept in desired_dept_order:
            value = emp.predicted_stresses.get(dept, 1)
            row.append(value)
        prediction_matrix.append(row)
    
    initial_positions = []
    for emp in employees:
        # Confrontiamo in uppercase per sicurezza
        pos = Task[emp.reparto].value
        initial_positions.append(pos)
    
    # Stampa della matrice e del vettore delle posizioni iniziali
    """
    print("\nMatrice di previsione (righe: dipendenti, colonne: [PAUSA, PRODUZIONE, FORNI, CONFEZIONAMENTO]):")
    for row in prediction_matrix:
        print(row)
    
    print("\nVettore delle posizioni iniziali (uno per dipendente):")
    print(initial_positions)
    """
    
    return np.array(prediction_matrix), initial_positions

# costruzione della risposta alla richiesta di schedulazione nel formato richiesto
def build_schedule_response(matrix):
    global location 
    data = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        id = str(i+1)
        row = matrix[i].tolist()
        col = row.index(1) + 1
        for elem in Task:
            if elem.value == col:
                task = elem.name
                break
        if task == 'PAUSE':
            data[i][0] = id
            data[i][1] = task
            data[i][2] = ""
        else:
            data[i][0] = id
            data[i][1] = location
            data[i][2] = task 
    assignments = []
    pauses = []
    
    for row in data:
        if row[1] == "PAUSE":
            pauses.append({"id": row[0]})
        else:
            assignments.append({"id": row[0], "location": row[1], "task": row[2]})
    
    response = {"assignments": assignments, "pauses": pauses}
    return jsonify(response)

# Ottimizzazioni oltre la prima iterazione
def solve_optimization(rows=None, cols=None, matrix=None, assignment=None):
    """
    Genera una matrice casuale di numeri tra 0 e 100 rappresentanti la percentuale di stress previste per il dipendente i-esimo se assegnato al reparto j-esimo.
    Risolve il problema di ottimizzazione.
    Utilizza il valore precedente come parametro per la soluzione successiva.
    """

    global curr_assignment

    ampl = AMPL()

    # lettura modello
    ampl.read("paramfunction.mod")
    # lettura parametri
    ampl.readData("param_values.dat")

    if matrix is None: # non sto usando api
        rows, cols = 10, 4  # dipendenti, reparti
        matrix = generate_stress_matrix(rows, cols, curr_assignment)
        if curr_assignment is None: # assegnamento iniziale
            curr_assignment = [1 for _ in range(rows)]
            #val = ampl.getParameter("j0").get_values().toList()
            #curr_assignment = [elem for _, elem in val]
    else: # sto usando api
        if curr_assignment is None: # prima chiamata, lo schedule corrente non e' stato ancora assegnato
            curr_assignment = assignment

    # liste di indici per i set EMPLOYEES e DEPARTMENTS
    employees = list(range(1, rows + 1))
    departments = list(range(0, cols))
    # caricamento valori nei set
    employees_set = ampl.getSet("EMPLOYEES")
    employees_set.setValues(employees)
    departments_set = ampl.getSet("DEPARTMENTS")
    departments_set.setValues(departments)

    # caricamento matrice alpha come parametro del modello
    alpha_matrix = [[1 for _ in range(cols)] for _ in range(rows)]
    alpha = ampl.getParameter("alpha")
    for i in range(rows):
        for j in range(cols):
            alpha.set((i+1, j), alpha_matrix[i][j])

    # caricamento vettore Smax come parametro del modello
    smax_vec = [90 for _ in range(rows)]
    smax = ampl.getParameter("Smax")
    for i in range(rows):
        smax.set((i+1), smax_vec[i])

    # caricamento matrice s come parametro del modello
    s_matrix = ampl.getParameter("s")
    for i in range(rows):
        for j in range(cols):
            s_matrix.set((i+1, j), matrix[i, j])

    # caricamento assegnamento iniziale come parametro del modello
    j0_vec = ampl.getParameter("j0")
    for i in range(rows):
        j0_vec.set((i+1), curr_assignment[i])

    # selezione solver
    ampl.option["solver"] = "cplex"

    ampl.solve()

    # lettura risultati
    solution = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            solution[i, j] = ampl.getVariable("x").get(i + 1, j).value()

    response = build_schedule_response(solution)

    # costruzione nuovo assegnamento
    check = (solution == 1).any(axis=1)
    new_j0 = np.where(check, np.argmax(solution == 1, axis=1), 0)

    # aggiornamento j0 per la chiamata successiva
    curr_assignment = new_j0.tolist()

    return response

@app.route('/schedule', methods=['GET'])
def schedule():
    """
    Endpoint API che fornisce il
    nuovo assegnamento dei dipendeti ai reparti o alla pausa
    """
    global curr_assignment, api
    if api: # sto usando servizi api
        return solve_optimization(len(employees), len(prediction_matrix[0]), prediction_matrix, initial_positions)
    else:
        return solve_optimization()
    
parser = argparse.ArgumentParser(description="Argomento flag per chiamata API")
parser.add_argument("--api", action="store_true", help="Attiva la chiamata ai servizi tramite API")
args = parser.parse_args()

api = args.api
    
if api:
    employees = build_employee_list(launch_service())
    prediction_matrix, initial_positions = build_prediction_matrix_and_initial_positions(employees)

if __name__ == '__main__':
    app.run(debug=True)
