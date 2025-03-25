import requests
from flask import Flask, jsonify
import numpy as np
from amplpy import AMPL
from enum import Enum
class Task(Enum):
    PAUSE = 1
    PRODUZIONE = 2
    FORNI = 3
    CONFEZIONAMENTO = 4
app = Flask(__name__)
# variabile globale per tenere traccia dell'ultimo risultato tra chiamate
curr_assignment = None # param j0
location = "CAPANNONE NUOVO"
def getUrl():
    # URL
    url_lista_1 = "https://target-ai.revelis.eu/gestu-service/api/gen-turni/list-reparti"  # lista reparti
    url_meanst_2 = "https://target-ai.revelis.eu/gestu-service/api/gen-turni/avg-stress"  # stress medio per reparto
    url_attuale_3 = "https://target-ai.revelis.eu/gestu-service/api/gen-turni/turni-da-data-ora"  # scheduling attuale + tempo trascorso
    url_lettura_4 = "https://target-ai.revelis.eu/gestu-service/api/gen-turni/stress-values"  # lettura stress dai device
    url_forecast_5 = "https://target-ai.revelis.eu/gestu-service/api/gen-turni/stress-predictions"  # previsioni stress per device
    with open("token.txt", "r", encoding="utf-8") as file: token = file.readline().strip()
    # Header personalizzato
    headers = {
        "Cdclient": "target-wfm",
        "Origin": "https://target-ai.revelis.eu",
        "Authorization": "Bearer "+token
    }

    # Parametri per le richieste
    params_2_3 = {
        "day": 24,
        "month": 3,
        "year": 2025,
        "h": 11,
        "min": 00
    }

    params_4 = {
        "day": 24,
        "month": 3,
        "year": 2025,
        "h": 11,
        "min": 00,
        "numValues": 1
    }

    params_5 = {
        "day": 24,
        "month": 3,
        "year": 2025,
        "h": 11,
        "min": 30,
        "numValues": 1
    }

    # Invio delle richieste
    response_lista = requests.get(url_lista_1, headers=headers)
    response_meanst = requests.get(url_meanst_2, params=params_2_3, headers=headers)
    response_attuale = requests.get(url_attuale_3, params=params_2_3, headers=headers)
    response_lettura = requests.post(url_lettura_4, params=params_4, headers=headers)
    response_forecast = requests.post(url_forecast_5, params=params_5, headers=headers)

    # Stampare le risposte per verificare
    print("Status Code lista reparti:", response_lista.status_code)
    print("Response Body lista reparti:", response_lista.json())

    print("Status Code stress medio:", response_meanst.status_code)
    print("Response Body stress medio:", response_meanst.json())

    print("Status Code scheduling attuale:", response_attuale.status_code)
    print("Response Body scheduling attuale:", response_attuale.json())

    print("Status Code lettura stress:", response_lettura.status_code)
    print("Response Body lettura stress:", response_lettura.json())

    print("Status Code forecast stress:", response_forecast.status_code)
    print("Response Body forecast stress:", response_forecast.json())

    # Ritorniamo lo scheduling e i dati di stress
    return response_attuale.json(), response_lettura.json(), response_meanst.json()

# Classe per rappresentare il dipendente (con i dati del turno e lo stress rilevato)
class Employee:
    def __init__(self, id_smartwatch, turno, stress=None):
        self.id_smartwatch = id_smartwatch
        self.stress = stress  # valore di stress rilevato dal device
        # Dati relativi al turno
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
        # Questo dizionario conterrà le previsioni dello stress per ogni reparto
        self.predicted_stresses = {}
    
    def __repr__(self):
        return (f"Employee(id_smartwatch={self.id_smartwatch}, id_turno={self.id_turno}, "
                f"giorno={self.giorno}, reparto={self.reparto}, stress={self.stress}, "
                f"predicted_stresses={self.predicted_stresses})")

# Funzione per creare la lista di Employee e calcolare la matrice di previsione dello stress
def create_employee_list(api_response, stress_data, department_stress):
    """
    Crea una lista di oggetti Employee a partire dalla risposta dello scheduling e dai dati di stress,
    e calcola la matrice s di previsione dello stress per ogni dipendente in ogni reparto.
    
    Parametri:
      api_response (list): Lista di dizionari ottenuti dalla chiamata API dello scheduling.
      stress_data (dict): Dizionario con i dati di stress, con chiavi tipo "dev-sim-<id>".
      department_stress (list): Lista di dizionari contenenti lo stress medio per ciascun reparto.
    
    Ritorna:
      list: Lista di istanze Employee con l'attributo 'predicted_stresses' valorizzato.
    """
    # Creazione di un dizionario {idReparto: stress_medio}
    department_stress_dict = {dep["idReparto"]: dep["stress"] for dep in department_stress}
    
    employee_list = []
    for item in api_response:
        id_smartwatch = item.get("idSmartwatch")
        turno = item.get("turno", {})
        # Recuperiamo lo stress misurato dal device
        key = f"dev-sim-{id_smartwatch}"
        stress = None
        if key in stress_data:
            measurements = stress_data[key]
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
                predicted = 1
            emp.predicted_stresses[dept_id] = predicted

    # Costruiamo e stampiamo la matrice s: per ogni dipendente mostriamo lo stress previsto in ogni reparto
    print("Matrice di previsione dello stress per dipendente (chiave: idSmartwatch):")
    for emp in employee_list:
        print(f"Employee {emp.id_smartwatch} (Reparto corrente: {emp.reparto}): {emp.predicted_stresses}")
    
    return employee_list

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
    
    # Mappatura per la posizione iniziale:
    # Consideriamo che se l'employee è in "PRODUZIONE" -> posizione 2, "FORNI" -> posizione 3, "CONFEZIONAMENTO" -> posizione 4.
    # Se il reparto non è tra questi, si assume la pausa (1)
    dept_mapping = {
        "PRODUZIONE": 2,
        "FORNI": 3,
        "CONFEZIONAMENTO": 4
    }
    initial_positions = []
    for emp in employees:
        # Confrontiamo in uppercase per sicurezza
        pos = dept_mapping.get(emp.reparto.upper(), 1)
        initial_positions.append(pos)
    
    # Stampa della matrice e del vettore delle posizioni iniziali
    print("\nMatrice di previsione (righe: dipendenti, colonne: [PAUSA, PRODUZIONE, FORNI, CONFEZIONAMENTO]):")
    for row in prediction_matrix:
        print(row)
    
    print("\nVettore delle posizioni iniziali (uno per dipendente):")
    print(initial_positions)
    
    return prediction_matrix, initial_positions

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

def solve_optimization(rows, cols, matrix, initial_assignment):
    """
    Genera una matrice casuale di numeri tra 0 e 100 rappresentanti la percentuale di stress previste per il dipendente i-esimo se assegnato al reparto j-esimo.
    Risolve il problema di ottimizzazione.
    Utilizza il valore precedente come parametro per la soluzione successiva.
    """
    ampl = AMPL()

    # lettura modello
    ampl.read("paramfunction.mod")
    # lettura parametri
    ampl.readData("param_values.dat")
    # liste di indici per i set EMPLOYEES e DEPARTMENTS
    employees = list(range(1, rows + 1))
    departments = list(range(0, cols))
    # caricamento valori nei set
    employees_set = ampl.getSet("EMPLOYEES")
    employees_set.setValues(employees)
    departments_set = ampl.getSet("DEPARTMENTS")
    departments_set.setValues(departments)

    # caricamento matrice s come parametro del modello
    s = ampl.getParameter("s")
    for i in range(rows):
        for j in range(cols):
            s.set((i+1, j), matrix[i][j])
    print(s)
    # caricamento assegnamento iniziale come parametro del modello
    j0 = ampl.getParameter("j0")
    print(rows)
    print(initial_assignment)
    for i in range(rows):
        print(i)
        j0.set((i+1), initial_assignment[i])

    # selezione solver
    ampl.option["solver"] = "cplex"

    ampl.solve()

    # lettura risultati
    #z_value = ampl.getObjective("TotalCost").value()
    solution = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            solution[i, j] = ampl.getVariable("x").get(i + 1, j).value()

    response = build_schedule_response(solution)
    print(solution)

    # costruzione nuovo assegnamento
    check = (solution == 1).any(axis=1)
    new_j0 = np.where(check, np.argmax(solution == 1, axis=1), 0)

    #old_assignment = curr_assignment
    # aggiornamento j0 per la chiamata successiva

    return response

@app.route('/schedule', methods=['GET'])
def schedule():
    """
    Endpoint API che fornisce il
    nuovo assegnamento dei dipendeti ai reparti o alla pausa
    """
    global curr_assignment
    return solve_optimization(len(employees), len(prediction_matrix[0]), prediction_matrix, initial_positions)

# Utilizzo delle funzioni
response_attuale_json, response_lettura_json, response_meanst_json = getUrl()

# Creiamo la lista di oggetti Employee includendo l'informazione dello stress e le previsioni per ogni reparto
employees = create_employee_list(response_attuale_json, response_lettura_json, response_meanst_json)

prediction_matrix, initial_positions = build_prediction_matrix_and_initial_positions(employees)
# Stampa per verificare il risultato finale
if __name__ == '__main__':
    app.run(debug=True)