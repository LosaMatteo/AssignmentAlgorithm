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

class Task(Enum):
    PAUSE = 1
    PRODUZIONE = 2
    FORNI = 3
    CONFEZIONAMENTO = 4

app = Flask(__name__)

# variabile globale per tenere traccia dell'ultimo risultato tra chiamate
curr_assignment = None # param j0
location = "CAPANNONE NUOVO"

# generazione della matrice di stress
def generate_stress_matrix(rows, cols, assignment):
    if assignment is not None:
        matrix = np.empty((0, cols))
        for elem in assignment:
            if elem == 0:
                row = np.full((1, cols), 50)
            else:
                row = np.random.randint(20, 100, (1, cols)) 
            matrix = np.vstack((matrix, row))
        matrix[:, 0] = 10 # colonna pausa
        return matrix
    else:
        return np.hstack((np.full((rows, 1), 10), np.random.randint(20, 100, size=(rows, cols - 1))))



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

def solve_optimization(previous_value=None):
    """
    Genera una matrice casuale di numeri tra 0 e 100 rappresentanti la percentuale di stress previste per il dipendente i-esimo se assegnato al reparto j-esimo.
    Risolve il problema di ottimizzazione.
    Utilizza il valore precedente come parametro per la soluzione successiva.
    """
    global curr_assignment

    rows, cols = 5, 4  # dipendenti, reparti
    #matrix = np.random.randint(10, 101, (rows, cols))  # generazione matrice casuale s
    matrix = generate_stress_matrix(rows, cols, curr_assignment)
   
    ampl = AMPL()

    # lettura modello
    ampl.read("paramfunction.mod")
    # lettura parametri
    ampl.readData("param_values.dat")

    if curr_assignment is None: # assegnamento iniziale
        val = ampl.getParameter("j0").get_values().toList()
        curr_assignment = [elem for _, elem in val]

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
            s.set((i+1, j), matrix[i, j])

    # caricamento assegnamento iniziale come parametro del modello
    j0 = ampl.getParameter("j0")
    for i in range(rows):
        j0.set((i+1), curr_assignment[i])

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
    curr_assignment = new_j0.tolist()

    return response

@app.route('/schedule', methods=['GET'])
def schedule():
    """
    Endpoint API che fornisce il
    nuovo assegnamento dei dipendeti ai reparti o alla pausa
    """
    global curr_assignment
    return solve_optimization(curr_assignment)

if __name__ == '__main__':
    app.run(debug=True)
