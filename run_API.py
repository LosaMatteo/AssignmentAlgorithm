"""
- python -m pip install amplpy --upgrade
- python -m amplpy.modules install cplex
- pip install flask numpy
Lanciare:
- python run_API.py
In un altro terminale:
- curl http://127.0.0.1:5000/solve
"""

from flask import Flask, jsonify
import numpy as np
from amplpy import AMPL

app = Flask(__name__)

# variabile globale per tenere traccia dell'ultimo risultato tra chiamate
curr_assignment = [1, 1, 1, 1, 1] # param j0

def solve_optimization(previous_value=None):
    """
    Genera una matrice casuale di numeri tra 0 e 100 rappresentanti la percentuale di stress del dipendente i-esimo se assegnato al reparto j-esimo.
    Risolve il problema di ottimizzazione.
    Utilizza il valore precedente come parametro per la soluzione successiva.
    """
    global curr_assignment

    rows, cols = 5, 4  # dipendenti, reparti
    matrix = np.random.randint(0, 101, (rows, cols))  # generazione matrice casuale s
    #employees va da 1 a 5, departments da 0 a 3
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
            s.set((i+1, j), matrix[i, j])

    # caricamento assegnamento iniziale come parametro del modello
    j0 = ampl.getParameter("j0")
    for i in range(rows):
        j0.set((i+1), curr_assignment[i])

    # selezione solver
    ampl.option["solver"] = "cplex"

    ampl.solve()

    # lettura risultati
    z_value = ampl.getObjective("TotalCost").value()
    solution = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            solution[i, j] = ampl.getVariable("x").get(i + 1, j).value()

    # costruzione nuovo assegnamento
    check = (solution == 1).any(axis=1)
    new_j0 = np.where(check, np.argmax(solution == 1, axis=1), 0)

    old_assignment = curr_assignment
    # aggiornamento j0 per la chiamata successiva
    curr_assignment = new_j0.tolist()

    return solution.tolist(), z_value, old_assignment

@app.route('/solve', methods=['GET'])
def solve():
    """
    Endpoint API che risolve il problema
    e mantiene il risultato per fornirlo come parametro al modello
    nelle chiamate successive.
    """
    global curr_assignment
    solution, result, old_pos= solve_optimization(curr_assignment)

    return jsonify({
        "old_pos": old_pos,
        "solution": solution,
        "new_pos": curr_assignment,
        "opt_value": result
    })

if __name__ == '__main__':
    app.run(debug=True)
