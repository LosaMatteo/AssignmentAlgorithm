#!/usr/bin/env python3
import time
import csv
import numpy as np
import os
from amplpy import AMPL

def solve_dat_file(dat_filename):
    """
    Legge il modello AMPL (paramfunction.mod) e i dati dal file .dat specificato,
    esegue il solver (CPLEX) e raccoglie le metriche:
      - valore dell'obiettivo (TotalCost)
      - matrice soluzione per la variabile decisionale x
      - nuovo vettore di assegnazione (calcolato come la posizione in cui compare il valore 1)
      - tempo di risoluzione
      - MIP gap (in percentuale) estratto dall'output del solver
    Restituisce una tupla con (nome_file, obj_value, solve_time, new_assignment, solution_matrix, mip_gap)
    """
    ampl = AMPL()
    # Carica il modello e i dati
    ampl.read("paramfunction.mod")
    ampl.readData(dat_filename)
    ampl.setOption('solver', 'cplex')

    start_time = time.time()
    ampl.solve()
    end_time = time.time()
    solve_time = end_time - start_time

    #mip_gap = ampl.getObjective("TotalCost").result()
    #print(f"MIP Gap: {mip_gap}")
    #if mip_gap == 'solved': mip_gap = 0

    # Verifica lo stato della soluzione
    status = ampl.get_value("solve_result")
    print(f"Stato: {status}")

    # Ottieni il MIP gap
    mip_gap = ampl.get_value("solve_result_num") # %
    print(f"MIP Gap: {mip_gap}%") 
    
    try:
        obj_value = ampl.getObjective("TotalCost").value()
    except Exception as e:
        print(f"Errore nel recupero del valore dell'obiettivo: {e}")
        obj_value = None

    # Recupera i set EMPLOYEES e DEPARTMENTS definiti nel modello
    try:
        employees = list(ampl.getSet("EMPLOYEES").getValues())
        departments = list(ampl.getSet("DEPARTMENTS").getValues())
    except Exception as e:
        print(f"Errore nel recupero dei set: {e}")
        employees, departments = [], []
    
    rows = len(employees)
    cols = len(departments)

    # Costruisce la matrice soluzione a partire dalla variabile x
    solution = np.zeros((rows, cols))
    try:
        x_var = ampl.getVariable("x")
        for i in range(rows):
            for j in range(cols):
                solution[i, j] = x_var.get(i + 1, j).value()
    except Exception as e:
        print(f"Errore durante la costruzione della matrice soluzione: {e}")

    # Calcola il nuovo vettore di assegnazione:
    check = (solution == 1).any(axis=1)
    new_assignment = np.where(check, np.argmax(solution == 1, axis=1), 0)

    return dat_filename, obj_value, solve_time, new_assignment, solution, mip_gap, status

def main():
    # Legge il file "dati.txt" che contiene (uno per riga) i nomi dei file .dat da processare
    try:
        with open("dati.txt", "r", encoding="utf-8") as f:
            dat_files = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print("Errore nella lettura di 'dati.txt':", e)
        return

    results = []
    for dat_file in dat_files:
        print("Processando:", dat_file)
        try:
            res = solve_dat_file(dat_file)
            results.append(res)
            print(f"File {dat_file} processato. Obiettivo: {res[1]}, Tempo: {res[2]:.2f}s, MIP Gap: {res[5]}%")
        except Exception as e:
            print(f"Errore durante l'elaborazione di {dat_file}: {e}")
            results.append((dat_file, "Error", None, None, None, None))
    
    # Scrive i risultati in "risultati.csv"
    with open("risultati.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Intestazioni CSV: File, Stato Soluzione, Valore Obiettivo, Tempo di Risoluzione, MIP Gap, Nuova Assegnazione, Matrice Soluzione
        if os.path.getsize(csvfile.name) == 0:
            writer.writerow(["File", "State", "ObjectiveValue", "SolveTime(s)", "MIPGap%", "NewAssignment", "SolutionMatrix"])
        for entry in results:
            file_name, obj_value, solve_time, new_assignment, solution_matrix, mip_gap, state = entry
            new_assignment_str = str(new_assignment.tolist()) if new_assignment is not None else ""
            solution_str = str(solution_matrix.tolist()) if solution_matrix is not None else ""
            writer.writerow([file_name, state, obj_value, solve_time, mip_gap, new_assignment_str, solution_str])
            print("Tutti i risultati sono stati scritti in 'risultati.csv'.")

    with open('dati.txt', 'w') as file:
        file.write('')

if __name__ == '__main__':
    main()
