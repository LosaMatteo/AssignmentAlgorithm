#!/usr/bin/env python3
import time
import csv
import numpy as np
from amplpy import AMPL

def solve_dat_file(dat_filename):
    ampl = AMPL()
    ampl.read("paramfunction.mod")
    ampl.readData(dat_filename)
    ampl.setOption('solver', 'cplex')
    ampl.setOption('cplex_options', 'timelimit=600')

    start_time = time.time()
    ampl.solve()
    end_time = time.time()
    solve_time = end_time - start_time

    status = ampl.get_value("solve_result")
    print(f"Stato: {status}")

    mip_gap = ampl.get_value("solve_result_num")
    print(f"MIP Gap: {mip_gap}%")

    try:
        obj_value = ampl.getObjective("TotalCost").value()
    except Exception as e:
        print(f"Errore nel recupero del valore dell'obiettivo: {e}")
        obj_value = None

    try:
        employees = list(ampl.getSet("EMPLOYEES").getValues())
        departments = list(ampl.getSet("DEPARTMENTS").getValues())
    except Exception as e:
        print(f"Errore nel recupero dei set: {e}")
        employees, departments = [], []

    rows = len(employees)
    cols = len(departments)

    # Costruzione della matrice x e del vettore j0 (output)
    solution = np.zeros((rows, cols))
    j0_output = []
    try:
        x_var = ampl.getVariable("x")
        for i in range(rows):
            found = False
            for j in range(cols):
                val = x_var.get(i + 1, j).value()
                solution[i, j] = val
                if not found and val == 1:
                    j0_output.append(j)
                    found = True
            if not found:
                j0_output.append(-1)
    except Exception as e:
        print(f"Errore durante il recupero della soluzione x o del vettore j0 (output): {e}")
        solution = None
        j0_output = [-1] * rows

    # Lettura del vettore j0 di input
    try:
        j0_param = ampl.getParameter("j0")
        j0_input = [int(j0_param.get(i + 1)) for i in range(rows)]
    except Exception as e:
        print(f"Errore nel recupero del parametro 'j0' dal file .dat: {e}")
        j0_input = [-1] * rows

    # Calcolo delle nuove metriche:
    num_pausa = sum(1 for j in j0_output if j == 0)
    num_trasferimenti = sum(
        1 for old, new in zip(j0_input, j0_output)
        if old > 0 and new > 0 and old != new
        )


    try:
        onpause = ampl.getParameter("onpause").value()
    except Exception as e:
        print(f"Errore nel recupero di 'onpause': {e}")
        onpause = None

    try:
        tStart = ampl.getParameter("tStart").value()
    except Exception as e:
        print(f"Errore nel recupero di 'tStart': {e}")
        tStart = None

    try:
        T_param = ampl.getParameter("T")
        T_size = cols - 1
        T_matrix = np.zeros((T_size, T_size))
        for i in range(T_size):
            for j in range(T_size):
                T_matrix[i, j] = T_param.get(i + 1, j + 1)
    except Exception as e:
        print(f"Errore nel recupero della matrice T: {e}")
        T_matrix = None

    return (
        dat_filename, obj_value, solve_time, solution,
        j0_output, j0_input, mip_gap, onpause, tStart, T_matrix,
        num_pausa, num_trasferimenti
    )

def main():
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
            print(f"File {dat_file} processato. Obiettivo: {res[1]}, Tempo: {res[2]:.2f}s, MIP Gap: {res[6]}")
        except Exception as e:
            print(f"Errore durante l'elaborazione di {dat_file}: {e}")
            results.append((dat_file, "Error", None, None, None, None, None, None, None, None, None, None))

    with open("risultati.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "File", "ObjectiveValue", "SolveTime(s)", "MIPGap%", 
            "SolutionMatrix", "j0_Output", "j0_Input", 
            "OnPause", "tStart", "TMatrix",
            "NumInPause", "NumTransfers"
        ])

        for entry in results:
            (
                file_name, obj_value, solve_time, solution_matrix,
                j0_output, j0_input, mip_gap, onpause, tStart, T_matrix,
                num_pausa, num_trasferimenti
            ) = entry

            solution_str = str(solution_matrix.tolist()) if solution_matrix is not None else ""
            j0_out_str = str(j0_output) if j0_output is not None else ""
            j0_in_str = str(j0_input) if j0_input is not None else ""
            T_matrix_str = str(T_matrix.tolist()) if T_matrix is not None else ""

            writer.writerow([
                file_name, obj_value, solve_time, mip_gap,
                solution_str, j0_out_str, j0_in_str,
                onpause, tStart, T_matrix_str,
                num_pausa, num_trasferimenti
            ])

    print("Tutti i risultati sono stati scritti in 'risultati.csv'.")

if __name__ == '__main__':
    main()
