#!/usr/bin/env python3
"""
Questo modulo implementa un programma che permette all'utente di creare 
un file .dat personalizzato, contenente i valori dei parametri del modello per un
problema di ottimizzazione. Questo file verra' salvato nella directory 'dat_files'.
L'utente potra' scegliere quali valori impostare manualmente e quali far generare
automaticamente dal programma. Per poter riprodurre i risultati, qual'ora si scegliesse
di generare automaticamente alcuni valori, e' previsto l'utilizzo di un seed randomico.
"""

import random
import numpy as np

def main():
    SEED = 42
    print(f"Attenzione: presente un random_seed impostato a {SEED}.")
    random.seed(SEED)
    # Costanti definite nel codice
    epsilon = 0.001
    tEnd = 28800
    onpause = 70
    # Input numero di dipendenti e reparti lavorativi (il reparto "pausa" è sempre 0)
    n = int(input("Numero di dipendenti: "))
    m = int(input("Numero di reparti lavorativi (escludendo il reparto pausa): "))
    total_depts = m + 1  # includiamo il reparto 0 per la pausa

    # Input assegnazione iniziale per ogni dipendente (j0)
    same_j0 = input("Vuoi impostare la stessa posizione iniziale per tutti i dipendenti o distribuirli in maniera randomica escludendo la pausa?  (S/N/R): ").strip().upper()
    j0 = []
    if same_j0 == 'S':
        j0_value = int(input("Inserisci la posizione iniziale (0 per pausa, oppure 1..{}): ".format(m)))
        j0 = [j0_value] * n
    elif same_j0 == "N":
        print("Inserisci la posizione iniziale per ciascun dipendente (0 per pausa, oppure 1..{}):".format(m))
        for i in range(n):
            val = int(input(f"Dipendente {i+1}: "))
            j0.append(val)
    elif same_j0 == "R": j0 = [random.randint(1, m) for _ in range(n)]

    # Input minimo numero di dipendenti per ogni reparto (M)
    same_M = input("Vuoi impostare lo stesso numero minimo di dipendenti per ogni reparto lavorativo? (S/N): ").strip().upper()
    M = {}
    M[0] = 0  # per il reparto pausa, M[0] deve essere 0
    if same_M == 'S':
        M_value = int(input("Inserisci il numero minimo per ogni reparto lavorativo: "))
        for j in range(1, total_depts):
            M[j] = M_value
    else:
        for j in range(1, total_depts):
            M[j] = int(input(f"Numero minimo di dipendenti per il reparto {j}: "))

    # Input soglia massima di stress per ogni dipendente (Smax)
    same_Smax = input("Vuoi impostare lo stesso livello massimo di stress per tutti i dipendenti? (S/N): ").strip().upper()
    Smax = []
    if same_Smax == 'S':
        Smax_value = float(input("Inserisci il livello massimo di stress: "))
        Smax = [Smax_value] * n
    else:
        print("Inserisci il livello massimo di stress per ciascun dipendente:")
        for i in range(n):
            val = float(input(f"Dipendente {i+1}: "))
            Smax.append(val)

    # Generazione della matrice s (stress atteso)
    auto_s = input("Vuoi generare automaticamente la matrice s? (S/N): ").strip().upper()
    s_matrix = []
    if auto_s == 'S':
        # Per ogni dipendente, genera i valori per i reparti lavorativi in maniera casuale
        for i in range(n):
            row = []
            working_stresses = []
            # Genera valori casuali per i reparti lavorativi (j=1,...,m)
            for j in range(1, total_depts):
                stress_val = random.randint(45, 80)
                working_stresses.append(stress_val)
            # Calcola la media dei valori dei reparti lavorativi
            avg_stress = sum(working_stresses) / len(working_stresses)
            # Il valore per la pausa (j=0) sarà il 60% della media
            pause_stress = 0.6 * avg_stress
            # Inserisce il valore per la pausa in prima posizione e poi i valori casuali
            row.append(pause_stress)
            row.extend(working_stresses)
            s_matrix.append(row)
    else:
        # Input manuale della matrice s
        same_s_pause = input("Vuoi impostare lo stesso livello di stress atteso per il reparto pausa (j=0) per tutti i dipendenti? (S/N): ").strip().upper()
        for i in range(n):
            row = []
            if same_s_pause == 'S':
                if i == 0:
                    s_pause = float(input("Inserisci il livello di stress atteso per il reparto pausa (j=0): "))
                    row.append(s_pause)
            else:
                val = float(input(f"Inserisci il livello di stress atteso per il dipendente {i+1} in pausa (j=0): "))
                row.append(val)
            # Per i reparti lavorativi (j da 1 a m)
            for j in range(1, total_depts):
                val = float(input(f"Inserisci il livello di stress atteso per il dipendente {i+1} nel reparto {j}: "))
                row.append(val)
            s_matrix.append(row)

    # Input matrice di competenze alpha
    same_alpha = input("Vuoi impostare tutti i valori di competenza a 1 per tutti i dipendenti e reparti lavorativi? (S/N): ").strip().upper()
    alpha_matrix = []
    for i in range(n):
        row = []
        # Per il reparto pausa (j=0), alpha deve essere 1
        row.append(1)
        if same_alpha == 'S':
            for j in range(1, total_depts):
                row.append(1)
        else:
            print(f"Inserisci le competenze del dipendente {i+1} per i reparti lavorativi (0 o 1):")
            for j in range(1, total_depts):
                val = int(input(f"Competenza per il reparto {j} (1 per sì, 0 per no): "))
                row.append(val)
        alpha_matrix.append(row)

    T_matrix = np.zeros((total_depts - 1, total_depts - 1), dtype=int)

    risposta = input("Vuoi inserire manualmente i coefficienti? (s/n): ").strip().lower()

    for i in range(total_depts - 1):
        for j in range(i + 1, total_depts - 1):
            if risposta == 's':
                while True:
                    try:
                        valore = int(input(f"Inserisci il valore per la posizione ({i+1},{j+1}): "))
                        break
                    except ValueError:
                        print("Valore non valido. Inserisci un numero intero.")
            else:
                valore = random.randint(5, 10)

            T_matrix[i][j] = valore
            T_matrix[j][i] = valore  # Mantieni la simmetria
    while True:
        try:
            tStart = int(input(f"Inserisci il timestamp attuale del turno. (tEnd = {tEnd}): "))
            break;
        except ValueError:
            print("Valore non valido. Inserisci un numero intero.")

    # Chiedi il nome del file .dat da generare
    dat_filename = input("Inserisci il nome del file .dat da generare (es. istanza1.dat): ").strip()

    # Genera il file .dat in formato compatibile con il modello AMPL
    with open("dat_files/" + dat_filename, "w", encoding="utf-8") as f:
        # Insiemi
        f.write("# Insiemi\n")
        f.write("set EMPLOYEES := ")
        for i in range(1, n + 1):
            f.write(f"{i} ")
        f.write(";\n\n")
        f.write("set DEPARTMENTS := 0 ")
        for j in range(1, total_depts):
            f.write(f"{j} ")
        f.write(";\n\n")
        # Parametro j0
        f.write("param j0 :=\n")
        for i in range(1, n + 1):
            f.write(f"{i} {j0[i-1]}\n")
        f.write(";\n\n")
        # Parametro s (matrice di stress)
        f.write("param s :")
        for j in range(0, total_depts):
            f.write(f" {j}")
        f.write(" :=\n")
        for i in range(n):
            f.write(f"{i+1} ")
            for j in range(total_depts):
                f.write(f"{s_matrix[i][j]} ")
            f.write("\n")
        f.write(";\n\n")
        # Parametro M (minimi per reparto)
        f.write("param M :=\n")
        # Per il reparto pausa, M[0] deve essere 0
        #----f.write("0 0\n")
        for j in range(1, total_depts):
            f.write(f"{j} {M[j]}\n")
        f.write(";\n\n")
        # Parametro Smax (stress massimo per dipendente)
        f.write("param Smax :=\n")
        for i in range(1, n + 1):
            f.write(f"{i} {Smax[i-1]}\n")
        f.write(";\n\n")
        # Parametro epsilon
        f.write(f"param epsilon := {epsilon};\n\n")
        # Parametro tStart
        f.write(f"param tStart := {tStart};\n\n")
        # Parametro tEnd
        f.write(f"param tEnd := {tEnd};\n\n")
        # Parametro onpause
        f.write(f"param onpause := {onpause};\n\n")
        # Parametro alpha (matrice di competenze)
        f.write("param alpha :")
        for j in range(0, total_depts):
            f.write(f" {j}")
        f.write(" :=\n")
        for i in range(n):
            f.write(f"{i+1} ")
            for j in range(total_depts):
                f.write(f"{alpha_matrix[i][j]} ")
            f.write("\n")
        f.write(";\n\n")
        f.write("param T :")
        for j in range(1, total_depts):  # Colonne intestate da 1 a total_depts - 1
            f.write(f" {j}")
        f.write(" :=\n")

        for i in range(1, total_depts):  # Righe intestate da 1 a total_depts - 1
            f.write(f"{i} ")
            for j in range(1, total_depts):
                f.write(f"{T_matrix[i-1][j-1]} ")
            f.write("\n")
        f.write(";\n\n")
    
    # Aggiunge il nome del file .dat generato in append a "dati.dat"
    with open("dati.txt", "a", encoding="utf-8") as f:
        f.write(dat_filename + "\n")
    
    print(f"Il file {dat_filename} è stato generato e aggiunto a 'dati.txt'.")

if __name__ == '__main__':
    main()
