# Insiemi
set EMPLOYEES;        # Insieme dei dipendenti (i)
set DEPARTMENTS;       # Insieme dei reparti (j), includendo il reparto "0" per la pausa

# Parametri
param j0 {EMPLOYEES};
# j0[i] rappresenta il reparto originariamente assegnato al dipendente i

param s {EMPLOYEES, DEPARTMENTS};
# s[i,j] è il livello di stress atteso del dipendente i se assegnato al reparto j.
# Per j=0 (pausa) vanno inseriti i valori di stress attesi (bassi).

param M {DEPARTMENTS};
# M[j] indica il numero minimo di dipendenti richiesti nel reparto j.
# Per il reparto pausa (j=0) impostare M[0]=0.

param Smax {EMPLOYEES};
# Smax[i] è il livello massimo di stress tollerabile per il dipendente i

param lambda;
# Penalizzazione per il cambiamento di assegnazione

param alpha {EMPLOYEES, DEPARTMENTS};
# alpha[i,j] = 1 se il dipendente i possiede le competenze per lavorare nel reparto j, 0 altrimenti.
# Assicurarsi che alpha[i,0]=1 per ogni i (ogni dipendente può essere messo in pausa).

# Nuovo parametro per penalizzare l’assegnamento alla pausa
param c_pausa;
# c_pausa rappresenta il costo addizionale (elevato) se un dipendente viene assegnato al reparto pausa

# Variabili decisionali
var x {EMPLOYEES, DEPARTMENTS} binary;
# x[i,j] = 1 se il dipendente i viene assegnato al reparto j, 0 altrimenti

# Funzione Obiettivo: Minimizzare il costo complessivo (stress + penalità per cambi + penalità per la pausa)
minimize TotalCost:
    sum {i in EMPLOYEES, j in DEPARTMENTS} s[i,j] * x[i,j]
  + lambda * sum {i in EMPLOYEES} (sum {j in DEPARTMENTS: j <> j0[i]} x[i,j])
  + c_pausa * sum {i in EMPLOYEES} x[i,0];

# Vincolo 1: Ogni reparto (eccetto la pausa) deve avere almeno M[j] dipendenti
subject to DeptCoverage {j in DEPARTMENTS}:
    sum {i in EMPLOYEES} x[i,j] >= M[j];

# Vincolo 2: Ogni dipendente deve essere assegnato ad un solo reparto
subject to OneAssignment {i in EMPLOYEES}:
    sum {j in DEPARTMENTS} x[i,j] = 1;

# Vincolo 3: Un dipendente può essere assegnato a un reparto solo se possiede le competenze
subject to Competence {i in EMPLOYEES, j in DEPARTMENTS}:
    x[i,j] <= alpha[i,j];

# Vincolo 4 (Opzionale): Il livello di stress per i reparti "lavorativi" non deve superare Smax[i]
subject to StressLimit {i in EMPLOYEES}:
    sum {j in DEPARTMENTS: j <> 0} s[i,j] * x[i,j] <= Smax[i];

