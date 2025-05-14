# Insiemi
set EMPLOYEES;        # Insieme dei dipendenti (i)
set DEPARTMENTS;       # Insieme dei reparti (j), includendo il reparto "0" per la pausa
set WORK_DEPARTMENTS = { j in DEPARTMENTS : j <> 0 }; # Insieme dei reparti lavorativi (senza la pausa)
# Parametri
param j0 {EMPLOYEES};
# j0[i] rappresenta il reparto originariamente assegnato al dipendente i

param s {EMPLOYEES, DEPARTMENTS};
# s[i,j] è il livello di stress atteso del dipendente i se assegnato al reparto j.
# Per j=0 (pausa) vanno inseriti i valori di stress attesi (bassi).

param M {WORK_DEPARTMENTS};
# M[j] indica il numero minimo di dipendenti richiesti nel reparto j.
# Per il reparto pausa (j=0) impostare M[0]=0.

param Smax {EMPLOYEES};
# Smax[i] è il livello massimo di stress tollerabile per il dipendente i

param alpha {EMPLOYEES, DEPARTMENTS};
param onpause;
param k;
param T {WORK_DEPARTMENTS, WORK_DEPARTMENTS};
# Variabili decisionali
var x {EMPLOYEES, DEPARTMENTS} binary;
# x[i,j] = 1 se il dipendente i viene assegnato al reparto j, 0 altrimenti

# Funzione Obiettivo: Minimizzare il costo complessivo (stress + penalità per cambi + penalità per la pausa)
minimize TotalCost:
    sum {i in EMPLOYEES, j in WORK_DEPARTMENTS} s[i,j] * x[i,j]
  + k * sum {i in EMPLOYEES} (sum {j in WORK_DEPARTMENTS: j0[i] <> 0} T[j0[i], j] * x[i,j] )
  + onpause * sum {i in EMPLOYEES} x[i,0];

# Vincolo 1: Ogni reparto (eccetto la pausa) deve avere almeno M[j] dipendenti
subject to DeptCoverage {j in WORK_DEPARTMENTS}:
    sum {i in EMPLOYEES} x[i,j] >= M[j];

# Vincolo 2: Ogni dipendente deve essere assegnato ad un solo reparto
subject to OneAssignment {i in EMPLOYEES}:
    sum {j in DEPARTMENTS} x[i,j] = 1;

# Vincolo 3: Un dipendente può essere assegnato a un reparto solo se possiede le competenze
subject to Competence {i in EMPLOYEES, j in WORK_DEPARTMENTS}:
    x[i,j] <= alpha[i,j];

# Vincolo 4 (Opzionale): Il livello di stress per i reparti "lavorativi" non deve superare Smax[i]
subject to StressLimit {i in EMPLOYEES}:
    sum {j in WORK_DEPARTMENTS: j <> 0} s[i,j] * x[i,j] <= Smax[i];

