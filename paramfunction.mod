# Insiemi
set EMPLOYEES;        # Insieme dei dipendenti (i)
set DEPARTMENTS;       # Insieme dei reparti (j)

# Parametri
param j0 {EMPLOYEES};  
# j0[i] rappresenta il reparto originariamente assegnato al dipendente i

param s {EMPLOYEES, DEPARTMENTS};
# s[i,j] è il livello di stress atteso del dipendente i se assegnato al reparto j

param M {DEPARTMENTS};
# M[j] indica il numero minimo di dipendenti richiesti nel reparto j

param Smax {EMPLOYEES};
# Smax[i] è il livello massimo di stress tollerabile per il dipendente i

param lambda;
# Penalizzazione per il cambiamento di assegnazione

param alpha {EMPLOYEES, DEPARTMENTS};
# alpha[i,j] = 1 se il dipendente i possiede le competenze per lavorare nel reparto j, 0 altrimenti

# Variabili decisionali
var x {EMPLOYEES, DEPARTMENTS} binary;
# x[i,j] = 1 se il dipendente i viene assegnato al reparto j, 0 altrimenti

var y {EMPLOYEES} binary;
# y[i] = 1 se il dipendente i viene riassegnato (cioè assegnato a un reparto diverso da j0[i]), 0 altrimenti

# Funzione Obiettivo: Minimizzare il livello complessivo di stress e penalizzare i cambi
minimize TotalCost:
    sum {i in EMPLOYEES, j in DEPARTMENTS} s[i,j] * x[i,j]
  + lambda * sum {i in EMPLOYEES} y[i];

# Vincolo 1: Ogni reparto deve avere almeno M[j] dipendenti
subject to DeptCoverage {j in DEPARTMENTS}:
    sum {i in EMPLOYEES} x[i,j] >= M[j];

# Vincolo 2: Ogni dipendente può essere assegnato a un solo reparto
subject to OneAssignment {i in EMPLOYEES}:
    sum {j in DEPARTMENTS} x[i,j] = 1;
    
# Vincolo 3: Un dipendente può essere assegnato a un reparto solo se ha le competenze (alpha[i,j] = 1)
subject to Competence {i in EMPLOYEES, j in DEPARTMENTS}:
    x[i,j] <= alpha[i,j];


# Vincolo 4: Se il dipendente viene assegnato a un reparto diverso da quello originario, y[i] deve essere 1
subject to ChangeIndicator {i in EMPLOYEES}:
    y[i] = sum {j in DEPARTMENTS: j <> j0[i]} x[i,j];

# Vincolo 5 (Opzionale): Il livello di stress assegnato al dipendente non deve superare Smax[i]
subject to StressLimit {i in EMPLOYEES}:
    sum {j in DEPARTMENTS} s[i,j] * x[i,j] <= Smax[i];

