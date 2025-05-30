# AssignmentAlgorithm

Programma per il ri-assegnamento dei dipendenti ai vari reparti di un'azienda in base ai loro livelli di stress.
Utilizza un modello di ottimizzazione (file `model.mod`) per assegnare i dipendenti ai reparti o a una pausa, minimizzando lo stress complessivo.
Alcuni parametri del modello possono essere modificati tramite il file `param_values.dat`.
Al momento, il codice è stato testato solo in ambiente Linux.

## Funzionalità
Chiamata API per ottenere il nuovo assegnamento dei dipendenti ai reparti.

## Librerie necessarie
- Amply e risolutore Cplex:
```bash
python -m pip install amplpy --upgrade
python -m amplpy.modules install cplex
```
- Flusk e Numpy:
```bash
pip install flask numpy
```
## Utilizzo del programma
- Lanciare il programma principale:
```bash
python prova_servizi.py
```
- Assicurarsi di avere un token aggiornato. Nel caso contrario, generare il token tramite:
```bash
python genera_token.py
```
- In un altro terminale, assicurandosi di aver installato curl nel proprio sistema:
```bash
curl http://127.0.0.1:5000/schedule
```
