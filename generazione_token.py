import requests

# URL per generazione token
url = "https://target-ai.revelis.eu/msf-service/api/token/generate"

# header
headers = {
    "Cdclient": "target-wfm",
    "Origin": "https://target-ai.revelis.eu"
}

# dati da inviare nel corpo del POST (JSON)
data = { 
    "username": "scheduler", 
    "password": "5ch3dUl3r!", 
    "tokenClient": "ZAip8BrbFB4QQSmjTW13mQ2lWB0dVtHQA9U7AVj1prpBccSUDswsvbxNYnilVFZavGmNDkpZFnUNdDEWDN7CTzxV4bPYPiOylgej"
} 

# invio richiesta
response = requests.post(url, json=data, headers=headers)

# verifica dello status della risposta
print("Status Code:", response.status_code)
print("Response Body:", response.text)

# Tentativo di estrazione del token dalla risposta JSON
try:
    response_json = response.json()
    token = response_json.get("token")
    if token:
        # Scrittura del token nel file, sovrascrivendo il contenuto precedente
        with open("token.txt", "w") as file:
            file.write(token)
        print("Token salvato correttamente in token.txt")
    else:
        print("Il campo 'token' non è stato trovato nella risposta.")
except Exception as e:
    print("Errore durante l'analisi della risposta JSON:", e)

