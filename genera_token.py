import requests
import json

file_name = "get_token.json"

with open(file_name, "r") as json_file:
    json_data = json.load(json_file)

# URL per generazione token
url = json_data["url"]

# costruzione header
cdclient = json_data["header"]["cdclient"]
origin = json_data["header"]["origin"]
headers = {
    "Cdclient": f"{cdclient}",
    "Origin": f"{origin}"
}

# dati da inviare
username = json_data["auth"]["username"]
password = json_data["auth"]["password"]
tokenclient = json_data["auth"]["tokenclient"]
data = { 

  "username": f"{username}", 

  "password": f"{password}", 

  "tokenClient": f"{tokenclient}"

} 

# invio richiesta
response = requests.post(url, json=data, headers=headers)

# risposta
print("Status Code:", response.status_code)

# salvataggio token
response_json = response.json()
print("Response JSON:", response_json)

token = response_json.get("token", "")

data = {
    "auth": {
        "token": token
    }
}

output_file = "service.json"

# lettura
with open(output_file, 'r') as file:
    data = json.load(file)

data['auth']['token'] = token

with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)
    print("Token salvato")
