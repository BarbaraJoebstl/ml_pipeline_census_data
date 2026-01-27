import requests


# Helper script to test the Heroku endpoint for post, sending a mock payload.

URL = "https://<your-app-name>.herokuapp.com/predict"


# sends a payload of on row with censusInput data conform structure.
payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


response = requests.post(URL, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
