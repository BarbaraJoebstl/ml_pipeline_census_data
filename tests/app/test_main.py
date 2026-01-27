from fastapi.testclient import TestClient
from app.main import app
from ml.consts import VALID_PAYLOAD

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Census Income Predictor API!" in response.json()["message"]


def test_predict_valid_input():
    """Test that the /predict endpoint returns a valid prediction for valid input."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "It is predicted that the person earns" in response.text


def test_predict_invalid_input():
    """Test that /predict returns a 422 when input validation fails."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload.pop("age")  # remove a required field
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
