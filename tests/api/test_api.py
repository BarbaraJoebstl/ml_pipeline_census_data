from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Census Income Predictor API!" in response.json()["message"]
