from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_success():
    response = client.post("/predict", json={
        "feature_1": 5.1,
        "feature_2": 3.5,
        "feature_3": 1.4,
        "feature_4": 0.2
    })
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_out_of_range():
    response = client.post("/predict", json={
        "feature_1": -1,
        "feature_2": 15,
        "feature_3": 0,
        "feature_4": 2
    })
    assert response.status_code == 422


def test_upload_model():
    with open("model.pkl", "rb") as f:
        response = client.post("/upload_model", files={"model_file": f})
    assert response.status_code == 200
    assert response.json()["detail"] == "Модель успешно обновлена."
