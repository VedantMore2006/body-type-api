from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict():
    with TestClient(app) as client:
        with open("extra/test/front2.jpeg", "rb") as f:
            response = client.post(
                "/predict",
                data={
                    "age": 25,
                    "gender": 1,
                    "person_height_cm": 180,
                    "person_weight_kg": 75
                },
                files={"image": ("front2.jpeg", f, "image/jpeg")}
            )
        print("Status Code:", response.status_code)
        print("Response:", response.json())

if __name__ == "__main__":
    test_predict()
