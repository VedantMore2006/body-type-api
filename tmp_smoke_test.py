from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

with open("test/front1.jpeg", "rb") as front_file, open("test/side1.jpeg", "rb") as side_file:
    response = client.post(
        "/detect/",
        files={
            "front_image": ("front1.jpeg", front_file, "image/jpeg"),
            "side_image": ("side1.jpeg", side_file, "image/jpeg"),
        },
        data={
            "age": "27",
            "gender": "1",
            "height": "175",
        },
    )

print("status:", response.status_code)
print("body:", response.text)
