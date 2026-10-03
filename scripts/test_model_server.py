import requests
import numpy as np


data = np.random.rand(1, 3, 224, 224).astype("float32")


response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json={
        "inputs": data.tolist()
    }
)


print("Status:", response.status_code)
print("Response:", response.text)