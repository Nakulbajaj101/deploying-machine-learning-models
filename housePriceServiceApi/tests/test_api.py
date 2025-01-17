import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_predictions(test_data: pd.DataFrame, client: TestClient) -> None:

    payload = {
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    response = client.post("http://localhost:8001/api/v1/predict",
                           json=payload)

    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"]
    assert prediction_data["errors"] == ""
    assert math.isclose(prediction_data["predictions"][0], 113422, rel_tol=100)
