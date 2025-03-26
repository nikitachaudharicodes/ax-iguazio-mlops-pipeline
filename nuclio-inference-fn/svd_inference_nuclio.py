import joblib
import os
import pandas as pd
import json  # Add this

model = None

def load_model():
    global model
    model_path = "/opt/nuclio/model/best_surprise_svd_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError("Model not found at inference time!")

def handler(context, event):
    global model
    if model is None:
        load_model()

    try:
        if isinstance(event.body, bytes):
            body = json.loads(event.body.decode("utf-8"))
        elif isinstance(event.body, dict):
            body = event.body  # Already parsed
        else:
            return {"error": "Unsupported payload format"}
    except Exception as e:
        return {"error": "Invalid JSON payload", "details": str(e)}

    user_id = body.get("user_id")
    movie_id = body.get("movie_id")

    try:
        prediction = model.predict(user_id, movie_id).est
        return {"prediction": prediction}
    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}
