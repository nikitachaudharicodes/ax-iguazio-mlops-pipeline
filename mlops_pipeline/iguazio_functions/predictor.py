import joblib
import json

# Load the model once when the function starts
model = joblib.load('model/best_surprise_svd_model.pkl')

def handler(context, event):
    data = event.body
    user_id = data.get("user_id")
    movie_id = data.get("movie_id")
    
    # Predict using the Surprise model
    prediction = model.predict(user_id, movie_id).est
    
    return context.Response(body=json.dumps({"predicted_rating": prediction}),
                            headers={},
                            content_type='application/json',
                            status_code=200)
