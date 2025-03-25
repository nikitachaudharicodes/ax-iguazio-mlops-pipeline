# mlops_pipeline/mlrun_train_surprise.py

import joblib
import mlrun
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

@mlrun.handler()
def train_surprise_svd(context,
                       n_factors: int = 50,
                       n_epochs: int = 20,
                       lr_all: float = 0.005,
                       reg_all: float = 0.02,
                       sample_size: int = 50000):
    
    # Load & prepare data
    from data_ingestion.data_loader import load_ratings
    from data_ingestion.preprocessing import preprocess_ratings

    df = load_ratings(limit=sample_size)
    df = preprocess_ratings(df)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, biased=True)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    # Log metric
    context.log_metric("val_rmse", rmse)

    # Save model
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    context.log_model("surprise_svd_model", local_path=model_path, model_file="model.pkl")

    return rmse
