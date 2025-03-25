# train_surprise_svd.py

from surprise import SVD, Dataset, Reader
import pandas as pd
import math

def train_surprise_svd(ratings_df, n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02):
    """
    Train a Surprise SVD model with the given hyperparams.
    Return the model & RMSE on a validation set you define.
    """
    # 1) Surprise needs a 'user item rating' CSV or DF, plus a Reader specifying rating_scale
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
    
    # 2) If you want a train/validation split, either do it manually or use Surprise's split
    # Example: do a random split:
    from surprise.model_selection import train_test_split
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # 3) Build the model
    algo = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        biased=True  # or false if you want to remove user/item biases
    )
    
    # 4) Train
    algo.fit(trainset)
    
    # 5) Evaluate on testset
    predictions = algo.test(testset)
    # Surprise has a built in accuracy function
    from surprise import accuracy
    rmse = accuracy.rmse(predictions, verbose=False)
    return algo, rmse
