import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import math

def train_model(train_df, val_df, n_components = 20):
    """
    simple matrix factorization approach using SVD.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        n_components (int): Number of components for SVD.

    Returns:
        model(TruncatedSVD): the SVD model
        train_rmse (float) : RMSE on the training set
        val_rmse (float): RMSE on the validation set(skipping unknown users/items)
    """


    unique_users = train_df['user_id'].unique()
    unique_movies = train_df['movie_id'].unique()

    user_encoder = {uid: idx for idx, uid in enumerate(unique_users)}
    item_encoder = {iid: idx for idx, iid in enumerate(unique_movies)}

    num_users = len(user_encoder)
    num_items = len(item_encoder)

    R = np.zeros((num_users, num_items), dtype = np.float32)

    for row in train_df.itertuples():
        u = user_encoder[row.user_id]
        i = item_encoder[row.movie_id]
        R[u, i] = row.rating
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(R)
    item_factors = svd.components_.T

    R_hat = user_factors @ item_factors.T

    train_sq_err = 0.0
    train_count = 0

    for row in train_df.itertuples():
            u = user_encoder[row.user_id]
            i = item_encoder[row.movie_id]
            pred = R_hat[u, i]
            train_sq_err += (pred - row.rating) ** 2
            train_count += 1
    train_rmse = math.sqrt(train_sq_err / train_count)


    val_sq_err = 0.0
    val_count = 0
    for row in val_df.itertuples():
        if (row.user_id in user_encoder) and (row.movie_id in item_encoder):
            u = user_encoder[row.user_id]
            i = item_encoder[row.movie_id]
            pred = R_hat[u, i]
            val_sq_err += (pred - row.rating) ** 2
            val_count += 1
    val_rmse = math.sqrt(val_sq_err / val_count) if val_count > 0 else None

    return svd, train_rmse, val_rmse
    


