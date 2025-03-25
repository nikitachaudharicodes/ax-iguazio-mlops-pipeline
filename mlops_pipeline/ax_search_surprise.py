# mlops_pipeline/ax_search_surprise.py

from ax.service.managed_loop import optimize
from mlops_pipeline.train_surprise_svd import train_surprise_svd
from data_ingestion.data_loader import load_ratings
from data_ingestion.preprocessing import preprocess_ratings

def evaluation_function(params):
    df = load_ratings(limit=50000)
    df_clean = preprocess_ratings(df)
    model, val_rmse = train_surprise_svd(
        ratings_df=df_clean,
        n_factors=params["n_factors"],
        n_epochs=params["n_epochs"],
        lr_all=params["lr_all"],
        reg_all=params["reg_all"]
    )
    return {"val_rmse": (val_rmse, 0.0)}

def run_hparam_search_surprise(total_trials=10):
    best_params, values, experiment, model = optimize(
        parameters=[
            {"name": "n_factors", "type": "range", "bounds": [10, 100], "value_type": "int"},
            {"name": "n_epochs", "type": "range", "bounds": [5, 100], "value_type": "int"},
            {"name": "lr_all", "type": "range", "bounds": [0.001, 0.1], "value_type": "float"},
            {"name": "reg_all", "type": "range", "bounds": [0.01, 0.5], "value_type": "float"}
        ],
        evaluation_function=evaluation_function,
        objective_name="val_rmse",
        minimize=True,
        total_trials=total_trials
    )
    return best_params, values, experiment, model
