# mlops_pipeline/ax_search.py

from ax.service.managed_loop import optimize
from data_ingestion.data_loader import load_ratings
from data_ingestion.preprocessing import preprocess_ratings, time_based_split
from mlops_pipeline.train_model import train_model

def evaluation_function(params):
    n_components = params["n_components"]

    df = load_ratings(limit=50000)
    df_clean = preprocess_ratings(df)
    train_df, val_df, test_df = time_based_split(df_clean)
    model, train_rmse, val_rmse = train_model(train_df, val_df, n_components=n_components)

    # We want to minimize val_rmse
    return {"val_rmse": (val_rmse, 0.0)}

def run_hparam_search(total_trials=20):
    """
    Runs the Ax hyperparam search, returns:
      - best_params
      - values
      - experiment
      - ax_model
    """
    best_params, values, experiment, ax_model = optimize(
        parameters=[
            {
                "name": "n_components",
                "type": "range",
                "bounds": [5, 60],
                "value_type": "int"
            }
        ],
        evaluation_function=evaluation_function,
        objective_name="val_rmse",
        minimize=True,
        total_trials=20
    )
    return best_params, values, experiment, ax_model
