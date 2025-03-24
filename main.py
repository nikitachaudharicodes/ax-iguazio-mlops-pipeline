# main.py

import joblib
from data_ingestion.data_loader import load_ratings
from data_ingestion.preprocessing import preprocess_ratings, time_based_split
from mlops_pipeline.train_model import train_model
from mlops_pipeline.ax_search import run_hparam_search

def main():
    print("Running Ax hyperparameter search...")
    best_params, values, experiment, ax_model = run_hparam_search(total_trials=10)

    # Show results
    print("Best parameters found:", best_params)
    print("All values:", values)
    print("Experiment object:", experiment)

    # Now train final model using best params
    n_components_best = best_params["n_components"]
    df = load_ratings(limit=50000)
    df_clean = preprocess_ratings(df)
    train_df, val_df, test_df = time_based_split(df_clean)
    final_svd, train_rmse, val_rmse = train_model(
        train_df, val_df, n_components=n_components_best
    )

    #  save final model
    joblib.dump(final_svd, "best_svd_model.pkl")
    print(f"Best SVD model saved with n_components={n_components_best} (val_rmse={val_rmse})")

if __name__ == "__main__":
    main()
