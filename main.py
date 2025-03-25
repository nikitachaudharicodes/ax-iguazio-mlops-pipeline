# main.py

import joblib
from data_ingestion.data_loader import load_ratings
from data_ingestion.preprocessing import preprocess_ratings, time_based_split
from mlops_pipeline.train_model import train_model
from mlops_pipeline.ax_search import run_hparam_search
from mlops_pipeline.ax_search_surprise import run_hparam_search_surprise
from mlops_pipeline.train_surprise_svd import train_surprise_svd



def main():
    print("Running Ax hyperparameter search...")
    best_params, values, experiment, ax_model = run_hparam_search_surprise(total_trials=10)

    # Show results
    print("Best parameters found:", best_params)
    print("All values:", values)
    print("Experiment object:", experiment)

    # train final model using best params
    # n_components_best = best_params["n_components"]

    mean_rmse = values[0]["val_rmse"]
    sem_rmse = values[1]["val_rmse"]["val_rmse"]

    print("\n=== Best Trial Details ===")
    print(f"Hyperparameters: {best_params}")
    print(f"Mean Val RMSE:   {mean_rmse:.4f}")
    print(f"SEM:             {sem_rmse:.4f}")
    print("==========================\n")

    


    df = load_ratings(limit=50000)
    df_clean = preprocess_ratings(df)


    model, val_rmse = train_surprise_svd(
        ratings_df=df_clean,
        n_factors=best_params["n_factors"],
        n_epochs=best_params["n_epochs"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"]
    )


    # train_df, val_df, test_df = time_based_split(df_clean)
    # final_svd, train_rmse, val_rmse = train_model(
    #     train_df, val_df, n_components=n_components_best
    # )

    #  save final model
    # joblib.dump(model, "best_svd_model.pkl")
    # print(f"Best SVD model saved with n_components={n_components_best} (val_rmse={val_rmse})")
    joblib.dump(model, "best_surprise_svd_model.pkl")
    print(f"Best Surprise SVD model saved! Val RSME: {val_rmse}")


if __name__ == "__main__":
    main()
