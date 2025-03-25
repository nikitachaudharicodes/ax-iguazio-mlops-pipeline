import nuclio
import joblib
from data_ingestion.data_loader import load_ratings
from data_ingestion.preprocessing import preprocess_ratings
from mlops_pipeline.train_surprise_svd import train_surprise_svd

def handler(context, event):
    data = event.body  # dict containing hyperparams
    context.logger.info(f"Received params: {data}")

    df = load_ratings(limit=50000)
    df_clean = preprocess_ratings(df)

    model, val_rmse = train_surprise_svd(
        ratings_df=df_clean,
        n_factors=data.get("n_factors", 50),
        n_epochs=data.get("n_epochs", 20),
        lr_all=data.get("lr_all", 0.005),
        reg_all=data.get("reg_all", 0.02),
    )

    joblib.dump(model, "/tmp/best_surprise_svd_model.pkl")
    return {"val_rmse": val_rmse}
