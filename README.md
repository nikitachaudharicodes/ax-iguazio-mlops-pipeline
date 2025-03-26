
# MLOps with Ax and Nuclio: Hyperparameter Optimization and Serverless Deployment

This repository demonstrates a structured Machine Learning Operations (MLOps) pipeline integrating Ax for hyperparameter tuning and Nuclio (by Iguazio) for serverless machine learning deployment. The workflow automates hyperparameter optimization, model training, and real-time inference deployment.

---

## Project Overview

- **Ax**: An adaptive experimentation platform from Meta (Facebook) used for systematic hyperparameter optimization.
- **Nuclio**: A high-performance serverless computing platform provided by Iguazio, enabling real-time model training and inference.
- **Surprise Library**: Used to implement a collaborative filtering algorithm (SVD) for demonstration purposes.

---

## Repository Structure

```
ax-iguazio-mlops-pipeline/
├── data_ingestion/
│   ├── data_loader.py
│   └── preprocessing.py
├── mlops_pipeline/
│   ├── train_surprise_svd.py
│   ├── train_surprise_svd_nuclio.py
│   └── svd_inference_nuclio.py
├── ax-nuclio-optimizer.py
└── requirements.txt
```

---

## Prerequisites

- Python 3.9+
- Docker Desktop
- Nuclio Dashboard (runs via Docker)
- PostgreSQL Database (to host rating data)

---

## Setup

### Step 1: Clone the Repository

```sh
git clone https://github.com/yourusername/ax-iguazio-mlops-pipeline.git
cd ax-iguazio-mlops-pipeline
```

### Step 2: Install Dependencies

```sh
pip install -r requirements.txt
```

Alternatively, manually install key dependencies:

```sh
pip install ax-platform scikit-surprise numpy pandas joblib psycopg2-binary requests python-dotenv
```

### Step 3: Setup Environment Variables

Create a `.env` file with database configurations:

```env
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_password
DB_HOST=your_db_host
DB_PORT=5432
```

---

## Nuclio Deployment

### Step 1: Launch Nuclio Dashboard

```sh
docker run -p 8070:8070 -v /var/run/docker.sock:/var/run/docker.sock quay.io/nuclio/dashboard:stable-amd64
```

Open [http://localhost:8070](http://localhost:8070).

### Step 2: Deploy Nuclio Training Function

- Create a new Nuclio function using the archive URL option with your GitHub release URL.
- Runtime: `Python 3.9`
- Handler: `train_surprise_svd_nuclio:handler`
- Ensure you install dependencies in the build commands:

```sh
pip install pandas numpy scikit-surprise psycopg2-binary joblib nuclio-sdk
```

### Step 3: Deploy Nuclio Inference Function

- Create a second Nuclio function similarly.
- Runtime: `Python 3.9`
- Handler: `svd_inference_nuclio:handler`
- Install similar dependencies for inference.

---

## Configure Shared Volume

Both training and inference functions must share a persistent volume to save/load trained models:

In Nuclio dashboard, under "Configuration -> Volumes":

- Name: `model-volume`
- Container Path: `/opt/nuclio/model`
- Volume Type: `HostPath`
- Path: `/path/on/local/system/shared-model`

Ensure this directory exists:

```sh
mkdir -p /path/on/local/system/shared-model
```

---

## Running Hyperparameter Optimization with Ax

Execute Ax hyperparameter optimization locally, which communicates with your Nuclio training endpoint:

```sh
python ax-nuclio-optimizer.py
```

Adjust the Nuclio URL in `ax-nuclio-optimizer.py` to your training function’s local endpoint (`http://localhost:PORT`).

---

## Testing and Running Inference

Test inference via HTTP requests using CURL or Postman:

```sh
curl -X POST http://localhost:NUCLIO_INFER_PORT \
-H "Content-Type: application/json" \
-d '{"user_id": "19477", "movie_id": "gladiator+2000"}'
```

Replace `NUCLIO_INFER_PORT` with the port provided in Nuclio dashboard after deployment.

---

## Flowchart of the Implementation

```plaintext
Ax Optimization
      │ Sends Hyperparameters
      ▼
Nuclio Training Function
      │ Trains & Saves Best Model
      ▼
Persistent Volume (/opt/nuclio/model)
      │ Loads Best Model
      ▼
Nuclio Inference Function
      │ Real-time Predictions
      ▼
Prediction Response
```

---

## Troubleshooting

- Ensure the Docker Desktop daemon is running.
- Check Nuclio dashboard logs for build/deployment errors.
- Verify shared volume path permissions.

---

## Conclusion

This repository demonstrates a powerful combination of hyperparameter tuning and serverless deployment, ideal for robust and scalable ML workflows. Leveraging Ax and Nuclio streamlines experimentation, enhances reproducibility, and accelerates deployment cycles, essential in modern MLOps practices.

---

**Enjoy experimenting with scalable MLOps!**
