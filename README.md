# COVID-19 Risk Prediction Pipeline

This project is a refactored and modularized machine learning pipeline for predicting COVID-19 confirmed cases and fatalities.  
It uses **CatBoost**, **XGBoost**, and other ML libraries, with a FastAPI interface for serving predictions and MLflow for experiment tracking.

---

## Project Structure
Assignment/
├── src/ # Source code (data loading, training, prediction, API)
│ ├── config.py
│ ├── data.py
│ ├── model.py
│ ├── train.py
│ ├── predict.py
│ └── serve.py
├── params.yaml # Configuration parameters
├── requirements.txt # Python dependencies
├── README.md # Documentation
└── .gitignore

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Covid-19.git
cd Covid-19

### 2. Create a virtual environment
```bash
conda create -n covid-19 python=3.9 -y
conda activate covid-19
(or use python -m venv venv && source venv/bin/activate)

### 3. Install dependencies
pip install -r requirements.txt

### 4. Training the Model
To train the model using the dataset in datasets/:
python -m src.train
The trained model will be saved to the models/ directory.

### 5. Making Predictions
Run:
python -m src.predict
This will load the saved model and generate predictions based on the input data.

### 6. Serving the Model via API
Start FastAPI server:
uvicorn src.serve:app --reload
By default, the API runs at:

http://localhost:8000

#### 7. Example Prediction Requests
1) Using Swagger UI
Open:
http://127.0.0.1:8000/docs
Find /predict-range, click "Try it out", paste your JSON, and Execute.

2) Using cURL (Linux / macOS / Git Bash on Windows)
curl -X POST http://localhost:8000/predict-range \
  -H "Content-Type: application/json" \
  -d '{"country":"Poland","province":"","start_date":"2020-03-25","end_date":"2020-04-23"}'
   
4) Using PowerShell
Invoke-WebRequest `
  -Uri "http://localhost:8000/predict-range" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"country":"Poland","province":"","start_date":"2020-03-25","end_date":"2020-04-23"}' |
  Select-Object -ExpandProperty Content

### 8. Tracking Experiments with MLflow
Start MLflow tracking server:

mlflow ui --port 5000

In src/train.py, set tracking URI:
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("covid19-risk")

Open MLflow UI in browser:
http://127.0.0.1:5000


### 9. All parameters (model hyperparameters, file paths, dates) are stored in:
params.yaml
Modify this file to adjust the training setup without changing the code.

Example params.yaml
train:
  iterations: 500
  learning_rate: 0.05
  depth: 8
data:
  start_date: "2020-03-01"
  end_date: "2020-06-01"
  dataset_path: "datasets/"


### 10. Notes
- The dataset is not included in this repository. Place it in datasets/ before training.

- For reproducibility, consider using conda env export > environment.yml after setting up the environment.

- Use .gitignore to avoid committing large model files and raw datasets.
