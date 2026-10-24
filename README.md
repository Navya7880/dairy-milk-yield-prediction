# A Multiple Linear Regression Model for Dairy Milk Yield Prediction

A complete academic decision-support prototype built with Python and Streamlit. It predicts daily milk yield per cow, manages cattle records, analyzes production observations, evaluates a Multiple Linear Regression model, and explains implemented limitations.

## Important dataset status

The included `data/synthetic_dairy_milk.csv` is explicitly synthetic demonstration data generated locally by `src/data_processing.py`. It is not from a real farm and must not be presented as evidence of real-world predictive performance. The target is `milk_yield_liters`, in liters per cow per day. Full source and limitations are in [data/README.md](data/README.md).

## Features

- Five-page Streamlit interface: dashboard, prediction, cattle CRUD, analytics, and model performance.
- Actual scikit-learn `Pipeline` with train-only median/frequent imputation, one-hot encoding, and `LinearRegression`.
- Reproducible 80/20 split, held-out R²/MAE/RMSE, mean-baseline comparison, and predicted-versus-actual chart.
- Validated predictions from the complete serialized Joblib pipeline.
- Local SQLite CRUD with optional PostgreSQL through `DATABASE_URL`.
- CSV import/export, record filtering, prediction history, and Plotly charts.
- Automated tests for the data contract, training, prediction validation, persistence, and error handling.

## Architecture

```text
app.py                 Streamlit entry point and pages
src/data_processing.py Dataset generation, schema, cleaning, and validation
src/train_model.py     Pipeline training, evaluation, persistence, and metadata
src/prediction.py      Model loading and validated prediction
src/storage.py         SQLite/PostgreSQL CRUD
src/visualizations.py  Reusable Plotly charts
data/                  Synthetic demonstration dataset and documentation
models/                Saved pipeline and evaluation metadata
tests/                 Pytest coverage
```

Streamlit directly calls these Python modules. No separate HTTP API is required.

## Setup on Windows with VS Code

1. Open this folder in VS Code.
2. Open a terminal and create an environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

VS Code should select the `.venv` interpreter as the project environment.

## Run

Generate or reuse the synthetic dataset, train the model, and save artifacts:

```powershell
python -m src.train_model
```

Start the application:

```powershell
streamlit run app.py
```

Open the URL shown in the terminal, normally `http://localhost:8501`.

## Test

```powershell
python -m compileall -q app.py src tests
python -m pytest -q -p no:cacheprovider
```

## Machine-learning workflow

1. Validate the seven input features and target.
2. Reject missing, invalid, duplicate, out-of-range, or unexpectedly columned data.
3. Split with 80% training and 20% held-out test data using seed 42.
4. Fit numeric/categorical preprocessing on training data only.
5. Fit the Multiple Linear Regression model.
6. Evaluate R², MAE, RMSE, and a mean-prediction baseline on held-out data.
7. Save the full preprocessing-and-model pipeline with Joblib.

## Database configuration

No `DATABASE_URL` means local SQLite storage at `data/dairy_records.db`. SQLite file persistence is suitable for a local demonstration, but Streamlit Community Cloud may reset local files across restarts or deployments.

For PostgreSQL-backed deployed storage, set the secret/environment variable:

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE
```

In Streamlit Community Cloud: **App settings > Secrets**, use the Streamlit TOML format:

```toml
DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
```

Do not commit credentials or secrets. The app uses parameterized queries and creates the `cattle_records` table if it does not exist.

PostgreSQL support uses `psycopg`. PostgreSQL configuration was implemented but requires a real deployment database and credentials to verify end-to-end.

## GitHub and deployment

Initialize and publish a repository only after reviewing the `.gitignore`:

```powershell
git init
git add .
git commit -m "Build dairy milk yield prediction app"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

Then deploy the repository in Streamlit Community Cloud with `app.py` as the entry point and `requirements.txt` in the project root. Add `DATABASE_URL` under Secrets if persistent PostgreSQL data is required.

## Deployment troubleshooting

- **Module not found:** verify every import is listed in `requirements.txt` and the entry point is `app.py`.
- **Model missing:** run the training script locally and commit generated model/metadata under `models/`.
- **Database lost:** configure PostgreSQL through `DATABASE_URL`; local SQLite is not guaranteed to persist on managed hosting.
- **PostgreSQL connection failed:** inspect host, database, user, password, SSL policy, and network allowlist without exposing secrets in screenshots or logs.
- **Upload rejected:** confirm exact `RECORD_COLUMNS`, CSV encoding, 2 MB limit, numeric fields, unique IDs, and required values.
- **Bad prediction:** verify all seven expected inputs and value ranges displayed in the form.

## Limitations and future work

- Synthetic data cannot establish generalizable or biological validity.
- The app does not implement authentication or user authorization.
- Shared SQLite/PostgreSQL access should be treated as a shared demo, not a regulated production system.
- Future work should add licensed real-farm data, external field validation, record provenance, verified actual-yield history, role-based access, audit logging, and expert review.

## Model claims

The trained demo pipeline currently reports held-out values from seed 42: R² 0.884, MAE 0.9062 L/cow/day, RMSE 1.1887 L/cow/day. These are transparent demonstration results only and should not be described as clinical or field accuracy.
