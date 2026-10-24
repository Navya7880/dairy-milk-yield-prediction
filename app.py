from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_processing import CATEGORICAL_FEATURES, FEATURES, LIMITS
from src.prediction import PredictionError, load_prediction_schema, predict_yield
from src.storage import (
    RECORD_COLUMNS,
    add_record,
    database_url,
    delete_record,
    list_records,
    update_record,
)
from src.visualizations import prediction_vs_actual, trend_chart, yield_by_category, yield_distribution

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "milk_yield_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Dairy Yield Prediction", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#1e2a23; --green:#1c6e4b; --mint:#eaf0eb; --amber:#d6a84b; }
html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
[data-testid="stAppViewContainer"] { background: #f7f8f5; }
[data-testid="stSidebar"] { background: #edf3ee; border-right: 1px solid #dbe5dc; }
[data-testid="stMetric"] { background: white; border: 1px solid #dfe6df; border-left: 4px solid var(--green); border-radius: 8px; padding: 14px; }
.block-container { max-width: 1180px; padding-top: 2rem; }
h1, h2, h3 { color: var(--ink); }
.stButton>button { border-radius: 6px; }
[data-testid="stForm"] { background: white; border: 1px solid #dfe6df; padding: 20px; border-radius: 8px; }
.small-note { color: #627067; font-size: .9rem; }
.warning-box { background: #fff7e6; border-left: 4px solid var(--amber); padding: 12px 16px; border-radius: 4px; }
.</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text)


@st.cache_data(ttl=5)
def cached_records() -> pd.DataFrame:
    return list_records()


def records_frame() -> pd.DataFrame:
    try:
        return cached_records()
    except Exception as exc:
        st.error(f"Records could not be loaded: {exc}")
        return pd.DataFrame(columns=RECORD_COLUMNS)


def refresh_records() -> None:
    cached_records.clear()


def csv_download(frame: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename, "text/csv")


def numeric_field(label: str, key: str, minimum: float, maximum: float, default: float, step: float, help_text: str = "") -> float:
    return st.number_input(label, min_value=float(minimum), max_value=float(maximum), value=float(default), step=float(step), key=key, help=help_text)


def record_form(prefix: str, initial: dict | None = None, categorical: dict[str, list[str]] | None = None) -> dict:
    categorical = categorical or {"breed": [], "parity_stage": []}
    initial = initial or {}
    defaults = {
        "breed": initial.get("breed", "Holstein"), "age_months": initial.get("age_months", 48),
        "lactation_number": initial.get("lactation_number", 2),
        "body_weight_kg": initial.get("body_weight_kg", 580), "daily_feed_kg": initial.get("daily_feed_kg", 18),
        "parity_stage": initial.get("parity_stage", "Mid"), "days_in_milk": initial.get("days_in_milk", 120),
        "milk_yield_liters": initial.get("milk_yield_liters", 20),
    }
    cattle_id = st.text_input("Cattle ID", value=initial.get("cattle_id", ""), key=f"{prefix}_id", max_chars=30)
    breed_options = categorical["breed"]
    breed = st.selectbox("Breed", breed_options, index=breed_options.index(defaults["breed"]) if defaults["breed"] in breed_options else 0, key=f"{prefix}_breed")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = numeric_field("Age (months)", f"{prefix}_age", *LIMITS["age_months"], defaults["age_months"], 1)
        weight = numeric_field("Body weight (kg)", f"{prefix}_weight", *LIMITS["body_weight_kg"], defaults["body_weight_kg"], 1)
    with c2:
        lactation = st.number_input("Lactation number", 1, 12, int(defaults["lactation_number"]), 1, key=f"{prefix}_lactation")
        feed = numeric_field("Daily feed (kg)", f"{prefix}_feed", *LIMITS["daily_feed_kg"], defaults["daily_feed_kg"], 0.1)
    with c3:
        stage_options = categorical["parity_stage"]
        stage = st.selectbox("Parity stage", stage_options, index=stage_options.index(defaults["parity_stage"]), key=f"{prefix}_stage")
        dim = st.number_input("Days in milk", 1, 305, int(defaults["days_in_milk"]), 1, key=f"{prefix}_dim")
    yield_value = numeric_field("Observed milk yield (L/cow/day)", f"{prefix}_yield", *LIMITS["milk_yield_liters"], float(defaults["milk_yield_liters"]), 0.1)
    record_date = st.date_input("Record date", value=date.fromisoformat(str(initial.get("record_date", date.today()))), key=f"{prefix}_date")
    return {
        "cattle_id": cattle_id.strip(), "breed": breed, "age_months": age, "lactation_number": lactation,
        "body_weight_kg": weight, "daily_feed_kg": feed, "parity_stage": stage, "days_in_milk": dim,
        "milk_yield_liters": yield_value, "record_date": record_date.isoformat(),
    }


def metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


with st.sidebar:
    st.markdown("### Dairy Yield")
    st.caption("Academic decision-support prototype")
    page = st.radio("Navigation", ["Dashboard", "Milk Yield Prediction", "Cattle Records", "Milk Analytics", "Model Performance"], label_visibility="collapsed")
    st.divider()
    if database_url():
        st.caption("Storage: configured PostgreSQL")
    else:
        st.caption("Storage: local SQLite")


if page == "Dashboard":
    st.title("Dairy operations dashboard")
    st.caption("Live record counts and milk-production summaries")
    records = records_frame()
    valid = records.dropna(subset=["milk_yield_liters"]) if not records.empty else records
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Cattle records", f"{len(records):,}")
    with c2: metric_card("Average daily yield", f"{valid['milk_yield_liters'].mean():.2f} L" if not valid.empty else "N/A", "L/cow/day")
    with c3: metric_card("Breeds represented", f"{records['breed'].nunique():,}" if not records.empty else "0")
    with c4:
        total = valid["milk_yield_liters"].sum() if not valid.empty else 0
        metric_card("Recorded yield", f"{total:,.2f} L", "Sum of valid record observations")
    if records.empty:
        st.info("No cattle records yet. Add one in Cattle Records to populate dashboard charts.")
    else:
        left, right = st.columns(2)
        with left: st.plotly_chart(yield_distribution(valid), width="stretch")
        with right: st.plotly_chart(trend_chart(records), width="stretch")
        st.subheader("Recent records")
        st.dataframe(records.head(8), width="stretch", hide_index=True)

elif page == "Milk Yield Prediction":
    st.title("Milk yield prediction")
    st.caption("Estimate liters per cow per day using the saved preprocessing and regression pipeline.")
    st.markdown('<div class="warning-box">Prediction is a model estimate, not a guarantee or veterinary advice.</div>', unsafe_allow_html=True)
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    try:
        features, categorical, types = load_prediction_schema(MODEL_PATH)
    except PredictionError as exc:
        st.error(str(exc))
        st.stop()
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        values = {}
        with c1:
            values["breed"] = st.selectbox("Breed", categorical["breed"])
            values["age_months"] = numeric_field("Age (months)", "pred_age", *LIMITS["age_months"], 48, 1)
            values["lactation_number"] = st.number_input("Lactation number", 1, 12, 2, 1)
            values["body_weight_kg"] = numeric_field("Body weight (kg)", "pred_weight", *LIMITS["body_weight_kg"], 580, 1)
        with c2:
            values["daily_feed_kg"] = numeric_field("Daily feed (kg)", "pred_feed", *LIMITS["daily_feed_kg"], 18, 0.1)
            values["parity_stage"] = st.selectbox("Parity stage", categorical["parity_stage"])
            values["days_in_milk"] = st.number_input("Days in milk", 1, 305, 120, 1)
        predict, reset = st.columns(2)
        with predict: submitted = st.form_submit_button("Predict yield", type="primary", width="stretch")
        with reset: reset_form = st.form_submit_button("Reset form", width="stretch")
    if reset_form:
        st.rerun()
    if submitted:
        try:
            estimate = predict_yield(MODEL_PATH, values)
            st.success(f"Estimated yield: **{estimate:.2f} L/cow/day**")
            st.session_state.prediction_history.insert(0, {**values, "predicted_yield_liters": estimate, "record_type": "Model estimate"})
            st.caption("This estimate should be compared only with real records from the target cattle and context.")
        except PredictionError as exc:
            st.error(str(exc))
    if st.session_state.prediction_history:
        st.subheader("Prediction history")
        history = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history, width="stretch", hide_index=True)
        csv_download(history, "Download prediction history", "milk_yield_predictions.csv")

elif page == "Cattle Records":
    st.title("Cattle record management")
    try:
        _, categorical, _ = load_prediction_schema(MODEL_PATH)
    except PredictionError as exc:
        st.error(str(exc))
        st.stop()
    records = records_frame()
    tab_add, tab_view, tab_import = st.tabs(["Add / edit", "View records", "Import CSV"])
    with tab_add:
        edit_id = st.selectbox("Edit existing cattle ID", ["New record", *records["cattle_id"].tolist()] if not records.empty else ["New record"])
        initial = records.loc[records["cattle_id"] == edit_id].iloc[0].to_dict() if edit_id != "New record" else None
        with st.form("record_form"):
            record = record_form("record", initial, categorical)
            saved = st.form_submit_button("Save record", type="primary")
        if saved:
            if not record["cattle_id"]:
                st.error("Cattle ID is required.")
            else:
                try:
                    if edit_id == "New record":
                        add_record(record)
                        st.success(f"Added cattle record {record['cattle_id']}.")
                    else:
                        update_record(edit_id, record)
                        st.success(f"Updated cattle record {edit_id}.")
                    refresh_records()
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
    with tab_view:
        if records.empty:
            st.info("No cattle records are saved yet.")
        else:
            search = st.text_input("Search cattle ID or breed")
            breed_filter = st.multiselect("Filter by breed", sorted(records["breed"].dropna().unique()))
            filtered = records
            if search:
                filtered = filtered[filtered["cattle_id"].astype(str).str.contains(search, case=False) | filtered["breed"].astype(str).str.contains(search, case=False)]
            if breed_filter:
                filtered = filtered[filtered["breed"].isin(breed_filter)]
            st.dataframe(filtered, width="stretch", hide_index=True)
            csv_download(filtered, "Export visible records", "cattle_records.csv")
            del_id = st.selectbox("Delete cattle ID", ["", *records["cattle_id"].tolist()], key="delete_id")
            confirm = st.checkbox("I confirm permanent deletion")
            if st.button("Delete record", disabled=not del_id or not confirm):
                try:
                    delete_record(del_id)
                    refresh_records()
                    st.success(f"Deleted record {del_id}.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    with tab_import:
        upload = st.file_uploader("Upload cattle records", type=["csv"], accept_multiple_files=False)
        if upload:
            if upload.size > 2_000_000:
                st.error("CSV must be smaller than 2 MB.")
            else:
                try:
                    incoming = pd.read_csv(upload)
                    if list(incoming.columns) != RECORD_COLUMNS:
                        raise ValueError(f"Expected columns: {', '.join(RECORD_COLUMNS)}")
                    incoming["cattle_id"] = incoming["cattle_id"].astype(str).str.strip()
                    if incoming["cattle_id"].duplicated().any() or incoming["cattle_id"].eq("").any():
                        raise ValueError("Cattle IDs must be unique and non-empty.")
                    numeric_columns = [c for c in RECORD_COLUMNS if c not in {"cattle_id", "breed", "parity_stage", "record_date"}]
                    incoming[numeric_columns] = incoming[numeric_columns].apply(pd.to_numeric, errors="raise")
                    if incoming[RECORD_COLUMNS].isna().any().any():
                        raise ValueError("Imported records contain missing values.")
                    st.dataframe(incoming, width="stretch", hide_index=True)
                    if st.button("Import validated records", type="primary"):
                        existing = set(records["cattle_id"])
                        if set(incoming["cattle_id"]) & existing:
                            raise ValueError("One or more cattle IDs already exist.")
                        for item in incoming.to_dict("records"):
                            add_record(item)
                        refresh_records()
                        st.success(f"Imported {len(incoming)} records.")
                        st.rerun()
                except (ValueError, pd.errors.ParserError) as exc:
                    st.error(f"Import failed: {exc}")

elif page == "Milk Analytics":
    st.title("Milk production analytics")
    records = records_frame()
    if records.empty:
        st.info("No data available. Save cattle records to enable filtered analytics.")
    else:
        c1, c2 = st.columns(2)
        with c1: selected_breed = st.multiselect("Breed", sorted(records["breed"].dropna().unique()), default=[])
        with c2: selected_stage = st.multiselect("Parity stage", sorted(records["parity_stage"].dropna().unique()), default=[])
        filtered = records.copy()
        if selected_breed: filtered = filtered[filtered["breed"].isin(selected_breed)]
        if selected_stage: filtered = filtered[filtered["parity_stage"].isin(selected_stage)]
        valid = filtered.dropna(subset=["milk_yield_liters"])
        if valid.empty:
            st.warning("No valid yield observations match these filters.")
        else:
            l, r = st.columns(2)
            with l: st.plotly_chart(yield_distribution(valid), width="stretch")
            with r: st.plotly_chart(yield_by_category(valid, "breed"), width="stretch")
            st.plotly_chart(trend_chart(valid), width="stretch")
            st.caption(f"Showing {len(valid):,} observations from {len(filtered):,} filtered records.")
            csv_download(valid, "Download filtered data", "filtered_milk_analytics.csv")

elif page == "Model Performance":
    st.title("Model performance")
    metrics = metadata()
    if not metrics:
        st.warning("Evaluation metadata is unavailable. Run `python -m src.train_model` to create it.")
    else:
        st.caption(f"Model: {metrics['model_name']} | Data: {metrics['dataset_type']}")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("R²", f"{metrics['r2']:.3f}", "Share of variance explained on held-out data")
        with c2: metric_card("MAE", f"{metrics['mae']:.3f} L", "Average absolute prediction error")
        with c3: metric_card("RMSE", f"{metrics['rmse']:.3f} L", "Error penalty emphasizing larger misses")
        with c4: metric_card("Held-out rows", f"{metrics['test_count']:,}")
        st.write(f"Training rows: **{metrics['train_count']}**; test rows: **{metrics['test_count']}**; seed: **{metrics['seed']}**.")
        st.plotly_chart(prediction_vs_actual(metrics["actual"], metrics["predicted"]), width="stretch")
        st.subheader("Mean-baseline comparison")
        st.write(f"Regression MAE **{metrics['mae']:.3f}** vs mean baseline **{metrics['baseline_mae']:.3f}** L/cow/day.")
        st.write(f"Regression RMSE **{metrics['rmse']:.3f}** vs mean baseline **{metrics['baseline_rmse']:.3f}** L/cow/day.")
        st.subheader("How to read the metrics")
        st.markdown("**R²** summarizes fit on the test rows. **MAE** is the average absolute miss in liters. **RMSE** is similar but penalizes larger misses more strongly. These numbers are based on the labelled synthetic demonstration dataset and do not establish real-farm accuracy.")
        st.subheader("Limitations")
        st.markdown("- The source data is synthetic, not a real dairy farm observation set.\n- Feature relationships may not transfer to real cattle, breeds, feed systems, or climates.\n- Regression coefficients and predictions do not prove causation.\n- A model can be useful for a demonstration and still require field validation before operational use.")

else:
    st.title("Unsupported page")
    st.subheader("A Multiple Linear Regression Model for Dairy Milk Yield Prediction")
    st.markdown("**Abstract:** This academic prototype combines a multiple linear regression pipeline with cattle-record CRUD, SQLite/PostgreSQL persistence, milk-production analytics, held-out model evaluation, and a Streamlit interface for liter-per-cow-per-day yield estimates.")
    st.subheader("Problem and objective")
    st.markdown("Dairy operations need a transparent, reusable way to explore factors associated with daily yield. The objective is to demonstrate a reproducible predictive workflow, not replace expert farm judgment.")
    st.subheader("Method")
    st.markdown("The app validates the dataset, separates features and target, applies train-only preprocessing with median/frequent imputation and one-hot encoding, fits scikit-learn `LinearRegression`, evaluates on an 80/20 held-out split, and serializes the complete pipeline with Joblib.")
    st.subheader("Architecture")
    st.markdown("Streamlit directly calls Python modules in `src/`. It does not require a separate HTTP API. `data_processing.py` owns schema and cleaning, `train_model.py` trains/evaluates, `prediction.py` validates and predicts, `storage.py` persists records, and `visualizations.py` builds Plotly charts.")
    st.subheader("Dataset")
    st.markdown("The included demonstration CSV is **synthetic** and is generated deterministically in `src/data_processing.py`. It is not from a real farm. Source, target, limits, and limitations are documented in [data/README.md](data/README.md).")
    st.subheader("Using the application")
    st.markdown("1. Open **Cattle Records** to add, edit, search, delete, import, or export records.\n2. Open **Milk Yield Prediction** to enter the same seven features used by the model.\n3. Open **Milk Analytics** to filter real saved records and export matching observations.\n4. Open **Model Performance** to inspect the genuine held-out metrics and baseline comparison.")
    st.subheader("Implemented versus future work")
    st.markdown("Implemented features are CRUD, PostgreSQL-capable storage, pipeline inference, prediction history, filters, charts, CSV import/export, and model evaluation. Future work may include real multi-farm data, independent field validation, verified-yield observations, authentication, audit trails, and clinician/veterinary review.")
    st.subheader("Cloud persistence")
    if database_url():
        st.success("DATABASE_URL is configured; PostgreSQL is used for record storage.")
    else:
        st.markdown('<div class="warning-box">Without DATABASE_URL, Streamlit Cloud may not preserve SQLite data across restarts or deployments. Use a managed PostgreSQL database for persistent deployed records.</div>', unsafe_allow_html=True)
    st.subheader("Technical stack")
    st.markdown("Python, Streamlit, Pandas, NumPy, scikit-learn, Plotly, Joblib, SQLite, optional PostgreSQL/psycopg, and pytest.")
