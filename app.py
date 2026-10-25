from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_processing import LIMITS
from src.prediction import (
    PredictionError,
    load_model,
    load_prediction_schema,
    predict_yield,
)
from src.storage import (
    RECORD_COLUMNS,
    add_record,
    database_url,
    delete_record,
    list_records,
    update_record,
)
from src.visualizations import (
    prediction_vs_actual,
    trend_chart,
    yield_by_category,
    yield_distribution,
)


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "milk_yield_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Dairy Yield Prediction",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. CUSTOM STYLING — TEAL + MINT + WHITE
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --teal: #146C65;
        --dark-teal: #0D514D;
        --light-teal: #218B80;
        --mint: #E5F7F1;
        --mint-white: #F6FCFA;
        --white: #FFFFFF;
        --text: #203B38;
        --muted: #64817C;
        --border: #CFE5DE;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            135deg, #F6FCFA 0%, #E8F7F2 100%
        );
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #DDF4EC 0%, #F6FCFA 100%
        );
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h2 {
        color: var(--dark-teal);
        font-weight: 800;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: var(--mint);
        border-color: var(--teal);
    }

    h1 {
        color: var(--dark-teal) !important;
        font-size: 2.35rem !important;
        font-weight: 800 !important;
    }

    h2 {
        color: var(--dark-teal) !important;
        font-weight: 750 !important;
    }

    h3 {
        color: var(--teal) !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg, #FFFFFF 0%, #F1FBF7 100%
        );
        border: 1px solid var(--border);
        border-left: 5px solid var(--teal);
        border-radius: 16px;
        padding: 20px 18px;
        min-height: 125px;
        box-shadow: 0 5px 18px rgba(13, 81, 77, 0.07);
    }

    [data-testid="stMetricLabel"] {
        color: #557873 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--teal) !important;
        font-weight: 800 !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        border: 1px solid var(--teal);
        padding: 0.65rem 1rem;
        font-weight: 650;
    }

    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(
            100deg, #146C65, #218B80
        );
        color: white;
        border: none;
    }

    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 5px 18px rgba(13, 81, 77, 0.05);
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {
        border-radius: 9px;
    }

    [data-baseweb="select"] > div {
        border-radius: 9px;
    }

    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid var(--border);
    }

    .stTabs [aria-selected="true"] {
        color: var(--teal) !important;
    }

    .warning-box {
        background: #EAF8F3;
        color: #245B53;
        border-left: 5px solid var(--teal);
        padding: 16px 20px;
        border-radius: 10px;
        margin: 14px 0 22px 0;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.9rem;
    }

    hr {
        border-color: var(--border) !important;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(
            90deg, #146C65, #83D8BE
        );
        border-radius: 10px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. REUSABLE FUNCTIONS
# --------------------------------------------------

def metric_card(
    label: str,
    value: str,
    help_text: str = "",
) -> None:
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


def csv_download(
    frame: pd.DataFrame,
    label: str,
    filename: str,
) -> None:
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def numeric_field(
    label: str,
    key: str,
    minimum: float,
    maximum: float,
    default: float,
    step: float,
    help_text: str = "",
) -> float:
    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default),
        step=float(step),
        key=key,
        help=help_text,
    )


# --------------------------------------------------
# 4. CATTLE RECORD FORM
# --------------------------------------------------

def record_form(
    prefix: str,
    initial: dict | None = None,
    categorical: dict[str, list[str]] | None = None,
) -> dict:

    categorical = categorical or {
        "breed": [],
        "parity_stage": [],
    }

    initial = initial or {}

    defaults = {
        "breed": initial.get("breed", "Holstein"),
        "age_months": initial.get("age_months", 48),
        "lactation_number": initial.get("lactation_number", 2),
        "body_weight_kg": initial.get("body_weight_kg", 580),
        "daily_feed_kg": initial.get("daily_feed_kg", 18),
        "parity_stage": initial.get("parity_stage", "Mid"),
        "days_in_milk": initial.get("days_in_milk", 120),
        "milk_yield_liters": initial.get("milk_yield_liters", 20),
    }

    cattle_id = st.text_input(
        "Cattle ID",
        value=str(initial.get("cattle_id", "")),
        key=f"{prefix}_id",
        max_chars=30,
    )

    breed_options = categorical.get("breed", [])

    if not breed_options:
        st.error("No breed options were found in the trained model.")
        st.stop()

    breed = st.selectbox(
        "Breed",
        breed_options,
        index=(
            breed_options.index(defaults["breed"])
            if defaults["breed"] in breed_options
            else 0
        ),
        key=f"{prefix}_breed",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = numeric_field(
            "Age (months)",
            f"{prefix}_age",
            *LIMITS["age_months"],
            defaults["age_months"],
            1,
        )

        weight = numeric_field(
            "Body weight (kg)",
            f"{prefix}_weight",
            *LIMITS["body_weight_kg"],
            defaults["body_weight_kg"],
            1,
        )

    with c2:
        lactation = st.number_input(
            "Lactation number",
            min_value=1,
            max_value=12,
            value=int(defaults["lactation_number"]),
            step=1,
            key=f"{prefix}_lactation",
        )

        feed = numeric_field(
            "Daily feed (kg)",
            f"{prefix}_feed",
            *LIMITS["daily_feed_kg"],
            defaults["daily_feed_kg"],
            0.1,
        )

    with c3:
        stage_options = categorical.get("parity_stage", [])

        if not stage_options:
            st.error("No parity-stage options were found in the model.")
            st.stop()

        stage = st.selectbox(
            "Parity stage",
            stage_options,
            index=(
                stage_options.index(defaults["parity_stage"])
                if defaults["parity_stage"] in stage_options
                else 0
            ),
            key=f"{prefix}_stage",
        )

        dim = st.number_input(
            "Days in milk",
            min_value=1,
            max_value=305,
            value=int(defaults["days_in_milk"]),
            step=1,
            key=f"{prefix}_dim",
        )

    yield_value = numeric_field(
        "Observed milk yield (L/cow/day)",
        f"{prefix}_yield",
        *LIMITS["milk_yield_liters"],
        float(defaults["milk_yield_liters"]),
        0.1,
    )

    initial_date = initial.get("record_date", date.today())

    if not isinstance(initial_date, date):
        initial_date = date.fromisoformat(str(initial_date)[:10])

    record_date = st.date_input(
        "Record date",
        value=initial_date,
        key=f"{prefix}_date",
    )

    return {
        "cattle_id": cattle_id.strip(),
        "breed": breed,
        "age_months": age,
        "lactation_number": lactation,
        "body_weight_kg": weight,
        "daily_feed_kg": feed,
        "parity_stage": stage,
        "days_in_milk": dim,
        "milk_yield_liters": yield_value,
        "record_date": record_date.isoformat(),
    }


# --------------------------------------------------
# 5. MODEL METADATA
# --------------------------------------------------

def metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}

    try:
        return json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return {}


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🥛 Dairy Yield")
    st.caption("Smart Dairy Farm Management")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Milk Yield Prediction",
            "Cattle Records",
            "Milk Analytics",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    if database_url():
        st.caption("🟢 Storage: PostgreSQL")
    else:
        st.caption("Storage: local SQLite")

    st.caption("Milk production decision-support prototype")


# --------------------------------------------------
# 7. INTERACTIVE DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #0D514D, #218B80);
                    padding: 30px;
                    border-radius: 20px;
                    margin-bottom: 24px;">
            <p style="color: #BDE9DC; font-weight: bold;
                      letter-spacing: 2px;">
                SMART DAIRY FARM MANAGEMENT
            </p>
            <h1 style="color: white; font-size: 32px; margin: 10px 0;">
                🥛 Dairy Operations Dashboard
            </h1>
            <p style="color: #EDF8F5; font-size: 16px;">
                Monitor milk production, explore cattle data,
                and discover valuable farm insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = records_frame()

    if records.empty:
        st.info(
            "No cattle records are available yet. "
            "Open Cattle Records and add your first record."
        )

    else:
        st.subheader("🔎 Dashboard Filters")

        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

        breeds = sorted(
            records["breed"].dropna().unique().tolist()
        )

        with filter_col1:
            selected_breeds = st.multiselect(
                "Select breed",
                options=breeds,
                default=breeds,
                key="dashboard_breeds",
            )

        with filter_col2:
            date_filter = st.checkbox(
                "Enable date filter",
                value=False,
                key="dashboard_date_filter",
            )

        with filter_col3:
            st.write("")
            st.write("")

            if st.button("Reset filters", use_container_width=True):
                st.session_state["dashboard_breeds"] = breeds
                st.session_state["dashboard_date_filter"] = False
                st.rerun()

        filtered_records = records[
            records["breed"].isin(selected_breeds)
        ].copy()

        if date_filter and not filtered_records.empty:
            parsed_dates = pd.to_datetime(
                filtered_records["record_date"],
                errors="coerce",
            )

            valid_dates = parsed_dates.dropna()

            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                selected_dates = st.date_input(
                    "Choose date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="dashboard_date_range",
                )

                if (
                    isinstance(selected_dates, (tuple, list))
                    and len(selected_dates) == 2
                ):
                    start_date, end_date = selected_dates

                    filtered_records = filtered_records[
                        parsed_dates.dt.date.between(start_date, end_date)
                    ]

        st.divider()

        valid = filtered_records.dropna(
            subset=["milk_yield_liters"]
        )

        st.subheader("📊 Production Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "🐄 Cattle Records",
                f"{len(filtered_records):,}",
                "Number of records matching the filters",
            )

        with c2:
            average = (
                valid["milk_yield_liters"].mean()
                if not valid.empty
                else None
            )

            metric_card(
                "🥛 Average Daily Yield",
                f"{average:.2f} L" if average is not None else "N/A",
                "Average recorded litres per cow per day",
            )

        with c3:
            metric_card(
                "🌿 Breeds Represented",
                f"{filtered_records['breed'].nunique():,}",
                "Distinct breeds in the filtered records",
            )

        with c4:
            total_yield = (
                valid["milk_yield_liters"].sum()
                if not valid.empty
                else 0
            )

            metric_card(
                "📈 Recorded Yield",
                f"{total_yield:,.2f} L",
                "Sum of recorded yield observations",
            )

        st.divider()

        if valid.empty:
            st.warning(
                "No records match the selected filters. "
                "Change the breed or date filters to view charts."
            )

        else:
            st.subheader("📈 Milk Production Insights")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.plotly_chart(
                    yield_distribution(valid),
                    use_container_width=True,
                    key="dashboard_yield_distribution",
                )

            with chart_col2:
                st.plotly_chart(
                    trend_chart(valid),
                    use_container_width=True,
                    key="dashboard_yield_trend",
                )

            st.plotly_chart(
                yield_by_category(valid, "breed"),
                use_container_width=True,
                key="dashboard_breed_comparison",
            )

            st.subheader("🐄 Recent Cattle Records")

            recent = filtered_records.copy()
            recent["record_date"] = pd.to_datetime(
                recent["record_date"],
                errors="coerce",
            )
            recent = recent.sort_values(
                "record_date",
                ascending=False,
            )

            st.dataframe(
                recent.head(10),
                use_container_width=True,
                hide_index=True,
            )

            csv_download(
                filtered_records,
                "⬇️ Download filtered records",
                "dairy_dashboard_records.csv",
            )


# --------------------------------------------------
# 8. MILK YIELD PREDICTION
# --------------------------------------------------

elif page == "Milk Yield Prediction":

    st.title("🥛 Milk Yield Prediction")

    st.caption(
        "Estimate milk production per cow per day using "
        "your saved machine-learning regression pipeline."
    )

    st.markdown(
        """
        <div class="warning-box">
            The prediction is a model estimate, not a guarantee
            of actual milk production or veterinary advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    try:
        _, categorical, _ = load_prediction_schema(MODEL_PATH)
    except PredictionError as exc:
        st.error(str(exc))
        st.stop()

    with st.form("prediction_form"):

        st.subheader("Enter Cattle Details")

        c1, c2 = st.columns(2)
        values = {}

        with c1:
            values["breed"] = st.selectbox(
                "Breed",
                categorical["breed"],
            )

            values["age_months"] = numeric_field(
                "Age (months)",
                "pred_age",
                *LIMITS["age_months"],
                48,
                1,
            )

            values["lactation_number"] = st.number_input(
                "Lactation number",
                min_value=1,
                max_value=12,
                value=2,
                step=1,
            )

            values["body_weight_kg"] = numeric_field(
                "Body weight (kg)",
                "pred_weight",
                *LIMITS["body_weight_kg"],
                580,
                1,
            )

        with c2:
            values["daily_feed_kg"] = numeric_field(
                "Daily feed (kg)",
                "pred_feed",
                *LIMITS["daily_feed_kg"],
                18,
                0.1,
            )

            values["parity_stage"] = st.selectbox(
                "Parity stage",
                categorical["parity_stage"],
            )

            values["days_in_milk"] = st.number_input(
                "Days in milk",
                min_value=1,
                max_value=305,
                value=120,
                step=1,
            )

        predict_col, reset_col = st.columns(2)

        with predict_col:
            submitted = st.form_submit_button(
                "🔮 Predict Milk Yield",
                type="primary",
                use_container_width=True,
            )

        with reset_col:
            reset_form = st.form_submit_button(
                "Reset Form",
                use_container_width=True,
            )

    if reset_form:
        st.rerun()

    if submitted:
        try:
            estimate = predict_yield(MODEL_PATH, values)

            st.success("Prediction completed successfully!")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Estimated Milk Yield", f"{estimate:.2f} L/day")

            with result_col2:
                st.metric(
                    "Estimated Weekly Yield",
                    f"{estimate * 7:.2f} L",
                    help="Assumes the same estimated daily yield for seven days.",
                )

            st.progress(min(max(float(estimate) / 50, 0.0), 1.0))

            st.caption(
                "The progress bar uses 50 L/day as a display scale, "
                "not as a prediction accuracy score."
            )

            st.session_state.prediction_history.insert(
                0,
                {
                    **values,
                    "predicted_yield_liters": estimate,
                    "record_type": "Model estimate",
                },
            )

        except PredictionError as exc:
            st.error(str(exc))

    if st.session_state.prediction_history:
        st.divider()
        st.subheader("📋 Prediction History")

        history = pd.DataFrame(st.session_state.prediction_history)

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        csv_download(
            history,
            "⬇️ Download Prediction History",
            "milk_yield_predictions.csv",
        )

        if st.button("Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()


# --------------------------------------------------
# 9. CATTLE RECORD MANAGEMENT
# --------------------------------------------------

elif page == "Cattle Records":

    st.title("🐄 Cattle Record Management")

    st.caption(
        "Add, edit, search, import, export, and delete cattle records."
    )

    try:
        _, categorical, _ = load_prediction_schema(MODEL_PATH)
    except PredictionError as exc:
        st.error(str(exc))
        st.stop()

    records = records_frame()

    tab_add, tab_view, tab_import = st.tabs(
        ["➕ Add / Edit", "📋 View Records", "📤 Import CSV"]
    )

    with tab_add:
        record_ids = (
            records["cattle_id"].astype(str).tolist()
            if not records.empty
            else []
        )

        edit_id = st.selectbox(
            "Select cattle record to edit",
            ["New record", *record_ids],
        )

        initial = None

        if edit_id != "New record":
            matched = records.loc[
                records["cattle_id"].astype(str) == edit_id
            ]

            if not matched.empty:
                initial = matched.iloc[0].to_dict()

        with st.form("record_form"):
            record = record_form("record", initial, categorical)

            saved = st.form_submit_button(
                "💾 Save Record",
                type="primary",
            )

        if saved:
            if not record["cattle_id"]:
                st.error("Cattle ID is required.")

            else:
                try:
                    if edit_id == "New record":
                        add_record(record)
                        st.success(
                            f"Added cattle record {record['cattle_id']}."
                        )
                    else:
                        update_record(edit_id, record)
                        st.success(f"Updated cattle record {edit_id}.")

                    refresh_records()
                    st.rerun()

                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

    with tab_view:
        if records.empty:
            st.info("No cattle records have been saved yet.")

        else:
            search = st.text_input("🔎 Search by cattle ID or breed")

            breed_filter = st.multiselect(
                "Filter by breed",
                sorted(records["breed"].dropna().unique()),
            )

            filtered = records.copy()

            if search:
                mask = (
                    filtered["cattle_id"].astype(str).str.contains(
                        search, case=False, na=False
                    )
                    |
                    filtered["breed"].astype(str).str.contains(
                        search, case=False, na=False
                    )
                )
                filtered = filtered[mask]

            if breed_filter:
                filtered = filtered[
                    filtered["breed"].isin(breed_filter)
                ]

            st.write(f"Matching records: {len(filtered)}")

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
            )

            csv_download(
                filtered,
                "⬇️ Export Visible Records",
                "cattle_records.csv",
            )

            st.divider()
            st.subheader("Delete a Record")

            del_id = st.selectbox(
                "Select cattle ID",
                ["", *records["cattle_id"].astype(str).tolist()],
                key="delete_id",
            )

            confirm = st.checkbox("I confirm permanent deletion")

            if st.button(
                "🗑️ Delete Record",
                disabled=not del_id or not confirm,
            ):
                try:
                    delete_record(del_id)
                    refresh_records()
                    st.success(f"Deleted record {del_id}.")
                    st.rerun()

                except ValueError as exc:
                    st.error(str(exc))

    with tab_import:
        upload = st.file_uploader(
            "Upload cattle records CSV",
            type=["csv"],
            accept_multiple_files=False,
        )

        if upload:
            if upload.size > 2_000_000:
                st.error("CSV must be smaller than 2 MB.")

            else:
                try:
                    incoming = pd.read_csv(upload)

                    if list(incoming.columns) != RECORD_COLUMNS:
                        raise ValueError(
                            "CSV columns must match the existing cattle "
                            "record format: " + ", ".join(RECORD_COLUMNS)
                        )

                    incoming["cattle_id"] = (
                        incoming["cattle_id"].astype(str).str.strip()
                    )

                    if (
                        incoming["cattle_id"].duplicated().any()
                        or incoming["cattle_id"].eq("").any()
                    ):
                        raise ValueError(
                            "Cattle IDs must be unique and non-empty."
                        )

                    numeric_columns = [
                        column
                        for column in RECORD_COLUMNS
                        if column not in {
                            "cattle_id",
                            "breed",
                            "parity_stage",
                            "record_date",
                        }
                    ]

                    incoming[numeric_columns] = incoming[
                        numeric_columns
                    ].apply(pd.to_numeric, errors="raise")

                    if incoming.isna().any().any():
                        raise ValueError(
                            "Imported records contain missing values."
                        )

                    st.dataframe(
                        incoming,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button(
                        "Import Validated Records",
                        type="primary",
                    ):
                        existing = set(
                            records["cattle_id"].astype(str)
                        )

                        if set(incoming["cattle_id"]) & existing:
                            raise ValueError(
                                "One or more cattle IDs already exist."
                            )

                        for item in incoming.to_dict("records"):
                            add_record(item)

                        refresh_records()
                        st.success(
                            f"Imported {len(incoming)} records."
                        )
                        st.rerun()

                except (ValueError, pd.errors.ParserError) as exc:
                    st.error(f"Import failed: {exc}")


# --------------------------------------------------
# 10. MILK ANALYTICS
# --------------------------------------------------

elif page == "Milk Analytics":

    st.title("📈 Milk Production Analytics")

    st.caption(
        "Explore yield distribution, breed comparisons, "
        "and milk production trends."
    )

    records = records_frame()

    if records.empty:
        st.info(
            "No data available. Save cattle records "
            "to enable milk analytics."
        )

    else:
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            selected_breed = st.multiselect(
                "Select breeds",
                sorted(records["breed"].dropna().unique()),
            )

        with filter_col2:
            selected_stage = st.multiselect(
                "Select parity stages",
                sorted(records["parity_stage"].dropna().unique()),
            )

        filtered = records.copy()

        if selected_breed:
            filtered = filtered[
                filtered["breed"].isin(selected_breed)
            ]

        if selected_stage:
            filtered = filtered[
                filtered["parity_stage"].isin(selected_stage)
            ]

        valid = filtered.dropna(subset=["milk_yield_liters"])

        if valid.empty:
            st.warning(
                "No valid yield observations match these filters."
            )

        else:
            avg_yield = valid["milk_yield_liters"].mean()
            max_yield = valid["milk_yield_liters"].max()
            min_yield = valid["milk_yield_liters"].min()

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("Average Yield", f"{avg_yield:.2f} L")

            with c2:
                metric_card(
                    "Highest Recorded Yield",
                    f"{max_yield:.2f} L",
                )

            with c3:
                metric_card(
                    "Lowest Recorded Yield",
                    f"{min_yield:.2f} L",
                )

            st.divider()

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.plotly_chart(
                    yield_distribution(valid),
                    use_container_width=True,
                )

            with chart_col2:
                st.plotly_chart(
                    yield_by_category(valid, "breed"),
                    use_container_width=True,
                )

            st.plotly_chart(
                trend_chart(valid),
                use_container_width=True,
            )

            st.caption(
                f"Showing {len(valid):,} observations "
                f"from {len(filtered):,} filtered records."
            )

            csv_download(
                valid,
                "⬇️ Download Filtered Analytics",
                "filtered_milk_analytics.csv",
            )


# --------------------------------------------------
# 11. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.caption(
        "Review the trained model's metadata, evaluation metrics, "
        "and predicted-versus-actual yield comparison."
    )

    model_info = metadata()

    if not MODEL_PATH.exists():
        st.error(f"Trained model not found: {MODEL_PATH}")
        st.info(
            "Train the model and save it to "
            "models/milk_yield_model.joblib."
        )
        st.stop()

    st.success("Trained model file found.")

    if model_info:
        st.subheader("📋 Model Information")

        metadata_frame = pd.DataFrame(
            [
                {
                    "Property": key.replace("_", " ").title(),
                    "Value": (
                        json.dumps(value)
                        if isinstance(value, (dict, list))
                        else str(value)
                    ),
                }
                for key, value in model_info.items()
            ]
        )

        st.dataframe(
            metadata_frame,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.warning(
            "Model metadata is unavailable. Check "
            "models/model_metadata.json."
        )

    st.divider()
    st.subheader("📊 Evaluation Metrics")

    metric_keys = {
        "r2": "R² Score",
        "r2_score": "R² Score",
        "mae": "Mean Absolute Error (MAE)",
        "mean_absolute_error": "Mean Absolute Error (MAE)",
        "mse": "Mean Squared Error (MSE)",
        "mean_squared_error": "Mean Squared Error (MSE)",
        "rmse": "Root Mean Squared Error (RMSE)",
        "root_mean_squared_error": "Root Mean Squared Error (RMSE)",
    }

    metric_values = {}

    def collect_metrics(data):
        for key, value in data.items():
            normalized_key = str(key).lower().replace("-", "_")

            if normalized_key in metric_keys:
                try:
                    metric_values[
                        metric_keys[normalized_key]
                    ] = float(value)
                except (TypeError, ValueError):
                    pass

            elif isinstance(value, dict):
                collect_metrics(value)

    collect_metrics(model_info)

    if metric_values:
        metric_columns = st.columns(min(len(metric_values), 4))

        for index, (label, value) in enumerate(metric_values.items()):
            with metric_columns[index % len(metric_columns)]:
                st.metric(label, f"{value:.4f}")

    else:
        st.info(
            "No recognized evaluation metrics were found "
            "in the model metadata."
        )

    # ----------------------------------------------
    # PREDICTION VS ACTUAL
    # ----------------------------------------------

    st.divider()
    st.subheader("📈 Prediction vs Actual")

    records = records_frame()

    if records.empty:
        st.info(
            "Add cattle records with observed milk yields "
            "to explore production data."
        )

    elif "milk_yield_liters" not in records.columns:
        st.warning(
            "The records do not contain actual milk-yield values."
        )

    else:
        try:
            model = load_model(MODEL_PATH)
            features, _, _ = load_prediction_schema(MODEL_PATH)

            missing_features = [
                feature for feature in features
                if feature not in records.columns
            ]

            if missing_features:
                st.warning(
                    "Required model input columns are missing from "
                    "the cattle records: "
                    + ", ".join(missing_features)
                )

            else:
                comparison_data = records[
                    features + ["milk_yield_liters"]
                ].copy()

                comparison_data["milk_yield_liters"] = pd.to_numeric(
                    comparison_data["milk_yield_liters"],
                    errors="coerce",
                )

                comparison_data = comparison_data.dropna()

                if comparison_data.empty:
                    st.info(
                        "No complete cattle records are available "
                        "for comparison."
                    )

                else:
                    actual = comparison_data[
                        "milk_yield_liters"
                    ].astype(float)

                    predicted = model.predict(
                        comparison_data[features]
                    )

                    comparison_frame = pd.DataFrame({
                        "Actual (L/cow/day)": actual.to_numpy(),
                        "Predicted (L/cow/day)": predicted,
                    })

                    comparison_chart = prediction_vs_actual(
                        comparison_frame[
                            "Actual (L/cow/day)"
                        ].tolist(),
                        comparison_frame[
                            "Predicted (L/cow/day)"
                        ].tolist(),
                    )

                    st.write(
                        "Each point compares an observed milk yield "
                        "with the model's prediction for the same "
                        "cattle inputs."
                    )

                    st.plotly_chart(
                        comparison_chart,
                        use_container_width=True,
                    )

                    st.subheader("📋 Comparison Details")

                    st.dataframe(
                        comparison_frame.round(2),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_download(
                        comparison_frame,
                        "⬇️ Download Prediction Comparison",
                        "prediction_vs_actual.csv",
                    )

                    st.caption(
                        f"Compared {len(comparison_frame)} records."
                    )

        except PredictionError as exc:
            st.error(f"Could not load the prediction model: {exc}")

        except (ValueError, KeyError, TypeError) as exc:
            st.warning(
                f"The comparison chart could not be generated: {exc}"
            )

    st.divider()

    st.caption(
        "The comparison chart uses the trained model on saved cattle "
        "records. These records may have been used during training, "
        "so this chart alone does not measure performance on unseen data."
    )

