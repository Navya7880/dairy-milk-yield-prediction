from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass(frozen=True)
class TrendData:
    frame: pd.DataFrame
    valid_count: int
    distinct_dates: int


def prepare_trend_data(data: pd.DataFrame) -> TrendData:
    if data.empty or "record_date" not in data or "milk_yield_liters" not in data:
        return TrendData(pd.DataFrame(columns=["record_date", "milk_yield_liters"]), 0, 0)
    frame = data[["record_date", "milk_yield_liters"]].copy()
    frame["record_date"] = pd.to_datetime(frame["record_date"], errors="coerce", format="mixed")
    frame["milk_yield_liters"] = pd.to_numeric(frame["milk_yield_liters"], errors="coerce")
    frame = frame.dropna(subset=["record_date", "milk_yield_liters"])
    frame["record_date"] = frame["record_date"].dt.normalize()
    frame = frame.groupby("record_date", as_index=False)["milk_yield_liters"].mean()
    frame = frame.sort_values("record_date")
    return TrendData(frame, len(frame), frame["record_date"].nunique())


def yield_distribution(data: pd.DataFrame):
    values = pd.to_numeric(data.get("milk_yield_liters", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        fig = go.Figure()
        fig.update_layout(title="Observed Milk Yield Distribution", annotations=[dict(text="No valid milk-yield observations", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")])
        return fig
    if len(values) <= 5:
        frame = pd.DataFrame({"Milk yield (L/cow/day)": values})
        fig = px.strip(frame, x="Milk yield (L/cow/day)", title="Observed Milk Yield Distribution")
        fig.update_traces(marker=dict(size=12, color="#1c6e4b", line=dict(width=1, color="#ffffff")))
        fig.update_xaxes(title="Milk yield (L/cow/day)")
        return fig
    fig = px.histogram(values, nbins=min(24, max(6, int(len(values) ** 0.5))), title="Observed Milk Yield Distribution")
    fig.update_xaxes(title="Milk yield (L/cow/day)")
    fig.update_yaxes(title="Observation count")
    return fig


def yield_by_category(data: pd.DataFrame, category: str):
    if data.empty or category not in data:
        return go.Figure()
    summary = data.dropna(subset=[category, "milk_yield_liters"]).groupby(category, as_index=False)["milk_yield_liters"].mean()
    summary.columns = [category, "Average yield (L/cow/day)"]
    return px.bar(summary.sort_values("Average yield (L/cow/day)", ascending=False), x=category, y="Average yield (L/cow/day)", title=f"Average Yield by {category.replace('_', ' ').title()}")


def prediction_vs_actual(actual: list[float], predicted: list[float]):
    frame = pd.DataFrame({"Actual (L/cow/day)": actual, "Predicted (L/cow/day)": predicted})
    fig = px.scatter(frame, x="Actual (L/cow/day)", y="Predicted (L/cow/day)", title="Predicted versus Actual Yield")
    limits = [min(frame.min()), max(frame.max())]
    fig.add_trace(go.Scatter(x=limits, y=limits, mode="lines", name="Perfect prediction", line=dict(dash="dash")))
    return fig


def trend_chart(data: pd.DataFrame):
    prepared = prepare_trend_data(data)
    fig = go.Figure()
    fig.update_layout(title="Average Milk Yield over Time", xaxis_title="Record date", yaxis_title="Yield (L/cow/day)")
    if prepared.distinct_dates < 2:
        fig.add_annotation(text="More than one distinct dated record is required", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    fig.add_trace(go.Scatter(
        x=prepared.frame["record_date"],
        y=prepared.frame["milk_yield_liters"],
        mode="lines+markers",
        line=dict(color="#1c6e4b", width=2.5),
        marker=dict(size=7),
        name="Observed average",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} L/cow/day<extra></extra>",
    ))
    fig.update_xaxes(type="date", tickformat="%d %b %Y", dtick="D1")
    fig.update_yaxes(rangemode="tozero")
    return fig
