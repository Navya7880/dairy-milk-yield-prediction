import pandas as pd

from src.visualizations import prepare_trend_data, trend_chart, yield_distribution


def test_empty_trend_data_is_safe():
    prepared = prepare_trend_data(pd.DataFrame(columns=["record_date", "milk_yield_liters"]))
    assert prepared.valid_count == 0
    assert prepared.distinct_dates == 0
    assert len(trend_chart(pd.DataFrame()).data) == 0


def test_invalid_dates_and_mixed_dates_are_aggregated_by_date():
    data = pd.DataFrame({
        "record_date": ["24 Oct 2026 18:30", "bad", "2026-10-24", "2026-10-25"],
        "milk_yield_liters": [20, 99, 22, 18],
    })
    prepared = prepare_trend_data(data)
    assert prepared.valid_count == 2
    assert prepared.distinct_dates == 2
    assert prepared.frame["record_date"].is_monotonic_increasing
    assert prepared.frame.iloc[0]["milk_yield_liters"] == 21


def test_single_valid_date_is_not_drawn_as_trend():
    data = pd.DataFrame({"record_date": ["2026-10-24", "bad"], "milk_yield_liters": [20, 21]})
    fig = trend_chart(data)
    assert len(fig.data) == 0
    assert fig.layout.annotations


def test_single_observation_uses_strip_plot():
    fig = yield_distribution(pd.DataFrame({"milk_yield_liters": [20]}))
    assert len(fig.data) == 1
    assert fig.data[0].type in {"strip", "box"}


def test_empty_distribution_is_labelled():
    fig = yield_distribution(pd.DataFrame())
    assert len(fig.data) == 0
    assert fig.layout.annotations
