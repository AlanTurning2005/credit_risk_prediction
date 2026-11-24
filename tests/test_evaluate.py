from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from src.evaluate import (
    calculate_ks_statistic,
    evaluate_model,
    generate_decile_table,
)


def test_calculate_ks_statistic_perfect_and_inverse_ranking():
    """Verify KS statistic on perfect ranking, inverse ranking, and identical distributions.
    
    Kolmogorov-Smirnov statistic measures the maximum distance between two empirical CDFs.
    Non-overlapping distributions result in KS = 1.0.
    Identical distributions result in KS = 0.0.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Perfect ranking: all bads have strictly higher predicted probability than all goods
    y_pred_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    stat_perf, pval_perf = calculate_ks_statistic(y_true, y_pred_perfect)
    assert np.isclose(stat_perf, 1.0)
    assert pval_perf < 0.05

    # Inverted ranking: non-overlapping distributions still have maximum CDF separation 1.0
    y_pred_inverse = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    stat_inv, pval_inv = calculate_ks_statistic(y_true, y_pred_inverse)
    assert np.isclose(stat_inv, 1.0)

    # Identical predictions across classes: zero separation power
    y_pred_identical = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    stat_ident, pval_ident = calculate_ks_statistic(y_true, y_pred_identical)
    assert np.isclose(stat_ident, 0.0)
    assert np.isclose(pval_ident, 1.0)


def test_calculate_ks_statistic_known_manual_values():
    """Verify KS statistic against hand-calculated step-CDF distance.
    
    Goods: [0.2, 0.7], Bads: [0.6, 0.8]
    Evaluated points:
      at x = 0.2: ECDF_bad = 0.0, ECDF_good = 0.5 -> |0.0 - 0.5| = 0.5
      at x = 0.6: ECDF_bad = 0.5, ECDF_good = 0.5 -> |0.5 - 0.5| = 0.0
      at x = 0.7: ECDF_bad = 0.5, ECDF_good = 1.0 -> |0.5 - 1.0| = 0.5
      at x = 0.8: ECDF_bad = 1.0, ECDF_good = 1.0 -> |1.0 - 1.0| = 0.0
    Supremum difference is exactly 0.5.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.2, 0.7, 0.6, 0.8])
    stat, _ = calculate_ks_statistic(y_true, y_pred)
    assert np.isclose(stat, 0.5)


def test_calculate_ks_statistic_single_class_behavior():
    """Verify behavior when input contains only one class.
    
    SciPy ks_2samp receives an empty slice for the missing class, issuing a
    SmallSampleWarning and returning NaN for both statistic and p-value.
    """
    y_true_all_zeros = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])

    with pytest.warns(RuntimeWarning):
        stat, pval = calculate_ks_statistic(y_true_all_zeros, y_pred)
    assert np.isnan(stat)
    assert np.isnan(pval)


def test_calculate_ks_statistic_python_list_input_behavior():
    """Document actual behavior when passing Python lists instead of NumPy ndarrays.
    
    Because y_true == 0 evaluates to boolean False, list indexing y_pred[False]
    resolves to index 0 (y_pred[0]), silently comparing identical scalar values.
    """
    y_true_list = [0, 1]
    y_pred_list = [0.2, 0.8]
    stat, pval = calculate_ks_statistic(y_true_list, y_pred_list)
    assert np.isclose(stat, 0.0)
    assert np.isclose(pval, 1.0)


def test_generate_decile_table_structure_and_totals():
    """Verify table schema, customer conservation, and bad customer sums."""
    rng = np.random.default_rng(20261121)
    n = 200
    y_true = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    y_pred = rng.uniform(0.01, 0.99, size=n)

    table = generate_decile_table(y_true, y_pred, num_bins=10)

    expected_cols = {
        "decile",
        "total_customers",
        "bad_customers",
        "min_prob",
        "max_prob",
        "avg_prob",
        "good_customers",
        "bad_rate_%",
        "cum_bad_%",
        "cum_good_%",
        "ks_%",
    }
    assert expected_cols.issubset(table.columns)

    # Customer conservation
    assert table["total_customers"].sum() == n
    assert table["bad_customers"].sum() == int(y_true.sum())
    assert table["good_customers"].sum() == int((y_true == 0).sum())

    # Decile 1 must correspond to the highest probability segment
    assert table.loc[table["decile"] == 1, "avg_prob"].iloc[0] >= table.loc[table["decile"] == 10, "avg_prob"].iloc[0]


def test_generate_decile_table_monotonic_signal():
    """Verify that a model with strong discriminative power yields monotonic bad rates."""
    rng = np.random.default_rng(42)
    n = 1000
    # True positives have high probabilities, true negatives have low probabilities
    y_true = np.array([0] * 800 + [1] * 200)
    y_pred = np.concatenate([rng.uniform(0.01, 0.40, size=800), rng.uniform(0.50, 0.99, size=200)])

    table = generate_decile_table(y_true, y_pred, num_bins=10)

    # Top risk decile (Decile 1) should have substantially higher bad rate than lowest risk decile
    top_bad_rate = table.loc[table["decile"] == 1, "bad_rate_%"].iloc[0]
    bottom_bad_rate = table.loc[table["decile"] == 10, "bad_rate_%"].iloc[0]
    assert top_bad_rate > bottom_bad_rate
    assert bottom_bad_rate == 0.0


def test_generate_decile_table_empty_input():
    """Verify behavior on empty input arrays."""
    empty_table = generate_decile_table(np.array([]), np.array([]), num_bins=5)
    assert isinstance(empty_table, pd.DataFrame)
    assert len(empty_table) == 0


def test_evaluate_model_raises_on_missing_files(tmp_path: Path):
    """Verify evaluate_model raises FileNotFoundError without touching models/ or data/."""
    missing_model = tmp_path / "non_existent_model.keras"
    missing_test_csv = tmp_path / "non_existent_test.csv"

    with pytest.raises(FileNotFoundError, match="Không tìm thấy file mô hình"):
        evaluate_model(model_path=str(missing_model), test_csv_path=str(missing_test_csv))

    fake_model_file = tmp_path / "fake_model.keras"
    fake_model_file.write_text("placeholder", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Không tìm thấy file test"):
        evaluate_model(model_path=str(fake_model_file), test_csv_path=str(missing_test_csv))
