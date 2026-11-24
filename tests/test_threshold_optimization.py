import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from src.threshold_optimization import (
    optimize_threshold,
    pd_to_credit_score,
)
import src.threshold_optimization as to_mod


def test_pd_to_credit_score_known_points():
    """Verify standard FICO mapping at defined anchor points.
    
    PD = 0.0001 (safest bound) -> 850
    PD = 0.5000 (neutral log-odds 0.0) -> 575
    PD = 0.9999 (default bound) -> 300
    """
    pd_inputs = np.array([0.0001, 0.5000, 0.9999])
    scores = pd_to_credit_score(pd_inputs, min_score=300, max_score=850)

    assert scores[0] == 850
    assert scores[1] == 575
    assert scores[2] == 300


def test_pd_to_credit_score_clipping_bounds():
    """Verify that extreme or out-of-range probability inputs remain strictly clamped."""
    extreme_inputs = np.array([-1.0, 0.0, 1.0, 2.5])
    scores = pd_to_credit_score(extreme_inputs, min_score=300, max_score=850)

    assert scores[0] == 850  # Clamped to eps -> 850
    assert scores[1] == 850  # Clamped to eps -> 850
    assert scores[2] == 300  # Clamped to 1-eps -> 300
    assert scores[3] == 300  # Clamped to 1-eps -> 300
    assert np.all((scores >= 300) & (scores <= 850))


def test_pd_to_credit_score_monotonicity():
    """Verify that credit score monotonically decreases as PD increases."""
    pd_range = np.linspace(0.001, 0.999, 100)
    scores = pd_to_credit_score(pd_range)

    # Every step should be non-increasing
    score_diffs = np.diff(scores)
    assert np.all(score_diffs <= 0)


def test_pd_to_credit_score_custom_scale():
    """Verify calculation with custom score ranges."""
    score_mid = pd_to_credit_score(0.5, min_score=0, max_score=1000)
    assert score_mid == 500


def test_pd_to_credit_score_empty_input():
    """Verify behavior on empty array input."""
    empty_scores = pd_to_credit_score(np.array([]))
    assert len(empty_scores) == 0
    assert issubclass(empty_scores.dtype.type, np.integer)


def test_optimize_threshold_hand_calculated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify optimize_threshold against an exact hand-calculated cost-benefit matrix.
    
    Data:
      Index 0: Good (TARGET=0), PD=0.10
      Index 1: Good (TARGET=0), PD=0.20
      Index 2: Bad  (TARGET=1), PD=0.30
      Index 3: Bad  (TARGET=1), PD=0.40
    
    Financial parameters:
      profit_margin = 0.20
      loss_given_default = 0.50
    
    Profit breakdown across thresholds:
      th in [0.05, 0.10]: approved = 0, profit = 0.0
      th in [0.11, 0.20]: approved = {0}, profit = 1 * 0.20 = 0.20
      th in [0.21, 0.30]: approved = {0, 1}, profit = 2 * 0.20 = 0.40 (Global maximum)
      th in [0.31, 0.40]: approved = {0, 1, 2}, profit = 2 * 0.20 - 1 * 0.50 = -0.10
      th in [0.41, 0.94]: approved = {0, 1, 2, 3}, profit = 2 * 0.20 - 2 * 0.50 = -0.60
    
    Optimal threshold identified must be 0.21, yielding net profit = 0.40 units.
    CRITICAL: output_config_path is isolated to tmp_path to protect config/threshold_config.json.
    """
    test_csv = tmp_path / "synthetic_test.csv"
    pd.DataFrame({"TARGET": [0, 0, 1, 1]}).to_csv(test_csv, index=False)

    class MockModel:
        def predict(self, ds, verbose=0):
            return np.array([0.10, 0.20, 0.30, 0.40]).reshape(-1, 1)

    monkeypatch.setattr(tf.keras.models, "load_model", lambda path: MockModel())
    monkeypatch.setattr(to_mod, "get_dataset", lambda path, **kwargs: None)

    out_config_path = tmp_path / "custom_threshold_config.json"

    result_config = optimize_threshold(
        model_path=str(tmp_path / "fake_model.keras"),
        test_csv_path=str(test_csv),
        profit_margin=0.20,
        loss_given_default=0.50,
        output_config_path=str(out_config_path),
    )

    # Optimal threshold verification
    assert np.isclose(result_config["optimal_threshold"], 0.21)
    assert np.isclose(result_config["profit_margin"], 0.20)
    assert np.isclose(result_config["loss_given_default"], 0.50)
    assert np.isclose(result_config["expected_approval_rate_%"], 50.0)
    assert np.isclose(result_config["expected_portfolio_bad_rate_%"], 0.0)

    # File output verification
    assert out_config_path.exists()
    saved_data = json.loads(out_config_path.read_text(encoding="utf-8"))
    assert saved_data["optimal_threshold"] == 0.21
    assert "risk_tiers" in saved_data


def test_optimize_threshold_severe_loss_regime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify that severe penalty for default forces a lower, conservative threshold."""
    test_csv = tmp_path / "synthetic_test_severe.csv"
    pd.DataFrame({"TARGET": [0, 1, 0, 1]}).to_csv(test_csv, index=False)

    class MockModel:
        def predict(self, ds, verbose=0):
            # Interleaved PDs
            return np.array([0.08, 0.12, 0.25, 0.35]).reshape(-1, 1)

    monkeypatch.setattr(tf.keras.models, "load_model", lambda path: MockModel())
    monkeypatch.setattr(to_mod, "get_dataset", lambda path, **kwargs: None)

    out_config_path = tmp_path / "severe_loss_config.json"

    # Extreme LGD of 1.0 against tiny profit margin of 0.05
    result_config = optimize_threshold(
        model_path=str(tmp_path / "fake_model.keras"),
        test_csv_path=str(test_csv),
        profit_margin=0.05,
        loss_given_default=1.00,
        output_config_path=str(out_config_path),
    )

    # The model must stop before approving the bad applicant at 0.12
    assert result_config["optimal_threshold"] <= 0.12
    assert out_config_path.exists()
