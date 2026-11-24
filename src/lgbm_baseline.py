from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS

def build_preprocessor() -> ColumnTransformer:
    """Tạo ColumnTransformer tiền xử lý dữ liệu chuẩn theo notebook 02."""
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS),
    ])


def load_processed_data(data_dir: Path | str | None = None) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Nạp train.csv, valid.csv, test.csv và tách X, y (TARGET)."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    splits = {}
    for name in ("train", "valid", "test"):
        df = pd.read_csv(data_dir / f"{name}.csv")
        X = df.drop(columns=["TARGET"])
        y = df["TARGET"].to_numpy(dtype=int)
        splits[name] = (X, y)
    return splits


from lightgbm import LGBMClassifier

def fit_and_predict(splits: dict, random_state: int = 23102026) -> tuple[LGBMClassifier, ColumnTransformer, dict[str, np.ndarray]]:
    """Fit preprocessor và LGBMClassifier chỉ trên train, dự báo xác suất cho 3 tập."""
    preprocessor = build_preprocessor()
    X_train, y_train = splits["train"]
    X_train_proc = preprocessor.fit_transform(X_train)

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.04,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train_proc, y_train)

    probs = {}
    for name in ("train", "valid", "test"):
        X, _ = splits[name]
        X_proc = preprocessor.transform(X)
        probs[name] = model.predict_proba(X_proc)[:, 1]

    return model, preprocessor, probs



from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Tính 5 chỉ số đánh giá: AUC, Gini, KS, PR-AUC, Brier score."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    gini = float(2.0 * auc - 1.0)

    bad_scores = y_pred_arr[y_true_arr == 1]
    good_scores = y_pred_arr[y_true_arr == 0]
    ks = float(ks_2samp(bad_scores, good_scores).statistic)

    pr_auc = float(average_precision_score(y_true_arr, y_pred_arr))
    brier = float(brier_score_loss(y_true_arr, y_pred_arr))

    return {"auc": auc, "gini": gini, "ks": ks, "pr_auc": pr_auc, "brier": brier}


def main():
    print("=== DANG NAP DU LIEU ===")
    splits = load_processed_data()
    for name in ("train", "valid", "test"):
        print(f"So dong {name}: {len(splits[name][0])}")

    print("\n=== DANG HUAN LUYEN LIGHTGBM BASELINE ===")
    model, preprocessor, probs = fit_and_predict(splits)

    print("\n=== KET QUA DANH GIA TREN 3 TAP ===")
    for name in ("train", "valid", "test"):
        y_true = splits[name][1]
        m = calculate_metrics(y_true, probs[name])
        print(f"[{name.upper()}] AUC: {m['auc']:.4f} | Gini: {m['gini']:.4f} | KS: {m['ks']*100:.2f}% | PR-AUC: {m['pr_auc']:.4f} | Brier: {m['brier']:.4f}")

if __name__ == "__main__":
    main()

