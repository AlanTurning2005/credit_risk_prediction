import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from pathlib import Path
import pandas as pd

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Tính toán AUC, Gini và KS từ nhãn thực tế và xác suất dự báo."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    gini = float(2.0 * auc - 1.0)
    
    bad_scores = y_pred_arr[y_true_arr == 1]
    good_scores = y_pred_arr[y_true_arr == 0]
    ks = float(ks_2samp(bad_scores, good_scores).statistic)
    
    return {"auc": auc, "gini": gini, "ks": ks}


def load_processed_data(data_dir: Path | str | None = None):
    """Nạp train.csv, valid.csv, test.csv và tách X, y (TARGET)."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)
        
    dfs = {name: pd.read_csv(data_dir / f"{name}.csv") for name in ("train", "valid", "test")}
    splits = {}
    for name, df in dfs.items():
        X = df.drop(columns=["TARGET"]) if "TARGET" in df.columns else df.copy()
        y = df["TARGET"].to_numpy(dtype=int) if "TARGET" in df.columns else None
        splits[name] = (X, y)
    return splits


from src.scorecard import WoEScorecard

def run_clean_fit(splits: dict, random_state: int = 42) -> dict:
    """Fit WoEScorecard sạch trên train, đo AUC/Gini/KS trên train, valid, test."""
    X_train, y_train = splits["train"]
    card = WoEScorecard(random_state=random_state)
    card.fit(X_train, y_train)

    results = {}
    for name in ("train", "valid", "test"):
        X, y = splits[name]
        if y is not None:
            probs = card.predict_proba(X)[:, 1]
            results[name] = calculate_metrics(y, probs)
    return {"card": card, "metrics": results}


def run_leakage_experiment(splits: dict, random_state: int = 42) -> dict:
    """Thí nghiệm rò rỉ nhãn: fit trên (train + valid), đo lại trên valid."""
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    
    X_leak = pd.concat([X_train, X_valid], ignore_index=True)
    y_leak = np.concatenate([y_train, y_valid])
    
    card = WoEScorecard(random_state=random_state)
    card.fit(X_leak, y_leak)
    
    probs_valid = card.predict_proba(X_valid)[:, 1]
    return calculate_metrics(y_valid, probs_valid)


def run_shuffled_label_experiment(
    splits: dict, random_state: int = 42, iv_threshold: float = 0.0
) -> dict:
    """Thí nghiệm xáo trộn nhãn: fit trên train có nhãn ngẫu nhiên, đo trên valid."""
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]

    rng = np.random.default_rng(random_state)
    y_shuffled = rng.permutation(y_train)

    card = WoEScorecard(random_state=random_state, iv_threshold=iv_threshold)
    card.fit(X_train, y_shuffled)

    probs_valid = card.predict_proba(X_valid)[:, 1]
    return calculate_metrics(y_valid, probs_valid)


def main():
    print("=== NAP DU LIEU ===")
    splits = load_processed_data()
    for name in ("train", "valid", "test"):
        print(f"So dong {name}: {len(splits[name][0])}")

    print("\n=== BƯỚC 3: FIT SACH ===")
    clean_res = run_clean_fit(splits, random_state=42)
    for name, m in clean_res["metrics"].items():
        print(f"[{name.upper()}] AUC: {m['auc']:.4f} | Gini: {m['gini']:.4f} | KS: {m['ks']*100:.2f}%")

    print("\n=== BƯỚC 4: THÍ NGHIỆM RÒ RỈ ===")
    leak_m = run_leakage_experiment(splits, random_state=42)
    diff_auc = leak_m["auc"] - clean_res["metrics"]["valid"]["auc"]
    print(f"[VALID LEAK] AUC: {leak_m['auc']:.4f} | Gini: {leak_m['gini']:.4f} | KS: {leak_m['ks']*100:.2f}%")
    print(f"Chenh lech AUC (leak - clean): {diff_auc:+.4f}")

    print("\n=== BƯỚC 5: XÁO TRỘN NHÃN ===")
    shuf_m = run_shuffled_label_experiment(splits, random_state=42, iv_threshold=0.0)
    print(f"[VALID SHUFFLED] AUC: {shuf_m['auc']:.4f} | Gini: {shuf_m['gini']:.4f} | KS: {shuf_m['ks']*100:.2f}%")

if __name__ == "__main__":
    main()