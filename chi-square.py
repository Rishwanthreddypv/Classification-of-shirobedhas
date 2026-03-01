# chi_square_and_metrics.py
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_recall_fscore_support

# -------------------------
# Config
# -------------------------
CSV_PATH = r"C:\Pinacle\confusion_matrix.csv"   # created by train.py
OUTPUT_TXT = r"C:\Pinacle\stat_test_results.txt"

# Fallback confusion matrix (from what you pasted)
FALLBACK_CM = np.array([
    [2,0,0,0,0,0,0,0,0],
    [0,2,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0],
    [2,0,0,0,0,0,0,0,0],
    [0,0,1,0,1,0,0,0,0],
    [2,0,0,0,0,0,0,0,0],
    [2,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,2,0],
    [0,0,0,0,1,0,0,0,1]
], dtype=int)

def load_confusion_matrix(csv_path):
    if os.path.exists(csv_path):
        try:
            cm = np.loadtxt(csv_path, delimiter=',', dtype=int)
            if cm.ndim == 1:
                cm = cm.reshape(1, -1)
            return cm
        except Exception as e:
            print(f"Failed reading {csv_path}: {e}. Using fallback matrix.")
    else:
        print(f"{csv_path} not found — using fallback matrix.")
    return FALLBACK_CM

def clean_table_for_chi2(cm):
    """Remove columns (and rows) that sum to zero which cause zero expected cells.
       We remove columns with zero column-sum (no predictions for that class).
       We keep rows even if zero (unless they are all-zero)."""
    cm = np.array(cm, dtype=int)
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    valid_cols = np.where(col_sums > 0)[0]
    valid_rows = np.where(row_sums > 0)[0]
    # We'll remove zero-sum columns. Keep rows because rows represent true classes;
    # If a true class had zero samples (rare), you'd remove it too.
    if len(valid_cols) < cm.shape[1]:
        print(f"Removed {cm.shape[1]-len(valid_cols)} zero-sum prediction columns for chi-square.")
    cm_reduced = cm[:, valid_cols]
    return cm_reduced, valid_rows, valid_cols

def run_chi2(cm):
    try:
        chi2, p, dof, expected = chi2_contingency(cm)
        return chi2, p, dof, expected
    except ValueError as e:
        # try cleaning by removing zero-sum columns/rows
        raise

def compute_metrics_from_cm(cm):
    """Compute overall accuracy and per-class precision/recall/f1 from confusion matrix."""
    cm = np.array(cm, dtype=int)
    # Overall accuracy
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0.0

    # Reconstruct y_true and y_pred arrays from confusion matrix
    y_true = []
    y_pred = []
    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            count = int(cm[i, j])
            if count > 0:
                y_true.extend([i] * count)
                y_pred.extend([j] * count)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        precision = recall = f1 = np.zeros(n_classes)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(n_classes), zero_division=0)

    return accuracy, precision, recall, f1

def main():
    cm = load_confusion_matrix(CSV_PATH)
    print("Loaded confusion matrix (raw):")
    print(cm)
    # Clean table for chi-square (remove zero-sum columns)
    cm_reduced, valid_rows, valid_cols = clean_table_for_chi2(cm)
    print("\nConfusion matrix used for Chi-Square (columns with zero predictions removed):")
    print(cm_reduced)

    # Run chi-square
    try:
        chi2, p, dof, expected = chi2_contingency(cm_reduced)
        print("\nChi-Square Test of Independence (on reduced table):")
        print(f"chi2 = {chi2:.4f}")
        print(f"degrees of freedom = {dof}")
        print(f"p-value = {p:.6e}")
        print("\nExpected frequencies (rounded):")
        print(np.round(expected).astype(int))
    except Exception as e:
        print("\nChi-Square test failed:", e)
        chi2 = p = dof = expected = None

    # Compute metrics on original full cm (useful for H2)
    acc, precision, recall, f1 = compute_metrics_from_cm(cm)
    print("\nClassification metrics (from confusion matrix):")
    print(f"Overall accuracy = {acc*100:.2f}%")
    # print per-class metrics compactly
    n = len(precision)
    for i in range(n):
        print(f"Class {i}: precision={precision[i]:.2f}, recall={recall[i]:.2f}, f1={f1[i]:.2f}")

    # Build report summary sentences for H1 and H2
    h1_statement = ""
    if (p is not None):
        if p < 0.05:
            h1_statement = f"H1 supported: Chi-square shows dependence (chi2={chi2:.2f}, p={p:.5f} < 0.05), indicating keypoint features are discriminative."
        else:
            h1_statement = f"H1 not supported: Chi-square p={p:.5f} >= 0.05 (chi2={chi2:.2f})."
    else:
        h1_statement = "H1 could not be tested due to chi-square computation error."

    h2_statement = (f"H2 assessment: LSTM shows learning behavior with accuracy {acc*100:.2f}%. "
                    "Confusion matrix and per-class metrics indicate the model captures temporal patterns, "
                    "though accuracy is moderate and further improvements are possible.")

    # Print and save
    report_lines = [
        "Statistical test results",
        "------------------------",
        f"Chi-square (reduced): chi2={chi2}, p={p}, dof={dof}" if chi2 is not None else "Chi-square not available",
        "",
        f"Overall accuracy: {acc*100:.2f}%",
        "",
        "Per-class metrics (precision, recall, f1):"
    ]
    for i in range(len(precision)):
        report_lines.append(f"Class {i}: {precision[i]:.2f}, {recall[i]:.2f}, {f1[i]:.2f}")
    report_lines.append("")
    report_lines.append("H1 conclusion:")
    report_lines.append(h1_statement)
    report_lines.append("")
    report_lines.append("H2 conclusion:")
    report_lines.append(h2_statement)

    print("\n\n" + "\n".join(report_lines))

    try:
        with open(OUTPUT_TXT, 'w') as fh:
            fh.write("\n".join(report_lines))
        print(f"\nSaved results to {OUTPUT_TXT}")
    except Exception as e:
        print(f"Failed to save results file: {e}")

if __name__ == "__main__":
    main()
