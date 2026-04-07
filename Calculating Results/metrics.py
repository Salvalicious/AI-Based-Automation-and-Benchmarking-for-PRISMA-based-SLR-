import numpy as np
from math import sqrt

def confusion_counts(y_true, y_pred):
    """
    Compute confusion matrix values:
    TP, FP, TN, FN
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, FP, TN, FN

def binomial_ci(p, n, z=1.96):
    """
    Wald 95% confidence interval for a proportion
    """
    if n == 0:
        return (0.0, 0.0)

    se = sqrt((p * (1 - p)) / n)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return lower, upper


def agresti_coull_ci(x, n, z=1.96):
    """
    Agresti Couli approximation for 95% confidence interval
    """
    if n == 0:
        return (0.0, 0.0)

    n_tilde = n + z**2
    p_tilde = (x + (z**2) / 2) / n_tilde

    se = sqrt((p_tilde * (1 - p_tilde)) / n_tilde)

    lower = max(0.0, p_tilde - z * se)
    upper = min(1.0, p_tilde + z * se)

    return lower, upper

def work_saved(TN, FN, total):
    """
    Percentage of papers NOT sent to humans
    """
    return (TN + FN) / total if total > 0 else 0

def wss_at_recall(recall_value, work_saved_value):
    """
    Work Saved over Sampling at Recall R
    """
    return work_saved_value - (1 - recall_value)

def recall(TP, FN):
    """
    Sensitivity / Recall
    Most important metric for SLR Screening
    """
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def precision(TP, FP):
    """
    Precision of AI_Dataset-selected studies
    """
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def specificity(TN, FP):
    """
    True Negative Rate
    How well irrelevant studies are excluded
    """
    return TN / (TN + FP) if (TN + FP) > 0 else 0.0


def accuracy(TP, FP, TN, FN):
    """
    Overall accuracy (use cautiously due to class imbalance)
    """
    total = TP + FP + TN + FN
    return (TP + TN) / total if total > 0 else 0.0

def balanced_accuracy(TP, FP, TN, FN):
    """
    Balanced Accuracy = (Sensitivity + Specificity) / 2
    More reliable than accuracy for imbalanced Golden_and_AI_Datasets
    """
    sens = recall(TP, FN)  # Sensitivity/Recall
    spec = specificity(TN, FP)
    return (sens + spec) / 2 if (sens + spec) > 0 else 0.0


def f1_score(TP, FP, FN):
    """
    Harmonic mean of precision and recall
    """
    p = precision(TP, FP)
    r = recall(TP, FN)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def screening_metrics(y_true, y_pred):
    TP, FP, TN, FN = confusion_counts(y_true, y_pred)
    total = TP + FP + TN + FN

    # Core metrics
    rec = recall(TP, FN)
    prec = precision(TP, FP)
    spec = specificity(TN, FP)
    acc = accuracy(TP, FP, TN, FN)
    bal_acc = balanced_accuracy(TP, FP, TN, FN)
    f1 = f1_score(TP, FP, FN)
    ws = work_saved(TN, FN, total)
    recall_CI = binomial_ci(rec, TP + FN)
    precision_CI = binomial_ci(prec, TP + FP)
    specificity_CI = binomial_ci(spec, TN + FP)
    accuracy_CI = binomial_ci(acc, total)
    bal_acc_CI = binomial_ci(bal_acc, total)
    recall_AC_CI = agresti_coull_ci(TP, TP + FN)
    precision_AC_CI = agresti_coull_ci(TP, TP + FP)
    specificity_AC_CI = agresti_coull_ci(TN, TN + FP)
    accuracy_AC_CI = agresti_coull_ci(TP + TN, total)

    recall_low, recall_high = recall_AC_CI
    spec_low, spec_high = specificity_AC_CI
    bal_acc_AC_CI = (
        (recall_low + spec_low) / 2,
        (recall_high + spec_high) / 2
    )
    metrics = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Recall": rec,
        "Precision": prec,
        "Specificity": spec,
        "Accuracy": acc,
        "BalancedAccuracy": bal_acc,
        "F1": f1,
        "WorkSaved": ws,
        "WSS@95": wss_at_recall(rec, ws),
        "Recall_CI": recall_CI,
        "Precision_CI": precision_CI,
        "Specificity_CI": specificity_CI,
        "Accuracy_CI":accuracy_CI,
        "BalancedAccuracy_CI":bal_acc_CI,
        "recall_AC_CI": recall_AC_CI,
        "precision_AC_CI": precision_AC_CI,
        "Specificity_AC_CI":specificity_AC_CI,
        "accuracy_AC_CI":accuracy_AC_CI,
        "bal_acc_AC_CI":bal_acc_AC_CI,
    }

    return metrics