from pathlib import Path
from typing import List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
from sklearn.calibration import calibration_curve


def plot_roc_pr_curves(y_pred_list, y_true_list, labels=None, save_path=None):

    # Forcer en liste si données uniques
    if not isinstance(y_pred_list, list):
        y_pred_list = [y_pred_list]
        y_true_list = [y_true_list]

    # Labels par défaut
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(y_pred_list))]

    # Créer la figure avec 2 sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ----------- Courbes ROC -----------
    ax = axes[0]
    for y_true, y_pred, label in zip(y_true_list, y_pred_list, labels):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid()

    # ----------- Courbes Precision-Recall -----------
    ax = axes[1]
    for y_true, y_pred, label in zip(y_true_list, y_pred_list, labels):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.2f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.grid()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels=None, save_path=None
):

    y_pred = y_pred.round().astype(int)
    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = [0, 1]  # Assuming binary classification

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(y_true, y_pred, n_bins=10, label=None, save_path=None):
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker="o", label=label or "Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Probabilité réelle")
    plt.title("Courbe de calibration")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_all_plots(
    y_pred_list: List[np.ndarray] | np.ndarray,
    y_true_list: List[np.ndarray] | np.ndarray,
    save_dir: Path,
    labels: Optional[List[str]] = None,
) -> None:
    """
    Appelle toutes les fonctions de plot et sauvegarde les figures dans save_dir.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(y_pred_list, list):
        y_pred_list = [y_pred_list]
        y_true_list = [y_true_list]

    roc_pr_path = save_dir / "roc_pr_curves.png"
    plot_roc_pr_curves(
        y_pred_list, y_true_list, labels=labels, save_path=str(roc_pr_path)
    )

    cm_path = save_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true_list[0], y_pred_list[0], save_path=str(cm_path))

    calib_path = save_dir / "calibration_curve.png"
    plot_calibration_curve(
        y_true_list[0],
        y_pred_list[0],
        n_bins=10,
        save_path=str(calib_path),
    )

    with open(save_dir / "metrics.txt", "w") as f:
        report = classification_report(
            y_true_list[0], y_pred_list[0].round().astype(int), digits=4
        )
        f.write(report)
