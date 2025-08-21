from pathlib import Path
from typing import Iterable, List, Optional, Sequence
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


def plot_roc_pr_curves(y_pred_list, y_true_list, labels=None, save_path: Path = None):

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
    y_true: np.ndarray, y_pred: np.ndarray, labels=None, save_path: Path = None
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


def plot_calibration_curve(
    y_true, y_pred, n_bins=10, label="Model", save_path: Path = None
):
    fig, (ax_cal, ax_hist) = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={"height_ratios": [2, 1]})

    # Courbe calibration - uniform
    prob_true_u, prob_pred_u = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="uniform"
    )
    ax_cal.plot(prob_pred_u, prob_true_u, "o-", label=f"{label} (uniform)")

    # Courbe calibration - quantile
    prob_true_q, prob_pred_q = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="quantile"
    )
    ax_cal.plot(prob_pred_q, prob_true_q, "s-", label=f"{label} (quantile)")

    # Ligne parfaite
    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    ax_cal.set_xlabel("Probabilité prédite")
    ax_cal.set_ylabel("Probabilité observée")
    ax_cal.set_title("Courbes de calibration")
    ax_cal.legend()
    ax_cal.grid(True)

    # Histogramme des proba prédites
    ax_hist.hist(y_pred, bins=n_bins, range=(0, 1), edgecolor="black")
    ax_hist.set_xlabel("Probabilité prédite")
    ax_hist.set_ylabel("Fréquence")
    ax_hist.set_title("Distribution des probabilités")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
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


def plot_feature_importances(
    feature_importances: Iterable[float],
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
    save_path: Optional[str] = None,
    title_top: str = "Top-K Feature Importances",
    title_all: str = "All Feature Importances (sorted)",
) -> None:
    # -- Préparation des données
    fi = np.asarray(list(feature_importances), dtype=float)
    n_features = fi.shape[0]
    if n_features == 0:
        raise ValueError("feature_importances est vide.")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names ({len(feature_names)}) doit avoir la même longueur "
                f"que feature_importances ({n_features})."
            )

    # Tri décroissant
    order = np.argsort(fi)[::-1]
    fi_sorted = fi[order]
    names_sorted = [feature_names[i] for i in order]

    k = max(1, min(top_k, n_features))
    top_vals = fi_sorted[:k]
    top_names = names_sorted[:k]

    # -- Figure & subplots
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Graphe du haut : Top-K avec labels
    ax_top.bar(range(k), top_vals)
    ax_top.set_xticks(range(k))
    ax_top.set_xticklabels(top_names, rotation=45, ha="right")
    ax_top.set_ylabel("Importance")
    ax_top.set_title(title_top)
    ax_top.margins(x=0.01)

    # Graphe du bas : All sorted sans labels en X
    ax_bottom.bar(range(n_features), fi_sorted)
    ax_bottom.set_xticks([])  # pas de labels en X
    ax_bottom.set_ylabel("Importance")
    ax_bottom.set_title(title_all)
    ax_bottom.margins(x=0.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()
