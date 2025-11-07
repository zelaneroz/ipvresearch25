"""
Reusable evaluation helpers for LLM-based IPV experiments.

The module exposes :class:`LLMEvalPipeline`, which ingests a dictionary that maps
friendly model names to result files (JSON/CSV) plus a ground-truth table.  Once
instantiated, the pipeline can generate the five requested artefacts:

1. ROC-AUC bar plot (binary and multitype†).
2. ROC curve overlays (FPR on X, TPR on Y).
3. Precision vs. residual (= 1 - recall) curves.
4. Tabular metrics (accuracy, precision, F1, ROC-AUC) printed + optionally saved.
5. Multitype confidence waterfall plot (only built when ``task='multitype'``).

† Multitype computations run one-vs-rest per subtype (Physical, Emotional, Sexual)
  and aggregate the per-class ROC-AUCs.

Public API quick reference
--------------------------
LLMEvalPipeline(...):
    Input:
        results_map          - dict ``{name: path_to_predictions}``
        ground_truth         - CSV/JSON path or DataFrame with reference labels
        task                 - ``"binary"`` or ``"multitype"``
        output_dir           - folder to drop generated figures
        positive_label       - positive class name for binary runs
    Output:
        A configured pipeline instance that caches processed predictions +
        truth for reuse across plots.

LLMEvalPipeline.save_and_print_metrics(csv_path=None):
    Input:
        csv_path (optional)  - where to save the summary table
    Output:
        pandas DataFrame indexed by model with accuracy / precision / F1 /
        ROC-AUC columns.

LLMEvalPipeline.plot_roc_auc_bars(filename='roc_auc_bar.png'):
    Input:
        filename             - name of the PNG saved inside ``output_dir``
    Output:
        pathlib.Path to the created image (binary → single bar chart, multitype
        → grouped bars per class).

LLMEvalPipeline.plot_roc_curves(filename_prefix='roc_curve'):
    Input:
        filename_prefix      - base name for one (binary) or three (multitype)
                               overlay ROC plots.
    Output:
        List[pathlib.Path] pointing to each PNG with FPR (x) vs TPR (y).

LLMEvalPipeline.plot_precision_vs_residual(filename_prefix='precision_residual'):
    Input:
        filename_prefix      - base name for the exported plot(s).
    Output:
        List[pathlib.Path] of figures showing precision on the y-axis and
        residual (= 1 - recall) on the x-axis.

LLMEvalPipeline.multi_confidence_score_plot(filename_prefix='confidence_waterfall'):
    Input:
        filename_prefix      - base name for per-model confidence waterfalls.
    Output:
        List[pathlib.Path] with sorted-bar visualizations; only available when
        ``task='multitype'`` because the plot requires subtype confidences.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = ["LLMEvalPipeline"]

MULTI_LABELS: Tuple[str, ...] = ("Physical", "Emotional", "Sexual")
_SCORE_COLUMNS = (
    "y_score",
    "score",
    "scores",
    "prob",
    "probability",
    "confidence",
    "confidence_score",
    "logit",
    "logprob",
)
_LABEL_COLUMNS = ("extracted_label", "prediction", "predicted_label", "label")


def _to_int_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)
    mapped = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"TRUE": 1, "FALSE": 0, "YES": 1, "NO": 0})
    )
    mapped = pd.to_numeric(mapped, errors="coerce").fillna(0)
    return mapped.astype(int)


def _ensure_dataframe(obj: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    path = Path(obj)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    raise ValueError(f"Unsupported file type for {path}")


def _normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(len(df))
    return df


def _prepare_binary_truth(
    ground_truth: pd.DataFrame, positive_label: str
) -> pd.DataFrame:
    df = _normalize_id_column(ground_truth)
    df = df.copy()
    df = df.rename(
        columns={
            "Physical Abuse": "Physical",
            "Emotional Abuse": "Emotional",
            "Sexual Abuse": "Sexual",
        }
    )

    positive_label = positive_label.upper()
    if "label" not in df.columns:
        if "IPV" in df.columns:
            ipv_bool = _to_int_series(df["IPV"]).astype(bool)
            df["label"] = ipv_bool.map({True: "IPV", False: "NOT_IPV"})
        elif set(MULTI_LABELS).issubset(df.columns):
            cols = list(MULTI_LABELS)
            numeric = df[cols].apply(_to_int_series)
            df["label"] = numeric.any(axis=1).map({True: "IPV", False: "NOT_IPV"})
        else:
            raise ValueError(
                "Binary ground truth must contain either 'label', 'IPV', "
                "or the multitype columns."
            )

    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["y_true"] = (df["label"] == positive_label).astype(int)
    return df[["id", "label", "y_true"]]


def _prepare_multilabel_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_id_column(ground_truth)
    df = df.rename(
        columns={
            "Physical Abuse": "Physical",
            "Emotional Abuse": "Emotional",
            "Sexual Abuse": "Sexual",
        }
    )
    missing = [c for c in MULTI_LABELS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Multitype ground truth missing columns: {', '.join(missing)}"
        )
    cols = list(MULTI_LABELS)
    df[cols] = df[cols].apply(_to_int_series)
    return df[["id", *MULTI_LABELS]]


def _try_parse_json_blob(text: str) -> Optional[dict]:
    if not isinstance(text, str) or not text.strip():
        return None
    # Prefer explicit <json> blocks if present.
    match = re.search(r"<json[^>]*>\s*(\{.*?\})\s*</json>", text, re.IGNORECASE | re.DOTALL)
    candidates: List[str] = []
    if match:
        candidates.append(match.group(1))
    else:
        # Grab the first JSON-looking substring.
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _extract_binary_labels(
    df: pd.DataFrame, positive_label: str
) -> pd.Series:
    positive_label = positive_label.upper()
    for col in _LABEL_COLUMNS:
        if col in df.columns:
            labels = (
                df[col]
                .astype(str)
                .str.upper()
                .str.strip()
                .replace({"": "NOT_IPV", "NONE": "NOT_IPV", "N/A": "NOT_IPV"})
            )
            return labels.eq(positive_label).astype(int)

    if "raw_response" in df.columns:
        extracted = []
        for raw in df["raw_response"]:
            parsed = _try_parse_json_blob(raw)
            if isinstance(parsed, dict):
                label = parsed.get("label") or parsed.get("prediction")
                if isinstance(label, str):
                    extracted.append(label.upper().strip() == positive_label)
                    continue
                if isinstance(label, dict):
                    # allow {"IPV": 0.73, "NOT_IPV": 0.27}
                    positive = label.get(positive_label) or label.get("yes")
                    extracted.append(bool(round(float(positive or 0))))
                    continue
            extracted.append(False)
        return pd.Series(extracted, index=df.index, dtype=int)

    raise ValueError("Could not infer binary predictions – add 'extracted_label'.")


def _extract_binary_scores(
    df: pd.DataFrame, positive_label: str
) -> pd.Series:
    for col in _SCORE_COLUMNS:
        if col in df.columns:
            with np.errstate(all="ignore"):
                ser = pd.to_numeric(df[col], errors="coerce")
            if ser.notna().any():
                return ser.fillna(0.5)

    # Look for dict-like probability fields.
    candidate_cols = [c for c in df.columns if isinstance(df[c].iloc[0], dict)]
    positive_label = positive_label.upper()
    for col in candidate_cols:
        values = []
        for entry in df[col]:
            if isinstance(entry, dict):
                val = entry.get(positive_label)
                if val is None:
                    # fall back to first numeric value
                    numeric = [v for v in entry.values() if isinstance(v, (int, float))]
                    val = numeric[0] if numeric else None
                values.append(val)
            else:
                values.append(None)
        ser = pd.Series(values, index=df.index, dtype=float)
        if ser.notna().any():
            return ser.fillna(0.5)

    if "raw_response" in df.columns:
        values = []
        for raw in df["raw_response"]:
            parsed = _try_parse_json_blob(raw)
            val = None
            if isinstance(parsed, dict):
                for key in (
                    "prob",
                    "probability",
                    "confidence",
                    "confidence_score",
                    positive_label,
                ):
                    if key in parsed and isinstance(parsed[key], (int, float)):
                        val = float(parsed[key])
                        break
                if val is None:
                    probs = parsed.get("probs") or parsed.get("probabilities") or parsed.get(
                        "scores"
                    )
                    if isinstance(probs, dict):
                        val = probs.get(positive_label)
            values.append(val)
        ser = pd.Series(values, index=df.index, dtype=float)
        if ser.notna().any():
            return ser.fillna(0.5)

    # Fallback: treat hard predictions as pseudo-scores.
    labels = _extract_binary_labels(df, positive_label)
    return labels.astype(float)


def _extract_multilabel_predictions(df: pd.DataFrame) -> pd.DataFrame:
    result = {}
    # Existing numeric columns.
    for label in MULTI_LABELS:
        if label in df.columns:
            result[label] = df[label].astype(int)

    # Raw lists (e.g., ["Physical", "Emotional"])
    if "extracted_labels" in df.columns:
        def _to_set(x):
            if isinstance(x, list):
                return {str(item).strip().capitalize() for item in x}
            if isinstance(x, str):
                return {part.strip().capitalize() for part in x.split(",")}
            return set()

        sets = df["extracted_labels"].apply(_to_set)
        for label in MULTI_LABELS:
            result.setdefault(label, sets.apply(lambda s, lbl=label: int(lbl in s)))

    # raw_response JSON booleans such as {"physical_abuse": false}
    if "raw_response" in df.columns:
        mapping = {
            "physical": "Physical",
            "physical_abuse": "Physical",
            "emotional": "Emotional",
            "emotional_abuse": "Emotional",
            "sexual": "Sexual",
            "sexual_abuse": "Sexual",
        }
        extracted = {label: [] for label in MULTI_LABELS}
        usable = False
        for raw in df["raw_response"]:
            parsed = _try_parse_json_blob(raw)
            if isinstance(parsed, dict):
                usable = True
                normalized = {mapping[k.lower()]: v for k, v in parsed.items() if k.lower() in mapping}
                for label in MULTI_LABELS:
                    extracted[label].append(int(bool(normalized.get(label, False))))
            else:
                usable = usable or False
                for label in MULTI_LABELS:
                    extracted[label].append(np.nan)
        if usable:
            for label in MULTI_LABELS:
                ser = pd.Series(extracted[label], index=df.index)
                if ser.notna().any():
                    result[label] = ser.fillna(0).astype(int)

    missing = [label for label in MULTI_LABELS if label not in result]
    if missing:
        raise ValueError(
            "Could not derive multitype predictions for columns: "
            + ", ".join(missing)
        )
    return pd.DataFrame(result, index=df.index)


def _safe_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return float("nan")


@dataclass
class _EvalResult:
    name: str
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_scores: Union[np.ndarray, Dict[str, np.ndarray]]
    df_pred: pd.DataFrame


class LLMEvalPipeline:
    """
    Unified interface for evaluating one or more LLM runs against the same dataset.

    Parameters
    ----------
    results_map:
        Mapping of ``model_name -> path`` where each file is JSON/CSV containing
        at least an ``id`` column and predictions.
    ground_truth:
        DataFrame or file containing the reference labels. Must include ``id``.
    task:
        ``'binary'`` or ``'multitype'``.
    output_dir:
        Directory where plots will be exported.
    positive_label:
        String representing the positive class for binary runs (default ``IPV``).
    """

    def __init__(
        self,
        results_map: Mapping[str, Union[str, Path]],
        ground_truth: Union[str, Path, pd.DataFrame],
        task: str = "binary",
        output_dir: Union[str, Path] = "1_LLM_Eval/test_results/figs",
        positive_label: str = "IPV",
    ) -> None:
        if not results_map:
            raise ValueError("results_map cannot be empty.")
        self.results_map = {
            name: Path(path) for name, path in results_map.items()
        }
        self.task = task.lower()
        if self.task not in {"binary", "multitype"}:
            raise ValueError("task must be either 'binary' or 'multitype'.")
        self.positive_label = positive_label
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        truth_df = _ensure_dataframe(ground_truth)
        self.truth = (
            _prepare_binary_truth(truth_df, positive_label)
            if self.task == "binary"
            else _prepare_multilabel_truth(truth_df)
        )

        self._evaluations: Dict[str, _EvalResult] = {}
        self._evaluate_all()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def metrics_table(self) -> pd.DataFrame:
        """Return the core metrics (accuracy/precision/F1/ROC-AUC) per model."""
        rows = []
        for name, result in self._evaluations.items():
            rows.append(
                {
                    "model": name,
                    "accuracy": result.metrics["accuracy"],
                    "precision": result.metrics["precision"],
                    "f1": result.metrics["f1"],
                    "roc_auc": result.metrics["roc_auc"],
                }
            )
        return pd.DataFrame(rows).set_index("model")

    def save_and_print_metrics(self, csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Print the metric summary table and optionally persist it as CSV.
        """
        table = self.metrics_table
        print("\n=== Evaluation Summary ===")
        print(table.round(4))
        if csv_path:
            csv_path = Path(csv_path)
            table.to_csv(csv_path)
            print(f"\nSaved metrics to {csv_path.resolve()}")
        return table

    def plot_roc_auc_bars(self, filename: str = "roc_auc_bar.png") -> Path:
        """
        Draw bar plots of ROC-AUC across models.
        """
        out_path = self.output_dir / filename
        plt.figure(figsize=(8, 5))

        names = list(self._evaluations.keys())
        if self.task == "binary":
            aucs = [self._evaluations[name].metrics["roc_auc"] for name in names]
            plt.bar(names, aucs, color="#3E7CB1")
            plt.ylim(0, 1)
            plt.ylabel("ROC-AUC")
            plt.title("Binary ROC-AUC by Prompt/Model")
        else:
            x = np.arange(len(names))
            width = 0.25
            for idx, cls in enumerate(MULTI_LABELS):
                offsets = x + (idx - 1) * width
                vals = [
                    self._evaluations[name].metrics["per_class_roc_auc"][cls]
                    for name in names
                ]
                plt.bar(offsets, vals, width=width, label=cls)
            plt.xticks(x, names)
            plt.ylim(0, 1)
            plt.ylabel("One-vs-Rest ROC-AUC")
            plt.title("Multitype ROC-AUC by Prompt/Model")
            plt.legend()

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_roc_curves(self, filename_prefix: str = "roc_curve") -> List[Path]:
        """
        Overlay ROC curves for every model. Multitype tasks emit one figure per class.
        """
        outputs: List[Path] = []
        if self.task == "binary":
            out_path = self.output_dir / f"{filename_prefix}.png"
            plt.figure(figsize=(7, 6))
            for name, result in self._evaluations.items():
                fpr, tpr, _ = roc_curve(result.y_true, result.y_scores)
                auc_val = result.metrics["roc_auc"]
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve Comparison")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            outputs.append(out_path)
            return outputs

        # Multitype → per-class figure
        for cls in MULTI_LABELS:
            out_path = self.output_dir / f"{filename_prefix}_{cls.lower()}.png"
            plt.figure(figsize=(7, 6))
            for name, result in self._evaluations.items():
                y_true_cls = result.y_true[:, MULTI_LABELS.index(cls)]
                y_score_cls = result.y_scores[cls]
                fpr, tpr, _ = roc_curve(y_true_cls, y_score_cls)
                auc_val = result.metrics["per_class_roc_auc"][cls]
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve — {cls}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            outputs.append(out_path)
        return outputs

    def plot_precision_vs_residual(
        self, filename_prefix: str = "precision_residual"
    ) -> List[Path]:
        """
        Plot precision on the Y axis and residual (= 1 - recall) on the X axis.
        """
        outputs: List[Path] = []
        if self.task == "binary":
            out_path = self.output_dir / f"{filename_prefix}.png"
            plt.figure(figsize=(7, 6))
            for name, result in self._evaluations.items():
                precision, recall, _ = precision_recall_curve(result.y_true, result.y_scores)
                residual = 1 - recall
                plt.plot(
                    residual,
                    precision,
                    label=f"{name} (AUC={np.trapz(precision, recall):.3f})",
                )
            plt.xlabel("Residual (1 - Recall)")
            plt.ylabel("Precision")
            plt.title("Precision vs Residual")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            outputs.append(out_path)
            return outputs

        for cls in MULTI_LABELS:
            out_path = self.output_dir / f"{filename_prefix}_{cls.lower()}.png"
            plt.figure(figsize=(7, 6))
            for name, result in self._evaluations.items():
                y_true_cls = result.y_true[:, MULTI_LABELS.index(cls)]
                y_score_cls = result.y_scores[cls]
                precision, recall, _ = precision_recall_curve(y_true_cls, y_score_cls)
                residual = 1 - recall
                plt.plot(
                    residual,
                    precision,
                    label=f"{name} ({cls})",
                )
            plt.xlabel("Residual (1 - Recall)")
            plt.ylabel("Precision")
            plt.title(f"Precision vs Residual — {cls}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            outputs.append(out_path)
        return outputs

    def multi_confidence_score_plot(
        self, filename_prefix: str = "confidence_waterfall"
    ) -> List[Path]:
        """
        Build waterfall plots of confidence scores for multitype predictions.
        """
        if self.task != "multitype":
            raise RuntimeError("Confidence plot is only available for multitype tasks.")

        outputs: List[Path] = []
        for name, result in self._evaluations.items():
            df = result.df_pred.copy()
            if "confidence_score" in df.columns:
                confidence = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0.0)
            else:
                # fallback: use max score across subtypes
                scores = np.vstack([result.y_scores[cls] for cls in MULTI_LABELS]).T
                confidence = pd.Series(scores.max(axis=1), index=df.index)
                df["confidence_score"] = confidence

            # Determine dominant predicted class.
            def _top_class(row: pd.Series) -> str:
                for cls in MULTI_LABELS:
                    col = f"{cls}_pred"
                    if col in row and row[col] == 1:
                        return cls
                return "Negative"

            for cls in MULTI_LABELS:
                pred_col = f"{cls}_pred"
                if pred_col not in df.columns:
                    df[pred_col] = (
                        result.y_scores[cls] > 0.5
                    ).astype(int)

            df["PredictedType"] = df.apply(_top_class, axis=1)
            df = df.sort_values("confidence_score", ascending=False).reset_index(drop=True)

            colors = {
                "Physical": "#2ECC71",
                "Emotional": "#F4D03F",
                "Sexual": "#9B59B6",
                "Negative": "#E74C3C",
            }
            df["Color"] = df["PredictedType"].map(colors)

            per_class_auc = result.metrics["per_class_roc_auc"]
            legend_labels = [
                f"{cls} (AUC={per_class_auc[cls]:.2f})" for cls in MULTI_LABELS
            ] + ["Negative"]

            out_path = self.output_dir / f"{filename_prefix}_{name}.png"
            plt.figure(figsize=(12, 4))
            plt.bar(
                x=np.arange(len(df)),
                height=df["confidence_score"],
                color=df["Color"],
                width=1.0,
                edgecolor="none",
            )
            plt.ylim(0, max(1.0, df["confidence_score"].max() * 1.05))
            plt.xlabel("Samples (sorted by confidence)")
            plt.ylabel("Confidence Score")
            plt.title(f"{name} — Multitype Confidence Waterfall")
            handles = [
                plt.Line2D([0], [0], color=colors.get(lbl.split()[0], "#999999"), lw=4)
                for lbl in legend_labels
            ]
            plt.legend(handles, legend_labels, loc="upper right")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            outputs.append(out_path)
        return outputs

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _evaluate_all(self) -> None:
        for name, path in self.results_map.items():
            df_pred = _ensure_dataframe(path)
            df_pred = _normalize_id_column(df_pred)

            if self.task == "binary":
                y_pred = _extract_binary_labels(df_pred, self.positive_label)
                y_scores = _extract_binary_scores(df_pred, self.positive_label)
                merged = self.truth.merge(
                    df_pred.assign(y_pred=y_pred, y_score=y_scores),
                    on="id",
                    how="inner",
                )
                y_true = merged["y_true"].to_numpy()
                y_pred_arr = merged["y_pred"].to_numpy()
                y_score_arr = merged["y_score"].to_numpy()
                metrics = self._compute_binary_metrics(y_true, y_pred_arr, y_score_arr)
                self._evaluations[name] = _EvalResult(
                    name=name,
                    metrics=metrics,
                    y_true=y_true,
                    y_pred=y_pred_arr,
                    y_scores=y_score_arr,
                    df_pred=merged,
                )
            else:
                preds = _extract_multilabel_predictions(df_pred)
                preds = preds.rename(columns={c: f"{c}_pred" for c in preds.columns})
                merged = self.truth.merge(
                    pd.concat([df_pred, preds], axis=1), on="id", how="inner"
                )
                y_true = merged[list(MULTI_LABELS)].to_numpy()
                y_pred_arr = merged[[f"{c}_pred" for c in MULTI_LABELS]].to_numpy()

                # Build scores dict (prefer explicit floats, fall back to ints).
                scores: Dict[str, np.ndarray] = {}
                for cls in MULTI_LABELS:
                    score_col = None
                    for candidate in (
                        f"{cls}_score",
                        f"{cls}_prob",
                        f"{cls}_confidence",
                    ):
                        if candidate in merged.columns:
                            score_col = candidate
                            break
                    if score_col:
                        scores[cls] = merged[score_col].astype(float).to_numpy()
                    else:
                        scores[cls] = y_pred_arr[:, MULTI_LABELS.index(cls)].astype(float)

                metrics = self._compute_multilabel_metrics(
                    y_true=y_true, y_pred=y_pred_arr, y_scores=scores
                )
                self._evaluations[name] = _EvalResult(
                    name=name,
                    metrics=metrics,
                    y_true=y_true,
                    y_pred=y_pred_arr,
                    y_scores=scores,
                    df_pred=merged,
                )

    def _compute_binary_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
    ) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": _safe_auc(y_true, y_scores),
        }

    def _compute_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["accuracy"] = float((y_true == y_pred).all(axis=1).mean())
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

        per_class_auc = {}
        for idx, cls in enumerate(MULTI_LABELS):
            per_class_auc[cls] = _safe_auc(y_true[:, idx], y_scores[cls])
        metrics["per_class_roc_auc"] = per_class_auc
        metrics["roc_auc"] = float(
            np.nanmean(list(per_class_auc.values()))
        )
        return metrics
