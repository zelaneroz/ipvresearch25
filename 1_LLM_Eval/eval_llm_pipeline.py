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

† Multitype computations still track subtype scores internally, but plots now rely
  on exact-match correctness vs. model confidence so every figure captures how
  often the model predicted the full label set perfectly.

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
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "LLMEvalPipeline",
    "compute_binary_metrics_detailed",
    "compute_multitype_metrics_per_subgroup",
    "plot_confusion_matrix",
    "plot_roc_curve_binary",
    "plot_precision_recall_curve_binary",
    "plot_per_class_f1_bar_chart",
    "append_binary_results_to_json",
    "append_multitype_results_to_json",
]

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
_CONFIDENCE_COLUMNS = (
    "confidence_score",
    "confidence",
    "avg_confidence",
    "confidenceScore",
    "confidence_scores",
)


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


def _extract_confidence_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in _CONFIDENCE_COLUMNS:
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce")
            if ser.notna().any():
                return ser
    return None


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
    sample_scores: Optional[np.ndarray] = None
    exact_match: Optional[np.ndarray] = None


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
        aucs = [self._evaluations[name].metrics["roc_auc"] for name in names]
        plt.bar(names, aucs, color="#3E7CB1")
        plt.ylim(0, 1)
        ylabel = (
            "ROC-AUC"
            if self.task == "binary"
            else "Exact-match ROC-AUC (confidence vs correctness)"
        )
        title = (
            "Binary ROC-AUC by Prompt/Model"
            if self.task == "binary"
            else "Multitype ROC-AUC by Prompt/Model"
        )
        plt.ylabel(ylabel)
        plt.title(title)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_roc_curves(self, filename_prefix: str = "roc_curve") -> List[Path]:
        """
        Overlay ROC curves for every model. Multitype tasks use exact-match correctness vs confidence.
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

        out_path = self.output_dir / f"{filename_prefix}.png"
        plt.figure(figsize=(7, 6))
        for name, result in self._evaluations.items():
            y_true_exact = result.exact_match
            if y_true_exact is None or result.sample_scores is None:
                raise RuntimeError(
                    "Multitype ROC curves require confidence scores and exact-match flags."
                )
            fpr, tpr, _ = roc_curve(y_true_exact, result.sample_scores)
            auc_val = result.metrics["roc_auc"]
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Exact-match vs Confidence")
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

        out_path = self.output_dir / f"{filename_prefix}.png"
        plt.figure(figsize=(7, 6))
        for name, result in self._evaluations.items():
            if result.sample_scores is None or result.exact_match is None:
                raise RuntimeError(
                    "Multitype precision-residual plots require confidence scores."
                )
            precision, recall, _ = precision_recall_curve(
                result.exact_match, result.sample_scores
            )
            residual = 1 - recall
            plt.plot(
                residual,
                precision,
                label=f"{name}",
            )
        plt.xlabel("Residual (1 - Recall)")
        plt.ylabel("Precision")
        plt.title("Precision vs Residual — Exact-match correctness")
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
            if result.sample_scores is None:
                raise RuntimeError("Confidence scores missing for multitype plot.")

            n_samples = len(result.sample_scores)
            pred_matrix = result.y_pred
            score_map = {cls: result.y_scores[cls] for cls in MULTI_LABELS}

            def _dominant_type(idx: int) -> str:
                positives = [
                    cls
                    for cls_idx, cls in enumerate(MULTI_LABELS)
                    if pred_matrix[idx, cls_idx] == 1
                ]
                if not positives:
                    return "Not IPV"
                if len(positives) == 1:
                    return positives[0]
                return max(positives, key=lambda cls: score_map[cls][idx])

            plot_df = pd.DataFrame(
                {
                    "confidence": np.clip(result.sample_scores, 0.0, 1.0),
                    "PredictedType": [_dominant_type(i) for i in range(n_samples)],
                }
            )

            colors = {
                "Emotional": "#F4D03F",
                "Physical": "#2ECC71",
                "Sexual": "#9B59B6",
                "Not IPV": "#E74C3C",
            }
            plot_df["Color"] = plot_df["PredictedType"].map(colors)
            plot_df = plot_df.sort_values("confidence", ascending=False).reset_index(drop=True)

            legend_entries = [
                f"Emotional (AUC={result.metrics['per_class_roc_auc']['Emotional']:.2f})",
                f"Physical (AUC={result.metrics['per_class_roc_auc']['Physical']:.2f})",
                f"Sexual (AUC={result.metrics['per_class_roc_auc']['Sexual']:.2f})",
                "Not IPV",
            ]
            legend_handles = [
                plt.Line2D([0], [0], color=colors[label], lw=4, label=entry)
                for label, entry in zip(
                    ["Emotional", "Physical", "Sexual", "Not IPV"], legend_entries
                )
            ]

            out_path = self.output_dir / f"{filename_prefix}_{name}.png"
            plt.figure(figsize=(13, 4))
            plt.bar(
                x=np.arange(len(plot_df)),
                height=plot_df["confidence"],
                color=plot_df["Color"],
                width=1.0,
                edgecolor="none",
            )
            plt.ylim(0, max(1.0, plot_df["confidence"].max() * 1.05))
            plt.xlabel("Sentences (sorted by confidence)")
            plt.ylabel("Confidence Score")
            plt.title(f"{name} — IPV subtype predictions")
            plt.legend(handles=legend_handles, loc="upper right", frameon=False)
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
                    sample_scores=y_score_arr,
                    exact_match=(y_true == y_pred_arr).astype(int),
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

                confidence_series = _extract_confidence_series(merged)
                if confidence_series is not None:
                    confidence = (
                        confidence_series.fillna(0.0)
                        .clip(lower=0.0, upper=1.0)
                        .to_numpy(dtype=float)
                    )
                else:
                    score_matrix = np.vstack([scores[cls] for cls in MULTI_LABELS]).T
                    confidence = np.clip(score_matrix.max(axis=1), 0.0, 1.0)

                exact_match = (y_true == y_pred_arr).all(axis=1).astype(int)

                metrics = self._compute_multilabel_metrics(
                    y_true=y_true,
                    y_pred=y_pred_arr,
                    y_scores=scores,
                    exact_match=exact_match,
                    confidence=confidence,
                )
                self._evaluations[name] = _EvalResult(
                    name=name,
                    metrics=metrics,
                    y_true=y_true,
                    y_pred=y_pred_arr,
                    y_scores=scores,
                    df_pred=merged,
                    sample_scores=confidence,
                    exact_match=exact_match,
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
        exact_match: np.ndarray,
        confidence: np.ndarray,
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
        metrics["exact_match_accuracy"] = float(exact_match.mean())
        metrics["roc_auc_macro"] = float(np.nanmean(list(per_class_auc.values())))
        metrics["roc_auc"] = _safe_auc(exact_match, confidence)
        return metrics


# ============================================================================
# Public evaluation functions for detailed metrics and visualizations
# ============================================================================


def compute_binary_metrics_detailed(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
) -> Dict[str, Union[float, int]]:
    """
    Compute detailed binary classification metrics including confusion matrix components.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy: float
        - precision: float
        - recall: float
        - f1: float
        - true_positives: int
        - false_positives: int
        - true_negatives: int
        - false_negatives: int
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics


def compute_multitype_metrics_per_subgroup(
    df: pd.DataFrame,
    y_true_cols: List[str],
    y_pred_cols: List[str],
    subgroup_col: Optional[str] = None,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Compute multitype metrics per subgroup (or overall if subgroup_col is None).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing true labels, predictions, and optionally subgroup column
    y_true_cols : List[str]
        Column names for true labels (e.g., ["Physical", "Emotional", "Sexual"])
    y_pred_cols : List[str]
        Column names for predicted labels (same order as y_true_cols)
    subgroup_col : str, optional
        Column name for subgrouping (e.g., "gender", "age_group"). If None, computes overall metrics.

    Returns
    -------
    dict
        Dictionary with keys being subgroup values (or "overall") and values being
        dictionaries of metrics per class:
        {
            "subgroup_value": {
                "class_name": {
                    "accuracy": float,
                    "precision": float,
                    "recall": float,
                    "f1": float,
                    "true_positives": int,
                    "false_positives": int,
                    "true_negatives": int,
                    "false_negatives": int,
                }
            }
        }
    """
    results = {}

    if subgroup_col is None or subgroup_col not in df.columns:
        # Compute overall metrics
        subgroup_values = ["overall"]
        df_grouped = {None: df}
    else:
        subgroup_values = df[subgroup_col].unique()
        df_grouped = {val: df[df[subgroup_col] == val] for val in subgroup_values}

    for subgroup_val, df_subset in df_grouped.items():
        key = str(subgroup_val) if subgroup_val is not None else "overall"
        results[key] = {}

        for true_col, pred_col in zip(y_true_cols, y_pred_cols):
            if true_col not in df_subset.columns or pred_col not in df_subset.columns:
                continue

            y_true = df_subset[true_col].astype(int).values
            y_pred = df_subset[pred_col].astype(int).values

            class_metrics = compute_binary_metrics_detailed(y_true, y_pred)
            results[key][true_col] = class_metrics

    return results


def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred: Union[np.ndarray, pd.Series, List[int]],
    ax: Optional[plt.Axes] = None,
    title: str = "Confusion Matrix",
    xlabel: str = "Predicted",
    ylabel: str = "Actual",
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    use_tp_fp_labels: bool = False,
) -> plt.Axes:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    class_names : List[str], optional
        Names for classes (e.g., ["NOT_IPV", "IPV"]). If None, uses indices.
    normalize : bool
        If True, normalize the confusion matrix to show percentages.
    use_tp_fp_labels : bool
        If True, use TP/FP/TN/FN labels for binary classification. Only works for 2x2 matrices.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred)
    total = len(y_true)
    cm_normalized = cm.astype("float") / total if total > 0 else cm.astype("float")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Handle TP/FP/TN/FN labels for binary classification
    if use_tp_fp_labels and cm.shape == (2, 2):
        # For binary: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
        labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
        # Map to positions: [0,0]=TN, [0,1]=FP, [1,0]=FN, [1,1]=TP
        label_map = {
            (0, 0): "True Negative",
            (0, 1): "False Positive",
            (1, 0): "False Negative",
            (1, 1): "True Positive",
        }
        
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        
        # Set labels based on position
        x_labels = [label_map.get((0, i), f"Class {i}") for i in range(2)]
        y_labels = [label_map.get((i, 0), f"Class {i}") for i in range(2)]
        
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # Show both count and percentage
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                pct = cm_normalized[i, j]
                text = f"{count}\n({pct:.2f})"
                ax.text(
                    j,
                    i,
                    text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                )
    else:
        # Standard confusion matrix
        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    text = f"{cm[i, j]:.2f}"
                else:
                    count = int(cm[i, j])
                    pct = cm_normalized[i, j]
                    text = f"{count}\n({pct:.2f})"
                ax.text(
                    j,
                    i,
                    text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")  # Rotate y-axis labels horizontally

    return ax


def plot_roc_curve_binary(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_scores: Union[np.ndarray, pd.Series, List[float]],
    ax: Optional[plt.Axes] = None,
    title: str = "ROC Curve",
    xlabel: str = "False Positive Rate",
    ylabel: str = "True Positive Rate",
    label: Optional[str] = None,
) -> Tuple[plt.Axes, float]:
    """
    Plot ROC curve for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1)
    y_scores : array-like
        Prediction scores/probabilities for positive class
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    label : str, optional
        Label for the curve (for legend)

    Returns
    -------
    Tuple[matplotlib.axes.Axes, float]
        The axes object and the AUC score
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = _safe_auc(y_true, y_scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    curve_label = label if label else f"ROC (AUC = {auc_score:.3f})"
    ax.plot(fpr, tpr, label=curve_label, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax, auc_score


def plot_precision_recall_curve_binary(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_scores: Union[np.ndarray, pd.Series, List[float]],
    ax: Optional[plt.Axes] = None,
    title: str = "Precision-Recall Curve",
    xlabel: str = "Recall",
    ylabel: str = "Precision",
    label: Optional[str] = None,
) -> Tuple[plt.Axes, float]:
    """
    Plot Precision-Recall curve for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1)
    y_scores : array-like
        Prediction scores/probabilities for positive class
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    label : str, optional
        Label for the curve (for legend)

    Returns
    -------
    Tuple[matplotlib.axes.Axes, float]
        The axes object and the average precision score
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    curve_label = label if label else f"PR (AP = {avg_precision:.3f})"
    ax.plot(recall, precision, label=curve_label, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax, avg_precision


def plot_per_class_f1_bar_chart(
    metrics_per_class: Dict[str, Dict[str, float]],
    ax: Optional[plt.Axes] = None,
    title: str = "Per-Class F1 Scores",
    xlabel: str = "Class",
    ylabel: str = "F1 Score",
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Plot a bar chart of F1 scores per class for multitype classification.

    Parameters
    ----------
    metrics_per_class : dict
        Dictionary mapping class names to metric dictionaries:
        {"class_name": {"f1": float, ...}, ...}
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    colors : List[str], optional
        Colors for bars. If None, uses default colors.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    classes = list(metrics_per_class.keys())
    f1_scores = [metrics_per_class[cls].get("f1", 0.0) for cls in classes]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

    bars = ax.bar(classes, f1_scores, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def append_binary_results_to_json(
    json_path: Union[str, Path],
    model_name: str,
    prompt_version: str,
    metrics: Dict[str, Union[float, int]],
    date_tested: Optional[str] = None,
    notes: str = "",
) -> None:
    """
    Append binary evaluation results to a JSON file.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file (will be created if doesn't exist)
    model_name : str
        Name of the model
    prompt_version : str
        Version/name of the prompt used
    metrics : dict
        Dictionary of metrics (should include accuracy, precision, recall, f1, etc.)
    date_tested : str, optional
        Date in YYYY-MM-DD format. If None, uses today's date.
    notes : str
        Optional notes about the evaluation
    """
    from datetime import datetime

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    if date_tested is None:
        date_tested = datetime.now().strftime("%Y-%m-%d")

    entry = {
        "date_tested": date_tested,
        "model": model_name,
        "prompt_version": prompt_version,
        "metrics": {
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "true_positives": int(metrics.get("true_positives", 0)),
            "false_positives": int(metrics.get("false_positives", 0)),
            "true_negatives": int(metrics.get("true_negatives", 0)),
            "false_negatives": int(metrics.get("false_negatives", 0)),
        },
        "notes": notes,
    }

    data.append(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_multitype_results_to_json(
    json_path: Union[str, Path],
    model_name: str,
    prompt_version: str,
    metrics_per_subgroup: Dict[str, Dict[str, Dict[str, Union[float, int]]]],
    date_tested: Optional[str] = None,
    notes: str = "",
) -> None:
    """
    Append multitype evaluation results to a JSON file.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file (will be created if doesn't exist)
    model_name : str
        Name of the model
    prompt_version : str
        Version/name of the prompt used
    metrics_per_subgroup : dict
        Dictionary with structure:
        {
            "subgroup_value": {
                "class_name": {
                    "accuracy": float,
                    "precision": float,
                    "recall": float,
                    "f1": float,
                    "true_positives": int,
                    "false_positives": int,
                    "true_negatives": int,
                    "false_negatives": int,
                }
            }
        }
    date_tested : str, optional
        Date in YYYY-MM-DD format. If None, uses today's date.
    notes : str
        Optional notes about the evaluation
    """
    from datetime import datetime

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    if date_tested is None:
        date_tested = datetime.now().strftime("%Y-%m-%d")

    # Convert metrics to the expected format
    metrics_per_class = {}
    for subgroup, classes in metrics_per_subgroup.items():
        if subgroup not in metrics_per_class:
            metrics_per_class[subgroup] = {}
        for class_name, class_metrics in classes.items():
            metrics_per_class[subgroup][class_name.lower()] = {
                "accuracy": float(class_metrics.get("accuracy", 0.0)),
                "precision": float(class_metrics.get("precision", 0.0)),
                "recall": float(class_metrics.get("recall", 0.0)),
                "f1": float(class_metrics.get("f1", 0.0)),
                "true_positives": int(class_metrics.get("true_positives", 0)),
                "false_positives": int(class_metrics.get("false_positives", 0)),
                "true_negatives": int(class_metrics.get("true_negatives", 0)),
                "false_negatives": int(class_metrics.get("false_negatives", 0)),
            }

    entry = {
        "date_tested": date_tested,
        "model": model_name,
        "prompt_version": prompt_version,
        "metrics_per_subgroup": metrics_per_class,
        "notes": notes,
    }

    data.append(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
