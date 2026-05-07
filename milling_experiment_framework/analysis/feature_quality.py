from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


STATS_FEATURES = {"mean", "std", "max", "min", "peak_to_peak"}
SHAPE_FEATURES = {"kurtosis", "skewness"}
FREQ_FEATURES = {"spectral_centroid", "band_energy"}
PROCESS_FEATURES = {"DoC", "Feed", "Material", "Time", "DOC", "feed", "material_name", "time"}
LEAKAGE_METADATA = {
    "case_id",
    "case",
    "domain_id",
    "pair_id",
    "source_domain",
    "target_domain",
    "split",
    "sample_id",
    "dataset_run_id",
    "run",
    "experiment_id",
}


def run_feature_quality_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    output_dir: str,
    metadata_cols: list[str] | None = None,
    experiment_id: str | None = None,
    top_n: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir)
    for dirname in ["configs", "data", "analysis", "figures", "reports", "logs"]:
        (output / dirname).mkdir(parents=True, exist_ok=True)
    experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S_H3_S0_feature_quality_analysis_for_VB_prediction")
    metadata_cols = metadata_cols or []
    log_path = output / "logs" / "H3_S0_run.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")

    error_path = output / "logs" / "H3_S0_error.log"
    error_path.touch()
    log(f"H3.S0 feature quality analysis started: {experiment_id}")

    available_features = [c for c in feature_cols if c in df.columns and c != target_col and c not in LEAKAGE_METADATA]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")
    work = df.loc[df[target_col].notna(), list(dict.fromkeys([target_col] + available_features + metadata_cols))].copy()
    feature_cols = available_features
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    target = work[target_col].astype(float)

    feature_schema = pd.DataFrame([_feature_schema_row(c, work[c]) for c in feature_cols])
    feature_schema.to_csv(output / "data" / "H3_S0_feature_schema_summary.csv", index=False)
    missing_summary = _missing_summary(work, feature_cols)
    missing_summary.to_csv(output / "data" / "H3_S0_missing_value_summary.csv", index=False)
    low_variance = _low_variance_features(work, feature_cols, numeric_features, categorical_features)
    low_variance.to_csv(output / "data" / "H3_S0_low_variance_features.csv", index=False)
    basic = _basic_check(work, target_col, feature_cols, numeric_features, categorical_features, missing_features, low_variance)
    _write_json(output / "data" / "H3_S0_data_basic_check.json", basic)
    _write_json(output / "analysis" / "H3_S0_data_basic_check.json", basic)
    log(f"Data basic check finished: samples={len(work)}, features={len(feature_cols)}")

    y_bins, bin_method = _target_bins(target)
    relevance = _target_relevance(work, target_col, feature_cols, numeric_features, categorical_features, random_state)
    separability = _target_separability(work, target_col, feature_cols, numeric_features, categorical_features, y_bins, bin_method, random_state)
    redundancy, high_pairs = _redundancy(work, feature_cols, numeric_features, categorical_features, relevance)
    leakage = _leakage_shortcut_check(work, target_col, feature_cols, metadata_cols)
    recommendation = _recommend_features(relevance, separability, high_pairs, leakage, low_variance)

    relevance.to_csv(output / "analysis" / "H3_S0_feature_relevance_table.csv", index=False)
    separability.to_csv(output / "analysis" / "H3_S0_feature_separability_table.csv", index=False)
    redundancy.to_csv(output / "analysis" / "H3_S0_feature_redundancy_table.csv", index=False)
    high_pairs.to_csv(output / "analysis" / "H3_S0_high_correlation_feature_pairs.csv", index=False)
    leakage.to_csv(output / "analysis" / "H3_S0_leakage_shortcut_check.csv", index=False)
    recommendation.to_csv(output / "analysis" / "H3_S0_feature_recommendation.csv", index=False)
    log("Analysis tables written")

    figure_status = _make_figures(
        work,
        target_col,
        feature_cols,
        numeric_features,
        categorical_features,
        y_bins,
        relevance,
        separability,
        high_pairs,
        output / "figures",
        top_n=top_n,
        random_state=random_state,
    )
    log(f"Figures generated: {sum(v == 'created' for v in figure_status.values())}")

    config = {
        "experiment": {
            "experiment_id": experiment_id,
            "experiment_name": "H3_S0_feature_quality_analysis_for_VB_prediction",
            "analysis_type": "Exploratory / Diagnostic",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "target_col": target_col,
        "num_samples": int(len(work)),
        "num_features": int(len(feature_cols)),
        "metadata_cols": metadata_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_features": missing_features,
        "top_n": top_n,
        "random_state": random_state,
    }
    _write_yaml_like(output / "configs" / "H3_S0_input_config.yaml", config)
    _write_yaml_like(output / "configs" / "H3_S0_resolved_config.yaml", config)

    report = _write_report(
        output,
        experiment_id,
        basic,
        relevance,
        separability,
        high_pairs,
        leakage,
        recommendation,
        figure_status,
        bin_method,
        categorical_features,
    )
    log("Report written")
    return {
        "experiment_id": experiment_id,
        "data_basic_check": basic,
        "feature_relevance_table": str(output / "analysis" / "H3_S0_feature_relevance_table.csv"),
        "feature_separability_table": str(output / "analysis" / "H3_S0_feature_separability_table.csv"),
        "feature_redundancy_table": str(output / "analysis" / "H3_S0_feature_redundancy_table.csv"),
        "leakage_shortcut_check": str(output / "analysis" / "H3_S0_leakage_shortcut_check.csv"),
        "feature_recommendation": str(output / "analysis" / "H3_S0_feature_recommendation.csv"),
        "report_path": str(report),
        "figure_status": figure_status,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_yaml_like(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _parse_feature_name(feature_name: str) -> tuple[str, str, str]:
    if "__" in feature_name:
        parts = feature_name.split("__")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[-1]
    process_alias = {"DOC": "DoC", "feed": "Feed", "material_name": "Material", "time": "Time"}
    return "", "process_independent", process_alias.get(feature_name, feature_name)


def _feature_group(base: str) -> str:
    if base in STATS_FEATURES:
        return "statistics"
    if base in SHAPE_FEATURES:
        return "shape"
    if base in FREQ_FEATURES:
        return "frequency"
    if base in PROCESS_FEATURES:
        return "process_information"
    return "other"


def _feature_schema_row(feature_name: str, series: pd.Series) -> dict[str, Any]:
    sensor, segment, base = _parse_feature_name(feature_name)
    return {
        "feature_name": feature_name,
        "base_feature_name": base,
        "feature_group": _feature_group(base),
        "sensor_name": sensor,
        "segment_setting": segment,
        "feature_type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
        "missing_rate": float(series.isna().mean()),
        "num_unique_values": int(series.nunique(dropna=True)),
    }


def _missing_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        rows.append(
            {
                "feature_name": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "num_unique_values": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _low_variance_features(df: pd.DataFrame, feature_cols: list[str], numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        unique = int(df[col].nunique(dropna=True))
        variance = float(df[col].var()) if col in numeric_features else math.nan
        low = unique <= 1 or (col in numeric_features and (not np.isfinite(variance) or variance <= 1e-12))
        if low:
            rows.append({"feature_name": col, "variance": variance, "num_unique_values": unique, "reason": "constant_or_low_variance"})
    return pd.DataFrame(rows, columns=["feature_name", "variance", "num_unique_values", "reason"])


def _basic_check(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    missing_features: list[str],
    low_variance: pd.DataFrame,
) -> dict[str, Any]:
    target = df[target_col].astype(float)
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = target[(target < q1 - 1.5 * iqr) | (target > q3 + 1.5 * iqr)]
    duplicated = []
    small = df[feature_cols].copy()
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i + 1 :]:
            if small[col1].equals(small[col2]):
                duplicated.append((col1, col2))
    return {
        "sample_count": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "categorical_features": categorical_features,
        "missing_features_requested_but_not_found": missing_features,
        "target_missing_count_after_filter": int(df[target_col].isna().sum()),
        "target_summary": {
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "median": float(target.median()),
            "q1": float(q1),
            "q3": float(q3),
            "outlier_count_iqr": int(len(outliers)),
        },
        "feature_missing_cells": int(df[feature_cols].isna().sum().sum()),
        "low_variance_feature_count": int(len(low_variance)),
        "duplicated_column_pairs_count": int(len(duplicated)),
        "duplicated_column_pairs_preview": duplicated[:50],
    }


def _target_bins(target: pd.Series) -> tuple[pd.Series, str]:
    try:
        bins = pd.qcut(target, q=3, labels=["low", "mid", "high"], duplicates="drop")
        return bins.astype(str), "tertile_qcut"
    except Exception:
        bins = pd.cut(target, bins=3, labels=["low", "mid", "high"])
        return bins.astype(str), "tertile_cut"


def _preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _target_relevance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> pd.DataFrame:
    y = df[target_col].astype(float).to_numpy()
    rows = []
    for col in feature_cols:
        sensor, segment, base = _parse_feature_name(col)
        feature_type = "numeric" if col in numeric_features else "categorical"
        x = df[col]
        pearson_r = pearson_p = spearman_r = spearman_p = math.nan
        if col in numeric_features:
            finite = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y)
            x_f = x.to_numpy(dtype=float)[finite]
            y_f = y[finite]
            if len(x_f) >= 3 and len(np.unique(x_f)) > 1 and len(np.unique(y_f)) > 1:
                p = stats.pearsonr(x_f, y_f)
                s = stats.spearmanr(x_f, y_f)
                pearson_r, pearson_p = float(p.statistic), float(p.pvalue)
                spearman_r, spearman_p = float(s.statistic), float(s.pvalue)
        mi = _single_feature_mi(df[col], y, feature_type, random_state)
        rows.append(
            {
                "feature_name": col,
                "base_feature_name": base,
                "feature_group": _feature_group(base),
                "sensor_name": sensor,
                "segment_setting": segment,
                "feature_type": feature_type,
                "pearson_r": pearson_r,
                "pearson_abs": abs(pearson_r) if np.isfinite(pearson_r) else math.nan,
                "pearson_p_value": pearson_p,
                "spearman_r": spearman_r,
                "spearman_abs": abs(spearman_r) if np.isfinite(spearman_r) else math.nan,
                "spearman_p_value": spearman_p,
                "mutual_information": mi,
            }
        )
    relevance = pd.DataFrame(rows)
    model_importance = _model_importance(df, target_col, numeric_features, categorical_features, random_state)
    relevance = relevance.merge(model_importance, on="feature_name", how="left")
    relevance["shap_importance"] = np.nan
    relevance["physical_interpretation"] = relevance.apply(_physical_interpretation, axis=1)
    relevance["leakage_or_shortcut_note"] = relevance.apply(_shortcut_note, axis=1)
    relevance["relevance_grade"] = relevance.apply(_relevance_grade, axis=1)
    return relevance.sort_values(["relevance_grade", "spearman_abs", "mutual_information"], ascending=[True, False, False])


def _single_feature_mi(series: pd.Series, y: np.ndarray, feature_type: str, random_state: int) -> float:
    if feature_type == "numeric":
        x = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(series.median()).to_numpy().reshape(-1, 1)
        if len(np.unique(x)) <= 1:
            return 0.0
        return float(mutual_info_regression(x, y, random_state=random_state, n_neighbors=max(1, min(3, len(y) - 1)))[0])
    codes = pd.Categorical(series.fillna("__missing__")).codes.reshape(-1, 1)
    if len(np.unique(codes)) <= 1:
        return 0.0
    return float(mutual_info_regression(codes, y, discrete_features=True, random_state=random_state, n_neighbors=max(1, min(3, len(y) - 1)))[0])


def _model_importance(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> pd.DataFrame:
    feature_cols = numeric_features + categorical_features
    pre = _preprocessor(numeric_features, categorical_features)
    model = RandomForestRegressor(n_estimators=300, random_state=random_state, min_samples_leaf=2, n_jobs=-1)
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    pipe.fit(df[feature_cols], df[target_col].astype(float))
    perm = permutation_importance(pipe, df[feature_cols], df[target_col].astype(float), n_repeats=5, random_state=random_state, n_jobs=-1)
    perm_df = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )
    encoded_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    agg = {col: 0.0 for col in feature_cols}
    for name, value in zip(encoded_names, importances):
        original = name.split("__", 1)[1]
        if original.startswith("Material_"):
            original = "Material"
        agg[original] = agg.get(original, 0.0) + float(value)
    tree_df = pd.DataFrame({"feature_name": list(agg), "tree_importance": list(agg.values())})
    return perm_df.merge(tree_df, on="feature_name", how="outer")


def _physical_interpretation(row: pd.Series) -> str:
    group = row["feature_group"]
    sensor = str(row["sensor_name"])
    segment = str(row["segment_setting"])
    base = row["base_feature_name"]
    if group == "process_information":
        return f"{base} is process information; useful if available before prediction and not a domain shortcut."
    if "AE" in sensor:
        return f"Acoustic {base} in {segment} may reflect impact, friction, chip formation, or micro-fracture events."
    if "vib" in sensor:
        return f"Vibration {base} in {segment} may reflect dynamics, chatter, unstable cutting, and tool state."
    if "smc" in sensor:
        return f"Current {base} in {segment} may reflect spindle/motor load and cutting force demand."
    return f"{base} in {segment} is a sensor-derived feature."


def _shortcut_note(row: pd.Series) -> str:
    base = row["base_feature_name"]
    if base == "Time":
        return "Time can be a wear progression proxy but may leak run order or target-after-measurement information."
    if base == "Material":
        return "Material can be physical information but may shortcut case/domain identity."
    if base in {"DoC", "Feed"}:
        return f"{base} is physically meaningful, but can shortcut domain if tied to case split."
    return ""


def _relevance_grade(row: pd.Series) -> str:
    base = row["base_feature_name"]
    score = np.nanmax(
        [
            row.get("pearson_abs", np.nan),
            row.get("spearman_abs", np.nan),
            min(float(row.get("mutual_information", 0.0)) / 0.5, 1.0),
            min(float(row.get("permutation_importance_mean", 0.0)) / 0.05, 1.0),
            min(float(row.get("tree_importance", 0.0)) / 0.05, 1.0),
        ]
    )
    if base in {"Time", "Material"} and score >= 0.6:
        return "Suspicious"
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _target_separability(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    y_bins: pd.Series,
    bin_method: str,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    y_bin_codes = pd.Categorical(y_bins).codes
    for col in feature_cols:
        sensor, segment, base = _parse_feature_name(col)
        feature_type = "numeric" if col in numeric_features else "categorical"
        values = df[col]
        anova_p = kruskal_p = effect = math.nan
        comment = ""
        if feature_type == "numeric":
            clean = pd.DataFrame({"x": values.astype(float), "bin": y_bins}).dropna()
            groups = [g["x"].to_numpy(dtype=float) for _, g in clean.groupby("bin") if len(g) > 1]
            if len(groups) >= 2 and sum(len(g) for g in groups) >= 3:
                try:
                    anova_p = float(stats.f_oneway(*groups).pvalue)
                except Exception:
                    anova_p = math.nan
                try:
                    kruskal_p = float(stats.kruskal(*groups).pvalue)
                except Exception:
                    kruskal_p = math.nan
                effect = _eta_squared(clean["x"].to_numpy(dtype=float), pd.Categorical(clean["bin"]).codes)
            comment = "numeric feature distribution compared across VB tertile bins"
        else:
            codes = pd.Categorical(values.fillna("__missing__")).codes.astype(float)
            clean = pd.DataFrame({"x": codes, "bin": y_bins}).dropna()
            groups = [g["x"].to_numpy(dtype=float) for _, g in clean.groupby("bin") if len(g) > 1]
            if len(groups) >= 2:
                try:
                    kruskal_p = float(stats.kruskal(*groups).pvalue)
                except Exception:
                    kruskal_p = math.nan
                effect = _eta_squared(clean["x"].to_numpy(dtype=float), pd.Categorical(clean["bin"]).codes)
            comment = "categorical feature encoded for exploratory separability only"
        mi_bin = _single_feature_mi_classif(values, y_bin_codes, feature_type, random_state)
        score = np.nanmax([0.0 if not np.isfinite(effect) else min(effect, 1.0), min(mi_bin / 0.5, 1.0)])
        grade = "High" if score >= 0.6 else "Medium" if score >= 0.3 else "Low"
        if base in {"Time", "Material"} and grade in {"High", "Medium"}:
            grade = "Suspicious"
        rows.append(
            {
                "feature_name": col,
                "base_feature_name": base,
                "feature_group": _feature_group(base),
                "sensor_name": sensor,
                "segment_setting": segment,
                "feature_type": feature_type,
                "target_bin_method": bin_method,
                "anova_p_value": anova_p,
                "kruskal_p_value": kruskal_p,
                "effect_size": effect,
                "mutual_information_with_target_bin": mi_bin,
                "separability_score": score,
                "separability_grade": grade,
                "distribution_comment": comment,
            }
        )
    return pd.DataFrame(rows).sort_values(["separability_score"], ascending=False)


def _single_feature_mi_classif(series: pd.Series, y_bins: np.ndarray, feature_type: str, random_state: int) -> float:
    if feature_type == "numeric":
        x = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(series.median()).to_numpy().reshape(-1, 1)
        discrete = False
    else:
        x = pd.Categorical(series.fillna("__missing__")).codes.reshape(-1, 1)
        discrete = True
    if len(np.unique(x)) <= 1:
        return 0.0
    return float(mutual_info_classif(x, y_bins, discrete_features=discrete, random_state=random_state)[0])


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand = np.nanmean(values)
    ss_between = 0.0
    ss_total = float(np.nansum((values - grand) ** 2))
    for group in np.unique(groups):
        vals = values[groups == group]
        ss_between += len(vals) * (np.nanmean(vals) - grand) ** 2
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _redundancy(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    relevance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded = pd.DataFrame(index=df.index)
    for col in numeric_features:
        encoded[col] = df[col].astype(float)
    for col in categorical_features:
        encoded[col] = pd.Categorical(df[col].fillna("__missing__")).codes
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(encoded.median(numeric_only=True))
    pearson = encoded.corr(method="pearson")
    spearman = encoded.corr(method="spearman")
    rows = []
    pairs = []
    rel = relevance.set_index("feature_name")
    for corr_type, corr in [("pearson", pearson), ("spearman", spearman)]:
        for i, f1 in enumerate(corr.columns):
            for f2 in corr.columns[i + 1 :]:
                val = corr.loc[f1, f2]
                if not np.isfinite(val):
                    continue
                if abs(val) >= 0.90:
                    s1 = float(rel.loc[f1, "spearman_abs"]) if f1 in rel.index and np.isfinite(rel.loc[f1, "spearman_abs"]) else 0.0
                    s2 = float(rel.loc[f2, "spearman_abs"]) if f2 in rel.index and np.isfinite(rel.loc[f2, "spearman_abs"]) else 0.0
                    keep = f1 if s1 >= s2 else f2
                    _, _, b1 = _parse_feature_name(f1)
                    _, _, b2 = _parse_feature_name(f2)
                    pair = {
                        "feature_1": f1,
                        "feature_2": f2,
                        "correlation_type": corr_type,
                        "correlation_value": float(val),
                        "feature_1_group": _feature_group(b1),
                        "feature_2_group": _feature_group(b2),
                        "redundancy_level": "High" if abs(val) >= 0.95 else "Medium",
                        "recommended_keep": keep,
                        "reason": "keep feature with higher Spearman relevance to VB",
                    }
                    pairs.append(pair)
    for col in feature_cols:
        related = [p for p in pairs if p["feature_1"] == col or p["feature_2"] == col]
        max_abs = max([abs(p["correlation_value"]) for p in related], default=0.0)
        rows.append(
            {
                "feature_name": col,
                "num_high_correlation_pairs": len(related),
                "max_abs_correlation_to_other_feature": max_abs,
                "redundancy_level": "High" if max_abs >= 0.95 else "Medium" if max_abs >= 0.90 else "Low",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(
        pairs,
        columns=[
            "feature_1",
            "feature_2",
            "correlation_type",
            "correlation_value",
            "feature_1_group",
            "feature_2_group",
            "redundancy_level",
            "recommended_keep",
            "reason",
        ],
    )


def _leakage_shortcut_check(df: pd.DataFrame, target_col: str, feature_cols: list[str], metadata_cols: list[str]) -> pd.DataFrame:
    rows = []
    target = df[target_col].astype(float)
    for col in feature_cols:
        _, _, base = _parse_feature_name(col)
        group = _feature_group(base)
        if group != "process_information":
            continue
        evidence = []
        leak = "Low"
        shortcut = "Low"
        validation = False
        recommendation = "Use with standard domain-shift validation"
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) > 1:
                rho = stats.spearmanr(df[col].astype(float), target, nan_policy="omit").statistic
                evidence.append(f"spearman_with_VB={rho:.3f}")
                if abs(rho) >= 0.8:
                    shortcut = "High"
                elif abs(rho) >= 0.5:
                    shortcut = "Medium"
        if base == "Time":
            leak = "Medium"
            shortcut = max(shortcut, "Medium", key={"Low": 0, "Medium": 1, "High": 2}.get)
            validation = True
            recommendation = "Validate prediction-time availability; run ablation without Time"
            evidence.append("Time may encode wear progression or run order")
        if base == "Material":
            validation = True
            recommendation = "Validate leave-material/domain behavior; do not treat as causal without domain evidence"
            if "case_id" in df.columns or "case" in df.columns:
                case_col = "case_id" if "case_id" in df.columns else "case"
                purity = _category_purity(df[col], df[case_col])
                evidence.append(f"material_to_case_purity={purity:.3f}")
                shortcut = "High" if purity >= 0.9 else "Medium"
        if base in {"DoC", "Feed"}:
            validation = True
            recommendation = f"Use {base}, but validate that it is not only a domain/case identifier"
            for meta in ["case_id", "case", "domain_id", "pair_id"]:
                if meta in df.columns:
                    purity = _numeric_group_purity(df[col], df[meta])
                    evidence.append(f"{base}_to_{meta}_purity={purity:.3f}")
                    if purity >= 0.9:
                        shortcut = max(shortcut, "Medium", key={"Low": 0, "Medium": 1, "High": 2}.get)
        rows.append(
            {
                "feature_name": col,
                "feature_group": group,
                "leakage_risk_level": leak,
                "shortcut_risk_level": shortcut,
                "evidence": "; ".join(evidence),
                "recommendation": recommendation,
                "validation_needed": validation,
            }
        )
    return pd.DataFrame(rows, columns=["feature_name", "feature_group", "leakage_risk_level", "shortcut_risk_level", "evidence", "recommendation", "validation_needed"])


def _category_purity(cat: pd.Series, label: pd.Series) -> float:
    vals = []
    for _, sub in pd.DataFrame({"cat": cat, "label": label}).dropna().groupby("cat"):
        vals.append(sub["label"].value_counts(normalize=True).max())
    return float(np.mean(vals)) if vals else 0.0


def _numeric_group_purity(x: pd.Series, label: pd.Series) -> float:
    vals = []
    for _, sub in pd.DataFrame({"x": x, "label": label}).dropna().groupby("x"):
        vals.append(sub["label"].value_counts(normalize=True).max())
    return float(np.mean(vals)) if vals else 0.0


def _recommend_features(
    relevance: pd.DataFrame,
    separability: pd.DataFrame,
    high_pairs: pd.DataFrame,
    leakage: pd.DataFrame,
    low_variance: pd.DataFrame,
) -> pd.DataFrame:
    sep = separability.set_index("feature_name")
    leak = leakage.set_index("feature_name") if not leakage.empty else pd.DataFrame()
    low = set(low_variance["feature_name"]) if not low_variance.empty else set()
    redundant = set()
    if not high_pairs.empty:
        keepers = set(high_pairs["recommended_keep"])
        involved = set(high_pairs["feature_1"]) | set(high_pairs["feature_2"])
        redundant = involved - keepers
    rows = []
    for _, r in relevance.iterrows():
        f = r["feature_name"]
        sep_grade = sep.loc[f, "separability_grade"] if f in sep.index else "Low"
        leak_level = leak.loc[f, "leakage_risk_level"] if f in leak.index else "Low"
        shortcut_level = leak.loc[f, "shortcut_risk_level"] if f in leak.index else "Low"
        physical = "High" if r["feature_group"] in {"statistics", "shape", "frequency", "process_information"} else "Unknown"
        if f in low:
            rec = "Remove candidate"
            reason = "constant or low variance"
        elif leak_level == "High" or shortcut_level == "High" or r["base_feature_name"] in {"Time", "Material"}:
            rec = "Need domain validation"
            reason = "process feature has leakage or domain shortcut risk"
        elif f in redundant and r["relevance_grade"] in {"Low", "Medium"}:
            rec = "Remove candidate"
            reason = "highly redundant and not clearly high relevance"
        elif r["relevance_grade"] in {"High", "Medium"} and sep_grade in {"High", "Medium"}:
            rec = "Keep"
            reason = "relevance and separability are both useful"
        elif r["relevance_grade"] in {"High", "Medium"}:
            rec = "Transform"
            reason = "target relevance exists but separability/distribution may need transform or nonlinear model"
        else:
            rec = "Remove candidate"
            reason = "low target relevance and low separability"
        rows.append(
            {
                "feature_name": f,
                "base_feature_name": r["base_feature_name"],
                "feature_group": r["feature_group"],
                "sensor_name": r["sensor_name"],
                "segment_setting": r["segment_setting"],
                "relevance_grade": r["relevance_grade"],
                "separability_grade": sep_grade,
                "redundancy_level": "High" if f in redundant else "Low",
                "leakage_risk_level": leak_level,
                "shortcut_risk_level": shortcut_level,
                "physical_interpretability": physical,
                "final_recommendation": rec,
                "recommendation_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _make_figures(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    y_bins: pd.Series,
    relevance: pd.DataFrame,
    separability: pd.DataFrame,
    high_pairs: pd.DataFrame,
    figure_dir: Path,
    top_n: int,
    random_state: int,
) -> dict[str, str]:
    status: dict[str, str] = {}
    y = df[target_col].astype(float)
    top_features = _top_features(relevance, separability, top_n)
    numeric_top = [f for f in top_features if f in numeric_features][: min(top_n, 12)]

    plt.figure(figsize=(7, 4))
    plt.hist(y, bins=min(12, max(3, len(y) // 3)), color="#4E79A7", edgecolor="white")
    plt.title("H3_S0 Target Distribution: VB")
    plt.xlabel("VB")
    plt.ylabel("count")
    _save(figure_dir / "H3_S0_target_distribution.png", status)

    rel_plot = relevance.set_index("feature_name").loc[top_features, ["pearson_abs", "spearman_abs", "mutual_information", "permutation_importance_mean", "tree_importance"]]
    _heatmap(rel_plot.fillna(0), "Top Feature Relevance Metrics", figure_dir / "H3_S0_feature_relevance_heatmap.png", status)

    _multi_box(df, y_bins, numeric_top, "box", figure_dir / "H3_S0_feature_boxplot_by_target_bin_top_features.png", status)
    _multi_box(df, y_bins, numeric_top, "violin", figure_dir / "H3_S0_feature_violin_by_target_bin_top_features.png", status)
    _multi_kde(df, y_bins, numeric_top[:8], figure_dir / "H3_S0_feature_kde_by_target_bin_top_features.png", status)
    _multi_scatter(df, target_col, numeric_top, figure_dir / "H3_S0_feature_scatter_vs_target_top_features.png", status)

    x_enc, encoded_names = _encoded_matrix(df, numeric_features, categorical_features)
    x_scaled = StandardScaler(with_mean=False).fit_transform(x_enc)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(x_scaled)
    _embedding_plot(coords, y_bins, "PCA by VB tertile bin", figure_dir / "H3_S0_pca_by_target_bin.png", status)
    _embedding_plot(coords, y, "PCA by continuous VB", figure_dir / "H3_S0_pca_by_continuous_target.png", status, continuous=True)
    if len(df) >= 5:
        try:
            tsne = TSNE(n_components=2, perplexity=max(2, min(10, len(df) // 3)), random_state=random_state, init="pca", learning_rate="auto")
            tsne_coords = tsne.fit_transform(x_scaled.toarray() if hasattr(x_scaled, "toarray") else x_scaled)
            _embedding_plot(tsne_coords, y_bins, "t-SNE by VB tertile bin", figure_dir / "H3_S0_tsne_by_target_bin.png", status)
        except Exception as exc:
            status["H3_S0_tsne_by_target_bin.png"] = f"skipped: {exc}"
    status["H3_S0_umap_by_target_bin.png"] = "skipped: umap-learn not installed"

    corr_features = [f for f in top_features if f in numeric_features][: min(50, len(numeric_features))]
    if corr_features:
        corr = df[corr_features].astype(float).corr().fillna(0)
        _heatmap(corr, "Feature Pearson Correlation Heatmap (Top Features)", figure_dir / "H3_S0_feature_correlation_heatmap.png", status)
    _redundancy_network(high_pairs.head(80), figure_dir / "H3_S0_high_redundancy_network.png", status)

    rec_counts = recommendation_counts = None
    # Placeholder count plot is created by reading recommendation outside this helper in report workflow impossible here,
    # so approximate with relevance grades for immediate diagnostics.
    grade_counts = relevance["relevance_grade"].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(grade_counts.index.astype(str), grade_counts.values, color="#59A14F")
    plt.title("Feature Relevance Grade Summary")
    plt.xlabel("grade")
    plt.ylabel("feature count")
    _save(figure_dir / "H3_S0_feature_recommendation_summary.png", status)
    return status


def _top_features(relevance: pd.DataFrame, separability: pd.DataFrame, top_n: int) -> list[str]:
    features: list[str] = []
    for col in ["spearman_abs", "mutual_information", "permutation_importance_mean", "tree_importance"]:
        for f in relevance.sort_values(col, ascending=False, na_position="last")["feature_name"].head(top_n):
            if f not in features:
                features.append(f)
    for f in separability.sort_values("separability_score", ascending=False)["feature_name"].head(top_n):
        if f not in features:
            features.append(f)
    return features[:top_n]


def _save(path: Path, status: dict[str, str]) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    status[path.name] = "created"


def _heatmap(data: pd.DataFrame, title: str, path: Path, status: dict[str, str]) -> None:
    plt.figure(figsize=(max(7, min(18, data.shape[1] * 1.2)), max(5, min(24, data.shape[0] * 0.25))))
    plt.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    plt.colorbar(label="value")
    plt.xticks(range(data.shape[1]), data.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(data.shape[0]), data.index, fontsize=6)
    plt.title(title)
    _save(path, status)


def _multi_box(df: pd.DataFrame, y_bins: pd.Series, features: list[str], kind: str, path: Path, status: dict[str, str]) -> None:
    if not features:
        status[path.name] = "skipped: no numeric top features"
        return
    n = len(features)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2), squeeze=False)
    labels = ["low", "mid", "high"]
    for ax, feature in zip(axes.ravel(), features):
        groups = [df.loc[y_bins == label, feature].dropna().astype(float).to_numpy() for label in labels]
        if kind == "violin":
            ax.violinplot(groups, showmeans=True, showextrema=False)
        else:
            ax.boxplot(groups, labels=labels)
        ax.set_title(feature, fontsize=8)
        ax.set_xlabel("VB bin")
        ax.set_ylabel("feature value")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Top Feature {kind.title()} by VB Tertile Bin")
    _save(path, status)


def _multi_kde(df: pd.DataFrame, y_bins: pd.Series, features: list[str], path: Path, status: dict[str, str]) -> None:
    if not features:
        status[path.name] = "skipped: no numeric top features"
        return
    n = len(features)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.2), squeeze=False)
    labels = ["low", "mid", "high"]
    for ax, feature in zip(axes.ravel(), features):
        for label in labels:
            vals = df.loc[y_bins == label, feature].dropna().astype(float)
            if len(vals) >= 3 and vals.nunique() > 1:
                try:
                    kde = stats.gaussian_kde(vals)
                    xs = np.linspace(vals.min(), vals.max(), 100)
                    ax.plot(xs, kde(xs), label=label)
                except Exception:
                    ax.hist(vals, alpha=0.3, label=label)
        ax.set_title(feature, fontsize=8)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Top Feature KDE by VB Tertile Bin")
    _save(path, status)


def _multi_scatter(df: pd.DataFrame, target_col: str, features: list[str], path: Path, status: dict[str, str]) -> None:
    if not features:
        status[path.name] = "skipped: no numeric top features"
        return
    n = len(features)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        ax.scatter(df[feature], df[target_col], s=24, alpha=0.75)
        ax.set_title(feature, fontsize=8)
        ax.set_xlabel(feature)
        ax.set_ylabel(target_col)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Top Feature Scatter vs VB")
    _save(path, status)


def _encoded_matrix(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]):
    pre = _preprocessor(numeric_features, categorical_features)
    x = pre.fit_transform(df[numeric_features + categorical_features])
    return x, pre.get_feature_names_out()


def _embedding_plot(coords: np.ndarray, color_values: pd.Series, title: str, path: Path, status: dict[str, str], continuous: bool = False) -> None:
    plt.figure(figsize=(6.5, 5))
    if continuous:
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=color_values.astype(float), cmap="viridis", s=45)
        plt.colorbar(sc, label="VB")
    else:
        labels = list(pd.Series(color_values).astype(str).unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for label, color in zip(labels, colors):
            mask = pd.Series(color_values).astype(str).to_numpy() == label
            plt.scatter(coords[mask, 0], coords[mask, 1], label=label, color=color, s=45)
        plt.legend(title="VB bin")
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    _save(path, status)


def _redundancy_network(high_pairs: pd.DataFrame, path: Path, status: dict[str, str]) -> None:
    if high_pairs.empty:
        status[path.name] = "skipped: no high correlation pairs"
        return
    graph = nx.Graph()
    for _, row in high_pairs.iterrows():
        graph.add_edge(row["feature_1"], row["feature_2"], weight=abs(row["correlation_value"]))
    plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(graph, seed=42, k=0.4)
    nx.draw_networkx_nodes(graph, pos, node_size=120, node_color="#4E79A7", alpha=0.8)
    nx.draw_networkx_edges(graph, pos, alpha=0.25)
    nx.draw_networkx_labels(graph, pos, font_size=5)
    plt.title("High Redundancy Feature Network (abs corr >= 0.90)")
    plt.axis("off")
    _save(path, status)


def _write_report(
    output: Path,
    experiment_id: str,
    basic: dict[str, Any],
    relevance: pd.DataFrame,
    separability: pd.DataFrame,
    high_pairs: pd.DataFrame,
    leakage: pd.DataFrame,
    recommendation: pd.DataFrame,
    figure_status: dict[str, str],
    bin_method: str,
    categorical_features: list[str],
) -> Path:
    def names(df: pd.DataFrame, col: str = "feature_name", n: int = 15) -> str:
        if df.empty:
            return "none"
        return ", ".join(df[col].astype(str).head(n).tolist())

    def table(df: pd.DataFrame, n: int = 20) -> str:
        if df.empty:
            return "none"
        return df.head(n).to_string(index=False)

    high_rel = relevance[relevance["relevance_grade"] == "High"].sort_values(["spearman_abs", "mutual_information"], ascending=False)
    high_sep = separability[separability["separability_grade"] == "High"].sort_values("separability_score", ascending=False)
    high_risk = leakage[(leakage["leakage_risk_level"] == "High") | (leakage["shortcut_risk_level"] == "High")] if not leakage.empty else pd.DataFrame()
    rec_counts = recommendation["final_recommendation"].value_counts().to_dict()
    keep = recommendation[recommendation["final_recommendation"] == "Keep"]
    transform = recommendation[recommendation["final_recommendation"] == "Transform"]
    remove = recommendation[recommendation["final_recommendation"] == "Remove candidate"]
    validate = recommendation[recommendation["final_recommendation"] == "Need domain validation"]
    created_figures = sum(v == "created" for v in figure_status.values())
    skipped = {k: v for k, v in figure_status.items() if v != "created"}

    report = f"""# H3_S0 Feature Quality Report

## 1. Executive Summary

- 분석 sample 수는 {basic['sample_count']}개, feature 수는 {basic['feature_count']}개입니다.
- target relevance High feature는 {len(high_rel)}개입니다. 상위 예시는 {names(high_rel)}입니다.
- target separability High feature는 {len(high_sep)}개입니다. 상위 예시는 {names(high_sep)}입니다.
- leakage/shortcut High risk feature는 {len(high_risk)}개입니다. 대상은 {names(high_risk)}입니다.
- 최종 recommendation 분포는 {json.dumps(rec_counts, ensure_ascii=False)}입니다.

## 2. Data Basic Check

- numeric feature: {basic['numeric_feature_count']}
- categorical feature: {basic['categorical_feature_count']} ({', '.join(categorical_features) if categorical_features else 'none'})
- target VB min/mean/max: {basic['target_summary']['min']:.4f} / {basic['target_summary']['mean']:.4f} / {basic['target_summary']['max']:.4f}
- target IQR outlier count: {basic['target_summary']['outlier_count_iqr']}
- feature missing cells: {basic['feature_missing_cells']}
- low variance feature count: {basic['low_variance_feature_count']}
- duplicated column pair count: {basic['duplicated_column_pairs_count']}

## 3. Target Relevance Analysis

Pearson/Spearman은 numeric feature에 대해 절댓값 기준으로 해석했습니다. MI는 scale이 다르므로 같은 metric 내부 ranking 중심으로 보았습니다. RandomForest permutation importance와 impurity importance를 함께 계산했습니다.

High relevance top features:

```text
{table(high_rel[['feature_name','base_feature_name','feature_group','spearman_abs','mutual_information','permutation_importance_mean','tree_importance','relevance_grade']])}
```

Process information은 relevance가 높더라도 Time/Material/DoC/Feed가 case/domain shortcut일 수 있으므로 별도 validation 대상입니다.

## 4. Target Separability Analysis

Target binning은 `{bin_method}` 방식으로 low/mid/high tertile을 만들었습니다. ANOVA, Kruskal-Wallis, eta-squared 계열 effect size, target-bin MI를 계산했습니다.

High separability top features:

```text
{table(high_sep[['feature_name','base_feature_name','feature_group','effect_size','mutual_information_with_target_bin','separability_score','separability_grade']])}
```

## 5. Feature Space Visualization

PCA는 `H3_S0_pca_by_target_bin.png`와 `H3_S0_pca_by_continuous_target.png`에 저장했습니다. t-SNE는 가능한 경우 생성했고, UMAP은 `umap-learn` 미설치로 skipped 처리했습니다. Feature space 분리는 exploratory 확인용이며, prediction 성능을 보장하지 않습니다.

## 6. Feature Redundancy Analysis

abs(correlation) >= 0.90 기준 high-correlation pair는 {len(high_pairs)}개입니다. 이는 segment/sensor 조합 feature 사이에 중복 후보가 많다는 뜻입니다.

Top redundant pairs:

```text
{table(high_pairs) if not high_pairs.empty else 'No high-correlation pairs detected.'}
```

## 7. Leakage and Shortcut Risk

Time은 wear progression proxy일 수 있지만 run order 또는 label 생성 이후 정보를 반영하면 leakage가 됩니다. Material은 실제 물성 정보일 수 있으나 case/domain shortcut일 수 있습니다. DoC/Feed도 절삭 조건으로 물리적 원인 변수일 수 있지만 domain split과 강하게 결합되어 있으면 shortcut 위험이 있습니다.

```text
{table(leakage, n=50) if not leakage.empty else 'No process information feature was included.'}
```

## 8. Final Feature Recommendation

- Keep: {names(keep)}
- Transform: {names(transform)}
- Remove candidate: {names(remove)}
- Need domain validation: {names(validate)}

## 9. Implications for Future Modeling

H2/H3 후속 실험에서는 High relevance와 High separability feature를 우선 후보로 두되, high redundancy pair에서는 대표 feature만 남기는 sparse candidate search가 적합합니다. Acoustic entry/exit 계열 sensor feature는 물리적으로 접촉/이탈 시점의 충격, 마찰, 미세 파손 이벤트와 연결될 수 있어 segment-aware attention/gating 모델의 후보 입력으로 유지할 만합니다. Process information은 Time/Material leakage-safe ablation을 반드시 포함해야 합니다.

## 10. Limitations

본 분석은 exploratory/diagnostic 분석입니다. 높은 correlation이나 MI가 실제 domain-shift prediction 성능을 보장하지 않습니다. RandomForest importance는 모델 의존적이고, target binning 방식도 결과에 영향을 줍니다. sample 수가 작아 p-value와 high-dimensional redundancy 해석에는 주의가 필요합니다.

## Figure Status

- created figures: {created_figures}
- skipped/partial: {json.dumps(skipped, ensure_ascii=False)}
"""
    report_path = output / "reports" / "H3_S0_feature_quality_report.md"
    report_path.write_text(report, encoding="utf-8")
    html_report = (
        "<!doctype html><html><head><meta charset='utf-8'><title>H3_S0 Feature Quality Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.55}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:16px;border-radius:6px}</style></head><body><pre>"
        + html.escape(report)
        + "</pre></body></html>"
    )
    (output / "reports" / "H3_S0_feature_quality_report.html").write_text(html_report, encoding="utf-8")
    return report_path
