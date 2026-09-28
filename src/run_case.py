"""Deterministic end-to-end implementation for Customer Segmentation Case 05."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from nbclient import NotebookClient
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .segmentation_utils import (
    FUTURE_OUTCOMES,
    FrozenPreprocessor,
    apply_rfm,
    confidence_interval_mean,
    fit_rfm_cutoffs,
    internal_metrics,
    minimum_cluster_share,
    semantic_cluster_map,
)


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "case_05_customer_segmentation_raw.csv"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
MODELS = ROOT / "outputs" / "models"
NOTEBOOK = ROOT / "notebooks" / "01_end_to_end_case.ipynb"

DEV_QUARTER = "2026Q1"
VALIDATION_QUARTER = "2026Q2"
HOLDOUT_QUARTER = "2026Q3"

FEATURES = [
    "customer_tenure_months",
    "merchant_flag",
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "avg_ticket_90d",
    "txn_active_days_90d",
    "avg_balance_90d",
    "balance_volatility_90d",
    "income_proxy",
    "salary_inflow_flag",
    "has_credit_card",
    "credit_limit",
    "credit_utilization",
    "card_spend_90d",
    "pix_count_90d",
    "pix_value_90d",
    "merchant_txn_count_90d",
    "merchant_gmv_90d",
    "has_loan",
    "loan_balance",
    "has_investment",
    "investment_balance",
    "has_insurance",
    "has_pos",
    "number_of_products",
    "app_sessions_90d",
    "app_active_days_90d",
    "days_since_last_login",
    "push_open_rate_90d",
    "email_open_rate_90d",
    "campaigns_received_90d",
    "campaign_click_rate_90d",
    "revenue_90d",
    "margin_90d",
]

LOG_FEATURES = [
    "customer_tenure_months",
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "avg_ticket_90d",
    "txn_active_days_90d",
    "avg_balance_90d",
    "balance_volatility_90d",
    "income_proxy",
    "credit_limit",
    "card_spend_90d",
    "pix_count_90d",
    "pix_value_90d",
    "merchant_txn_count_90d",
    "merchant_gmv_90d",
    "loan_balance",
    "investment_balance",
    "app_sessions_90d",
    "app_active_days_90d",
    "days_since_last_login",
    "campaigns_received_90d",
    "revenue_90d",
    "margin_90d",
]

PROFILE_FEATURES = [
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "avg_balance_90d",
    "credit_utilization",
    "card_spend_90d",
    "pix_count_90d",
    "merchant_txn_count_90d",
    "merchant_gmv_90d",
    "investment_balance",
    "number_of_products",
    "app_sessions_90d",
    "campaign_click_rate_90d",
    "revenue_90d",
    "margin_90d",
]


def _ensure_directories() -> None:
    for path in [FIGURES, TABLES, METRICS, MODELS, NOTEBOOK.parent]:
        path.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def load_and_audit() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW_PATH, low_memory=False)
    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_customers": int(df["customer_id"].nunique()),
        "quarters": sorted(df["snapshot_quarter"].unique().tolist()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_quarter": int(df.duplicated(["customer_id", "snapshot_quarter"]).sum()),
        "customers_with_three_snapshots": int((df.groupby("customer_id").size() == 3).sum()),
        "future_outcomes_excluded_from_clustering": FUTURE_OUTCOMES,
    }
    (METRICS / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = (
        df.groupby("snapshot_quarter", observed=True)
        .apply(lambda x: x.isna().mean(), include_groups=False)
        .T.reset_index(names="feature")
    )
    missing.to_csv(TABLES / "missing_rate_by_quarter.csv", index=False)

    q = []
    for quarter, part in df.groupby("snapshot_quarter", observed=True):
        q.append(
            {
                "snapshot_quarter": quarter,
                "rows": len(part),
                "customers": part["customer_id"].nunique(),
                "credit_utilization_missing": part["credit_utilization"].isna().mean(),
                "push_missing": part["push_open_rate_90d"].isna().mean(),
                "email_missing": part["email_open_rate_90d"].isna().mean(),
                "login_sentinel_9999": part["days_since_last_login"].eq(9999).mean(),
            }
        )
    pd.DataFrame(q).to_csv(TABLES / "data_quality_by_quarter.csv", index=False)
    return df, audit


def plot_data_quality(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    miss = df.groupby("snapshot_quarter", observed=True)[
        ["credit_utilization", "push_open_rate_90d", "email_open_rate_90d"]
    ].apply(lambda x: x.isna().mean())
    miss.plot(kind="bar", ax=axes[0], color=["#24476b", "#2f7f7b", "#d68b45"])
    axes[0].set_title("Missing por trimestre: estrutural e crescente")
    axes[0].set_xlabel("Snapshot")
    axes[0].set_ylabel("Proporção missing")
    axes[0].legend(["Utilização de crédito", "Push", "E-mail"], frameon=False)
    sample = df.loc[df["snapshot_quarter"].eq(DEV_QUARTER), "monetary_90d"].sample(40_000, random_state=SEED)
    sns.histplot(np.log1p(sample.clip(lower=0)), bins=55, ax=axes[1], color="#24476b")
    axes[1].set_title("Cauda longa tratada com log1p (2026Q1)")
    axes[1].set_xlabel("log(1 + monetary_90d)")
    axes[1].set_ylabel("Clientes")
    fig.suptitle("Qualidade e geometria da base", fontsize=14, fontweight="bold")
    _savefig(FIGURES / "01_data_quality_and_scale.png")

    corr_cols = [
        "frequency_90d", "monetary_90d", "avg_ticket_90d", "avg_balance_90d",
        "credit_limit", "card_spend_90d", "pix_count_90d", "pix_value_90d",
        "app_sessions_90d", "revenue_90d", "margin_90d", "total_financial_assets",
        "transactional_intensity", "engagement_index_raw",
    ]
    corr = df.loc[df["snapshot_quarter"].eq(DEV_QUARTER), corr_cols].sample(60_000, random_state=SEED).corr()
    corr.to_csv(TABLES / "development_feature_correlation.csv")
    plt.figure(figsize=(10.5, 8.5))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Correlação revela janelas e features derivadas redundantes — 2026Q1")
    _savefig(FIGURES / "02_feature_correlation.png")


def benchmark_candidates(dev: pd.DataFrame, x_robust: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(SEED)

    for k in [4, 5, 6, 7]:
        model = MiniBatchKMeans(
            n_clusters=k, batch_size=8_192, n_init=10, max_iter=200, random_state=SEED
        )
        labels = model.fit_predict(x_robust)
        row = {
            "candidate": f"KMeans_Robust_k{k}", "algorithm": "MiniBatchKMeans",
            "preprocessing": "log+winsor+RobustScaler", "n_clusters": k,
            "fit_rows": len(dev), "scope": "full development", "status": "evaluated",
            "min_cluster_share": minimum_cluster_share(labels),
        }
        row.update(internal_metrics(x_robust, labels))
        rows.append(row)

    standard = FrozenPreprocessor(FEATURES, LOG_FEATURES, scaler_kind="standard")
    x_standard = standard.fit_transform(dev)
    model = MiniBatchKMeans(n_clusters=6, batch_size=8_192, n_init=10, max_iter=200, random_state=SEED)
    labels = model.fit_predict(x_standard)
    row = {
        "candidate": "KMeans_Standard_k6", "algorithm": "MiniBatchKMeans",
        "preprocessing": "log+winsor+StandardScaler", "n_clusters": 6,
        "fit_rows": len(dev), "scope": "full development", "status": "evaluated",
        "min_cluster_share": minimum_cluster_share(labels),
    }
    row.update(internal_metrics(x_standard, labels))
    rows.append(row)

    pca = PCA(n_components=0.90, svd_solver="full", random_state=SEED)
    x_pca = pca.fit_transform(x_robust)
    model = MiniBatchKMeans(n_clusters=6, batch_size=8_192, n_init=10, max_iter=200, random_state=SEED)
    labels = model.fit_predict(x_pca)
    row = {
        "candidate": "KMeans_Robust_PCA90_k6", "algorithm": "MiniBatchKMeans",
        "preprocessing": f"robust+PCA({x_pca.shape[1]} comps)", "n_clusters": 6,
        "fit_rows": len(dev), "scope": "full development", "status": "evaluated",
        "min_cluster_share": minimum_cluster_share(labels),
    }
    row.update(internal_metrics(x_pca, labels))
    rows.append(row)

    fit_idx = rng.choice(len(dev), size=60_000, replace=False)
    for k in [5, 6]:
        model = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-5, max_iter=150, random_state=SEED)
        model.fit(x_robust[fit_idx])
        eval_idx = rng.choice(len(dev), size=25_000, replace=False)
        labels = model.predict(x_robust[eval_idx])
        row = {
            "candidate": f"GMM_diag_Robust_k{k}", "algorithm": "GaussianMixture",
            "preprocessing": "log+winsor+RobustScaler", "n_clusters": k,
            "fit_rows": len(fit_idx), "scope": "representative sample", "status": "evaluated",
            "min_cluster_share": minimum_cluster_share(labels),
        }
        row.update(internal_metrics(x_robust[eval_idx], labels))
        rows.append(row)

    hier_idx = rng.choice(len(dev), size=8_000, replace=False)
    hier = AgglomerativeClustering(n_clusters=6, linkage="ward")
    labels = hier.fit_predict(x_robust[hier_idx])
    row = {
        "candidate": "Hierarchical_Ward_k6", "algorithm": "AgglomerativeClustering",
        "preprocessing": "log+winsor+RobustScaler", "n_clusters": 6,
        "fit_rows": len(hier_idx), "scope": "diagnostic sample; no native scoring", "status": "evaluated",
        "min_cluster_share": minimum_cluster_share(labels),
    }
    row.update(internal_metrics(x_robust[hier_idx], labels))
    rows.append(row)

    try:
        import hdbscan  # type: ignore

        h_idx = rng.choice(len(dev), size=30_000, replace=False)
        h_model = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=40, prediction_data=True)
        labels = h_model.fit_predict(x_robust[h_idx])
        valid = labels >= 0
        row = {
            "candidate": "HDBSCAN_Robust", "algorithm": "HDBSCAN",
            "preprocessing": "log+winsor+RobustScaler", "n_clusters": int(np.unique(labels[valid]).size),
            "fit_rows": len(h_idx), "scope": "diagnostic sample", "status": "evaluated",
            "noise_share": float((~valid).mean()), "min_cluster_share": minimum_cluster_share(labels[valid]),
        }
        row.update(internal_metrics(x_robust[h_idx], labels))
        rows.append(row)
    except ImportError:
        rows.append(
            {
                "candidate": "HDBSCAN_Robust", "algorithm": "HDBSCAN",
                "preprocessing": "log+winsor+RobustScaler", "n_clusters": np.nan,
                "fit_rows": 0, "scope": "optional dependency", "status": "skipped: package unavailable",
                "noise_share": np.nan, "min_cluster_share": np.nan,
                "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_benchmark(benchmark: pd.DataFrame) -> None:
    evaluated = benchmark.loc[benchmark["status"].eq("evaluated")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    order = evaluated.sort_values("silhouette", ascending=True)
    sns.barplot(data=order, x="silhouette", y="candidate", ax=axes[0], color="#2f7f7b")
    axes[0].set_title("Separação interna (maior é melhor)")
    axes[0].set_xlabel("Silhouette em amostra fixa")
    axes[0].set_ylabel("")
    order = evaluated.sort_values("davies_bouldin", ascending=False)
    sns.barplot(data=order, x="davies_bouldin", y="candidate", ax=axes[1], color="#d68b45")
    axes[1].set_title("Compacidade e separação (menor é melhor)")
    axes[1].set_xlabel("Davies–Bouldin")
    axes[1].set_ylabel("")
    fig.suptitle("Benchmark não supervisionado — 2026Q1", fontsize=14, fontweight="bold")
    _savefig(FIGURES / "03_internal_benchmark.png")


def freeze_champion(dev: pd.DataFrame, x_dev: np.ndarray, benchmark: pd.DataFrame):
    champion_name = "KMeans_Robust_k6"
    model = MiniBatchKMeans(n_clusters=6, batch_size=8_192, n_init=20, max_iter=300, random_state=SEED)
    cluster_id = model.fit_predict(x_dev)
    dev = dev.copy()
    dev["cluster_id"] = cluster_id
    raw_profile = dev.groupby("cluster_id", observed=True)[PROFILE_FEATURES].mean().reset_index()
    mapping = semantic_cluster_map(raw_profile)
    dev["segment_name"] = dev["cluster_id"].map(mapping)

    selected = benchmark.loc[benchmark["candidate"].eq(champion_name)].iloc[0].to_dict()
    freeze_record = {
        "freeze_timestamp_stage": "before Q2/Q3 future-outcome inspection",
        "development_quarter": DEV_QUARTER,
        "validation_quarter": VALIDATION_QUARTER,
        "holdout_quarter": HOLDOUT_QUARTER,
        "champion": champion_name,
        "n_clusters": 6,
        "feature_count_before_missing_indicators": len(FEATURES),
        "future_outcomes_excluded": FUTURE_OUTCOMES,
        "selection_evidence": {
            "silhouette": selected["silhouette"],
            "davies_bouldin": selected["davies_bouldin"],
            "minimum_cluster_share": selected["min_cluster_share"],
        },
        "rationale": (
            "Six robust K-Means segments balance separation, minimum operational scale, "
            "native out-of-sample scoring, explainable centroids, and stable business profiles. "
            "PCA was not selected because its modest geometric benefit did not offset lost interpretability; "
            "hierarchical and density approaches remained diagnostic samples."
        ),
        "cluster_name_map": {str(k): v for k, v in mapping.items()},
    }
    (METRICS / "taxonomy_freeze_record.json").write_text(
        json.dumps(freeze_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model, dev, mapping, freeze_record


def build_profiles(dev_labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = dev_labeled.groupby(["cluster_id", "segment_name"], observed=True)[PROFILE_FEATURES].mean().reset_index()
    size = dev_labeled.groupby(["cluster_id", "segment_name"], observed=True).size().rename("customers").reset_index()
    profile = profile.merge(size, on=["cluster_id", "segment_name"])
    profile["share"] = profile["customers"] / profile["customers"].sum()
    profile.to_csv(TABLES / "frozen_segment_profiles_q1.csv", index=False)

    heat = profile.set_index("segment_name")[PROFILE_FEATURES]
    heat_z = (heat - dev_labeled[PROFILE_FEATURES].mean()) / dev_labeled[PROFILE_FEATURES].std(ddof=0).replace(0, 1)
    heat_z.to_csv(TABLES / "frozen_segment_profile_zscores_q1.csv")
    plt.figure(figsize=(13.5, 6.5))
    sns.heatmap(heat_z, cmap="vlag", center=0, vmin=-2.5, vmax=2.5, linewidths=0.3, cbar_kws={"label": "Desvio-padrão vs. base"})
    plt.title("Perfis que dão significado operacional à taxonomia congelada — 2026Q1")
    plt.xlabel("Característica média")
    plt.ylabel("Segmento")
    _savefig(FIGURES / "04_frozen_segment_profiles.png")
    return profile, heat_z


def apply_frozen_taxonomy(
    df: pd.DataFrame,
    preprocessor: FrozenPreprocessor,
    model: MiniBatchKMeans,
    mapping: dict[int, str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labeled_parts = []
    transformed: dict[str, np.ndarray] = {}
    for quarter in [DEV_QUARTER, VALIDATION_QUARTER, HOLDOUT_QUARTER]:
        part = df.loc[df["snapshot_quarter"].eq(quarter)].copy()
        x = preprocessor.transform(part)
        part["cluster_id"] = model.predict(x)
        part["segment_name"] = part["cluster_id"].map(mapping)
        transformed[quarter] = x
        labeled_parts.append(part)
    return pd.concat(labeled_parts, ignore_index=True), transformed


def temporal_stability(labeled: pd.DataFrame, transformed: dict[str, np.ndarray], model) -> tuple[pd.DataFrame, pd.DataFrame]:
    shares = (
        labeled.groupby(["snapshot_quarter", "segment_name"], observed=True).size()
        .rename("customers").reset_index()
    )
    shares["share"] = shares["customers"] / shares.groupby("snapshot_quarter")["customers"].transform("sum")
    shares.to_csv(TABLES / "segment_share_by_quarter.csv", index=False)

    wide = labeled.pivot(index="customer_id", columns="snapshot_quarter", values="segment_name")
    stability_rows = []
    for start, end in [(DEV_QUARTER, VALIDATION_QUARTER), (VALIDATION_QUARTER, HOLDOUT_QUARTER), (DEV_QUARTER, HOLDOUT_QUARTER)]:
        stability_rows.append(
            {
                "from_quarter": start,
                "to_quarter": end,
                "label_retention": float((wide[start] == wide[end]).mean()),
                "adjusted_rand_index": float(adjusted_rand_score(wide[start], wide[end])),
                "normalized_mutual_information": float(normalized_mutual_info_score(wide[start], wide[end])),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(METRICS / "temporal_stability_metrics.csv", index=False)

    migration = pd.crosstab(wide[VALIDATION_QUARTER], wide[HOLDOUT_QUARTER], normalize="index")
    migration.to_csv(TABLES / "migration_matrix_q2_to_q3.csv")

    drift_rows = []
    for quarter, x in transformed.items():
        part = labeled.loc[labeled["snapshot_quarter"].eq(quarter)]
        for cid in sorted(part["cluster_id"].unique()):
            mask = part["cluster_id"].to_numpy() == cid
            observed_centroid = x[mask].mean(axis=0)
            drift_rows.append(
                {
                    "snapshot_quarter": quarter,
                    "cluster_id": int(cid),
                    "segment_name": part.loc[part["cluster_id"].eq(cid), "segment_name"].iloc[0],
                    "centroid_distance_from_q1_model": float(np.linalg.norm(observed_centroid - model.cluster_centers_[cid])),
                }
            )
    pd.DataFrame(drift_rows).to_csv(METRICS / "centroid_drift_by_quarter.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    sns.lineplot(data=shares, x="snapshot_quarter", y="share", hue="segment_name", marker="o", ax=axes[0])
    axes[0].set_title("Participação permanece operacionalmente estável")
    axes[0].set_xlabel("Snapshot")
    axes[0].set_ylabel("Participação")
    axes[0].legend(title="Segmento", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.barplot(data=stability, x=stability["from_quarter"] + " → " + stability["to_quarter"], y="label_retention", ax=axes[1], color="#2f7f7b")
    axes[1].set_title("Retenção de rótulo entre snapshots")
    axes[1].set_xlabel("Janela")
    axes[1].set_ylabel("Clientes no mesmo segmento")
    axes[1].set_ylim(0, 1)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1%%", padding=3)
    fig.suptitle("Estabilidade da taxonomia congelada", fontsize=14, fontweight="bold")
    _savefig(FIGURES / "05_temporal_stability.png")

    plt.figure(figsize=(9, 7))
    sns.heatmap(migration, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1, cbar=False)
    plt.title("Migração de segmentos: 2026Q2 → 2026Q3")
    plt.xlabel("Segmento em 2026Q3")
    plt.ylabel("Segmento em 2026Q2")
    _savefig(FIGURES / "06_migration_matrix.png")
    return shares, stability


def rfm_baseline(labeled: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dev = labeled.loc[labeled["snapshot_quarter"].eq(DEV_QUARTER)]
    cutoffs = fit_rfm_cutoffs(dev)
    work = labeled[["customer_id", "snapshot_quarter", "recency_days", "frequency_90d", "monetary_90d"]].copy()
    work["rfm_segment"] = apply_rfm(work, cutoffs)
    shares = work.groupby(["snapshot_quarter", "rfm_segment"], observed=True).size().rename("customers").reset_index()
    shares["share"] = shares["customers"] / shares.groupby("snapshot_quarter")["customers"].transform("sum")
    shares.to_csv(TABLES / "rfm_baseline_share_by_quarter.csv", index=False)
    wide = work.pivot(index="customer_id", columns="snapshot_quarter", values="rfm_segment")
    record = {
        "cutoffs": cutoffs,
        "q1_to_q2_label_retention": float((wide[DEV_QUARTER] == wide[VALIDATION_QUARTER]).mean()),
        "q2_to_q3_label_retention": float((wide[VALIDATION_QUARTER] == wide[HOLDOUT_QUARTER]).mean()),
    }
    (METRICS / "rfm_baseline_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return work, record


def external_validation(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (quarter, segment), part in labeled.groupby(["snapshot_quarter", "segment_name"], observed=True):
        row = {"snapshot_quarter": quarter, "segment_name": segment, "customers": len(part)}
        for outcome in FUTURE_OUTCOMES:
            low, high = confidence_interval_mean(part[outcome])
            row[outcome] = float(part[outcome].mean())
            row[f"{outcome}_ci_low"] = low
            row[f"{outcome}_ci_high"] = high
        rows.append(row)
    validation = pd.DataFrame(rows)
    validation.to_csv(TABLES / "future_outcome_validation_by_segment.csv", index=False)

    tests = []
    for quarter, part in labeled.groupby("snapshot_quarter", observed=True):
        groups = [g for _, g in part.groupby("segment_name", observed=True)]
        for outcome in ["revenue_next_90d", "margin_next_90d"]:
            stat, pvalue = stats.kruskal(*[g[outcome].to_numpy() for g in groups])
            tests.append({"snapshot_quarter": quarter, "outcome": outcome, "test": "Kruskal-Wallis", "statistic": stat, "p_value": pvalue})
        for outcome in ["churn_next_90d", "product_adoption_next_90d"]:
            contingency = pd.crosstab(part["segment_name"], part[outcome])
            stat, pvalue, _, _ = stats.chi2_contingency(contingency)
            tests.append({"snapshot_quarter": quarter, "outcome": outcome, "test": "Chi-square", "statistic": stat, "p_value": pvalue})
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(METRICS / "future_outcome_differentiation_tests.csv", index=False)

    q3 = validation.loc[validation["snapshot_quarter"].eq(HOLDOUT_QUARTER)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    panels = [
        ("revenue_next_90d", "Receita futura média", "R$"),
        ("margin_next_90d", "Margem futura média", "R$"),
        ("churn_next_90d", "Churn futuro", "Taxa"),
        ("product_adoption_next_90d", "Adoção futura de produto", "Taxa"),
    ]
    for ax, (col, title, ylabel) in zip(axes.flat, panels):
        order = q3.sort_values(col, ascending=False)
        sns.barplot(data=order, x=col, y="segment_name", ax=ax, color="#24476b")
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("")
    fig.suptitle("Validação externa aberta somente após o congelamento — holdout 2026Q3", fontsize=14, fontweight="bold")
    _savefig(FIGURES / "07_future_outcome_validation.png")
    return validation, tests_df


def build_actionability(profile: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    playbook = {
        "Comerciantes Ativos": ("Relacionamento e adquirência", "Ofertas de POS, capital de giro e serviços transacionais", "Margem e reclamações"),
        "Patrimônio & Valor": ("Proteção e profundidade", "Assessoria, investimentos e benefícios de relacionamento", "Evasão de saldo e churn"),
        "Crédito Intensivo": ("Crédito responsável", "Limite e parcelamento com elegibilidade e risco", "Inadimplência e sobre-endividamento"),
        "Digitais Engajados": ("Cross-sell digital", "Next-best-product em canais próprios", "Fadiga e opt-out"),
        "Baixo Engajamento": ("Retenção diagnóstica", "Testar reativação de baixo custo antes de incentivo", "ROI incremental e sleeping dogs"),
        "Relacionamento Essencial": ("Educação e ativação", "Jornadas simples para aumentar uso e profundidade", "Custo de contato"),
    }
    q3 = validation.loc[validation["snapshot_quarter"].eq(HOLDOUT_QUARTER)].set_index("segment_name")
    rows = []
    for _, row in profile.iterrows():
        segment = row["segment_name"]
        objective, action, guardrail = playbook[segment]
        rows.append(
            {
                "segment_name": segment,
                "q1_share": row["share"],
                "business_objective": objective,
                "action_hypothesis": action,
                "primary_guardrail": guardrail,
                "q3_revenue_next_90d": q3.loc[segment, "revenue_next_90d"],
                "q3_churn_next_90d": q3.loc[segment, "churn_next_90d"],
                "q3_product_adoption_next_90d": q3.loc[segment, "product_adoption_next_90d"],
                "causal_status": "Hypothesis requiring controlled experiment",
            }
        )
    actions = pd.DataFrame(rows).sort_values("q1_share", ascending=False)
    actions.to_csv(TABLES / "segment_actionability_playbook.csv", index=False)
    return actions


def save_assignments(labeled: pd.DataFrame) -> None:
    labeled[["customer_id", "snapshot_quarter", "cluster_id", "segment_name"]].to_csv(
        TABLES / "customer_segment_assignments.csv.gz", index=False, compression="gzip"
    )


def create_notebook(summary: dict) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Case 05 — Customer Segmentation\n\n"
            "Segmentação acionável para CRM, cross-sell, retenção e relacionamento. "
            "A taxonomia é formada sem outcomes futuros, congelada em 2026Q1 e aplicada sem refit em 2026Q2–Q3."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Decisão, unidade e fronteira temporal\n\n"
            "**Unidade:** `customer_id × snapshot_quarter`.  "
            "**Decision timestamp:** fechamento de cada trimestre.  "
            "**Desenvolvimento:** 2026Q1. **Validação temporal:** 2026Q2. **Holdout:** 2026Q3.  "
            "**Regra de leakage:** receita, margem, churn e adoção nos 90 dias seguintes não entram em nenhuma decisão de clusterização."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, pandas as pd\n"
            "from IPython.display import display, Image, Markdown\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            "display(json.loads((ROOT/'outputs/metrics/data_audit.json').read_text(encoding='utf-8')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Data quality, escala e redundância\n\n"
            "**Comentário Técnico:** zero é preservado como valor legítimo. A ausência de `credit_utilization` é estrutural para clientes sem cartão e recebe indicador próprio. "
            "O código sentinela `9999` em login é convertido em missing antes da imputação. Caudas são winsorizadas e transformadas somente com parâmetros aprendidos em 2026Q1."
        ),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(ROOT/'outputs/tables/data_quality_by_quarter.csv'))\n"
            "display(Image(filename=str(ROOT/'outputs/figures/01_data_quality_and_scale.png')))\n"
            "display(Image(filename=str(ROOT/'outputs/figures/02_feature_correlation.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Baseline RFM e benchmark não supervisionado\n\n"
            "A baseline RFM é útil por transparência, mas reduz o relacionamento a três dimensões. Os challengers são comparados com amostragem explícita nas métricas quadráticas. "
            "Hierarchical e HDBSCAN são diagnósticos quando não oferecem scoring operacional direto."
        ),
        nbf.v4.new_code_cell(
            "bench = pd.read_csv(ROOT/'outputs/metrics/internal_model_benchmark.csv')\n"
            "display(bench)\n"
            "display(Image(filename=str(ROOT/'outputs/figures/03_internal_benchmark.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Congelamento do champion e perfis\n\n"
            f"O champion foi congelado como **{summary['champion']}**, com seis segmentos. "
            "A escolha equilibra separação, escala mínima, capacidade de scoring, estabilidade e interpretabilidade; não maximiza isoladamente Silhouette."
        ),
        nbf.v4.new_code_cell(
            "freeze = json.loads((ROOT/'outputs/metrics/taxonomy_freeze_record.json').read_text(encoding='utf-8'))\n"
            "display(freeze)\n"
            "display(pd.read_csv(ROOT/'outputs/tables/frozen_segment_profiles_q1.csv'))\n"
            "display(Image(filename=str(ROOT/'outputs/figures/04_frozen_segment_profiles.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Estabilidade temporal\n\n"
            "O pipeline e os centroides de 2026Q1 são aplicados sem refit. Mudança de rótulo pode representar evolução real do cliente ou sensibilidade de fronteira; por isso são lidos em conjunto share, retenção, ARI/NMI e drift de centroide."
        ),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(ROOT/'outputs/metrics/temporal_stability_metrics.csv'))\n"
            "display(Image(filename=str(ROOT/'outputs/figures/05_temporal_stability.png')))\n"
            "display(Image(filename=str(ROOT/'outputs/figures/06_migration_matrix.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Validação externa pós-congelamento\n\n"
            "**Análise/Interpretação:** outcomes futuros avaliam utilidade, não formam clusters. Diferenças de receita, margem, churn e adoção mostram diferenciação prospectiva, mas permanecem associativas."
        ),
        nbf.v4.new_code_cell(
            "future = pd.read_csv(ROOT/'outputs/tables/future_outcome_validation_by_segment.csv')\n"
            "display(future[future.snapshot_quarter.eq('2026Q3')][['segment_name','customers','revenue_next_90d','margin_next_90d','churn_next_90d','product_adoption_next_90d']])\n"
            "display(Image(filename=str(ROOT/'outputs/figures/07_future_outcome_validation.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Tradução em ação e limites causais\n\n"
            "Os playbooks são hipóteses operacionalizáveis com guardrails. Segmentos não demonstram que a ação sugerida causa melhora; campanhas devem ser randomizadas dentro de cada segmento e avaliadas por efeito incremental."
        ),
        nbf.v4.new_code_cell(
            "display(pd.read_csv(ROOT/'outputs/tables/segment_actionability_playbook.csv'))"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Monitoramento e conclusão executiva\n\n"
            "Monitorar participação, distância a centroides, missing, estabilidade de perfis, outcomes externos, custo de contato e guardrails. Recalibrar a taxonomia somente com evidência de drift material e benefício operacional.\n\n"
            "**Conclusão obrigatória:** O melhor clustering não é necessariamente aquele com maior Silhouette. A solução final deve equilibrar separação matemática, estabilidade temporal, interpretabilidade, diferenciação futura e capacidade de gerar ações de negócio."
        ),
        nbf.v4.new_markdown_cell(
            "## 9. Reprodução completa\n\n"
            "Para regenerar todos os artefatos a partir da base raw, execute na raiz do repositório: `python -m src.run_case`. "
            "A implementação reutilizável está em `src/`, e todos os parâmetros críticos são congelados e versionados."
        ),
    ]
    nbf.write(nb, NOTEBOOK)
    executed = nbf.read(NOTEBOOK, as_version=4)
    client = NotebookClient(executed, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(executed, NOTEBOOK)


def main() -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    _ensure_directories()
    df, audit = load_and_audit()
    plot_data_quality(df)

    # Development is isolated before any future-outcome inspection.
    dev = df.loc[df["snapshot_quarter"].eq(DEV_QUARTER)].copy()
    preprocessor = FrozenPreprocessor(FEATURES, LOG_FEATURES, scaler_kind="robust")
    x_dev = preprocessor.fit_transform(dev)
    benchmark = benchmark_candidates(dev, x_dev)
    benchmark.to_csv(METRICS / "internal_model_benchmark.csv", index=False)
    plot_benchmark(benchmark)

    model, _, mapping, freeze = freeze_champion(dev, x_dev, benchmark)
    labeled, transformed = apply_frozen_taxonomy(df, preprocessor, model, mapping)
    profile, _ = build_profiles(labeled.loc[labeled["snapshot_quarter"].eq(DEV_QUARTER)])
    shares, stability = temporal_stability(labeled, transformed, model)
    _, rfm_record = rfm_baseline(labeled)

    # Only after the freeze above are future outcomes opened for external validation.
    validation, tests = external_validation(labeled)
    actions = build_actionability(profile, validation)
    save_assignments(labeled)

    joblib.dump(preprocessor, MODELS / "frozen_preprocessor.joblib", compress=3)
    joblib.dump(model, MODELS / "frozen_kmeans_taxonomy.joblib", compress=3)
    (MODELS / "cluster_name_map.json").write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2, ensure_ascii=False), encoding="utf-8")

    q3 = validation.loc[validation["snapshot_quarter"].eq(HOLDOUT_QUARTER)]
    summary = {
        "champion": freeze["champion"],
        "segments": 6,
        "development_rows": int((df["snapshot_quarter"] == DEV_QUARTER).sum()),
        "q2_to_q3_label_retention": float(stability.loc[(stability.from_quarter == VALIDATION_QUARTER) & (stability.to_quarter == HOLDOUT_QUARTER), "label_retention"].iloc[0]),
        "rfm_q2_to_q3_label_retention": rfm_record["q2_to_q3_label_retention"],
        "q3_revenue_ratio_max_min": float(q3["revenue_next_90d"].max() / q3["revenue_next_90d"].min()),
        "q3_churn_spread_pp": float(100 * (q3["churn_next_90d"].max() - q3["churn_next_90d"].min())),
        "q3_adoption_spread_pp": float(100 * (q3["product_adoption_next_90d"].max() - q3["product_adoption_next_90d"].min())),
        "future_outcome_tests_all_p_below_0_001": bool((tests["p_value"] < 0.001).all()),
        "actionable_segments": int(len(actions)),
    }
    (METRICS / "executive_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    create_notebook(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
