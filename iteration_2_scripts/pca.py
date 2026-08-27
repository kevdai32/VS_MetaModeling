import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURE_GROUPS = {
    "total_energy": ["Total Energy"],
    "vdw": ["Van der Waals"],
    "electrostatic": ["Electrostatic"],
    "desolvation": ["Ligand Polar Desolv", "Ligand Apolar Desolv"],
    "strain": ["Total Strain", "Max Strain"],
    "rdkit_basic": [
        "mol_wt",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rot_bonds",
        "ring_count",
        "heavy_atoms",
        "fraction_csp3",
    ],
}


def load_model_outputs(model_dir: str | Path):
    model_dir = Path(model_dir)

    pred_path = model_dir / "predictions.csv"
    config_path = model_dir / "config.json"

    df = pd.read_csv(pred_path)

    with open(config_path) as f:
        config = json.load(f)

    feature_cols = config["feature_columns"]

    return df, feature_cols


def pca_feature_matrix(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    scores_df = df.copy()
    scores_df["PC1"] = scores[:, 0]
    scores_df["PC2"] = scores[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1_loading", "PC2_loading"],
    )

    loadings["PC1_abs"] = loadings["PC1_loading"].abs()
    loadings["PC2_abs"] = loadings["PC2_loading"].abs()
    loadings["PC1_sq"] = loadings["PC1_loading"] ** 2
    loadings["PC2_sq"] = loadings["PC2_loading"] ** 2

    explained = pca.explained_variance_ratio_

    return scores_df, loadings, explained


def plot_pca_scatter(scores_df: pd.DataFrame, explained, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    labels = scores_df["label"].astype(int)

    for label_value, label_name in [(0, "Decoy"), (1, "Active")]:
        subset = scores_df[labels == label_value]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=label_name,
            alpha=0.75,
            s=35,
        )

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("PCA of P(bind) Feature Space")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_loading_bars(loadings: pd.DataFrame, output_path: str, top_n: int = 12):
    top_pc1 = loadings.sort_values("PC1_abs", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_pc1.index[::-1], top_pc1["PC1_loading"].iloc[::-1])
    ax.set_xlabel("PC1 loading")
    ax.set_title(f"Top {top_n} Feature Loadings on PC1")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def group_contributions(loadings: pd.DataFrame):
    rows = []

    for group_name, cols in FEATURE_GROUPS.items():
        cols_present = [c for c in cols if c in loadings.index]

        if not cols_present:
            continue

        pc1_contrib = loadings.loc[cols_present, "PC1_sq"].sum()
        pc2_contrib = loadings.loc[cols_present, "PC2_sq"].sum()

        rows.append(
            {
                "feature_group": group_name,
                "n_features": len(cols_present),
                "PC1_contribution": pc1_contrib,
                "PC2_contribution": pc2_contrib,
                "PC1_PC2_total": pc1_contrib + pc2_contrib,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("PC1_PC2_total", ascending=False)

    return out


def plot_group_contributions(group_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_df = group_df.sort_values("PC1_PC2_total", ascending=True)

    ax.barh(plot_df["feature_group"], plot_df["PC1_PC2_total"])
    ax.set_xlabel("Summed squared loading contribution to PC1 + PC2")
    ax.set_title("Feature Group Contribution to PCA Space")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    model_dir = Path("models/PXR_full")
    output_dir = model_dir / "pca_outputs"
    output_dir.mkdir(exist_ok=True)

    df, feature_cols = load_model_outputs(model_dir)

    print("\nFeature columns used:")
    for col in feature_cols:
        print(f"  {col}")

    scores_df, loadings, explained = pca_feature_matrix(df, feature_cols)

    print("\nExplained variance:")
    print(f"PC1: {explained[0] * 100:.2f}%")
    print(f"PC2: {explained[1] * 100:.2f}%")

    group_df = group_contributions(loadings)

    print("\nFeature group PCA contributions:")
    print(group_df)

    scores_df.to_csv(output_dir / "pca_scores.csv", index=False)
    loadings.to_csv(output_dir / "pca_loadings.csv")
    group_df.to_csv(output_dir / "pca_group_contributions.csv", index=False)

    plot_pca_scatter(
        scores_df,
        explained,
        output_dir / "pca_scatter_active_decoy.png",
    )

    plot_loading_bars(
        loadings,
        output_dir / "pca_pc1_top_loadings.png",
    )

    plot_group_contributions(
        group_df,
        output_dir / "pca_group_contributions.png",
    )

    print(f"\nSaved PCA outputs to: {output_dir}")


if __name__ == "__main__":
    main()