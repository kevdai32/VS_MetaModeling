from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def pick_score_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """
    Pick which prediction column to rank by.
    Priority:
      1. user-specified
      2. p_binding_bootstrap_mean
      3. p_binding
      4. binding_probability
    """
    candidates = []
    if preferred:
        candidates.append(preferred)

    candidates.extend([
        "p_binding_bootstrap_mean",
        "p_binding",
        "binding_probability",
    ])

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a usable score column. "
        "Expected one of: p_binding_bootstrap_mean, p_binding, binding_probability"
    )


def mols_from_df(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    name_col: str = "name",
    score_col: str = "p_binding_bootstrap_mean",
    label_col: str = "label",
    uncertainty_col: str = "p_binding_bootstrap_std",
):
    """
    Convert dataframe rows into RDKit molecules + legends for plotting
    """
    mols = []
    legends = []

    for _, row in df.iterrows():
        smi = row.get(smiles_col, None)
        if pd.isna(smi):
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        name = str(row.get(name_col, "NA"))
        score = row.get(score_col, None)
        label = row.get(label_col, None)

        score_str = f"{float(score):.3f}" if pd.notna(score) else "NA"

        if uncertainty_col in row.index and pd.notna(row.get(uncertainty_col)):
            unc = float(row[uncertainty_col])
            unc_str = f"{unc:.3f}"
        else:
            unc_str = "NA"

        label_str = "NA"
        if pd.notna(label):
            label_str = str(int(label))

        legend = (
            f"{name}\n"
            f"label={label_str}  P(bind)={score_str}\n"
            f"unc={unc_str}"
        )

        mols.append(mol)
        legends.append(legend)

    return mols, legends


def save_grid(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    score_col: str,
    top_n: int = 20,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (300, 250),
):
    """
    Save a molecule grid PNG.
    """
    df = df.head(top_n).copy()

    mols, legends = mols_from_df(df, score_col=score_col)

    if not mols:
        print(f"[WARN] No valid molecules to draw for {title}")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        useSVG=False,
    )

    img.save(str(output_path))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize top predicted molecules from predictions.csv"
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to predictions.csv from your trained model",
    )
    parser.add_argument(
        "--output_dir",
        default="model_visualizations",
        help="Directory to save PNG outputs",
    )
    parser.add_argument(
        "--score_col",
        default=None,
        help="Optional score column to rank by. "
             "Defaults to p_binding_bootstrap_mean, then p_binding, then binding_probability.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="How many molecules to show in each grid",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    score_col = pick_score_column(df, args.score_col)

    print(f"Using score column: {score_col}")
    print(f"Loaded {len(df)} rows")

    # Top predicted molecules overall
    df_top = df.sort_values(score_col, ascending=False)

    save_grid(
        df=df_top,
        output_path=output_dir / "top_predicted_overall.png",
        title="Top predicted molecules overall",
        score_col=score_col,
        top_n=args.top_n,
    )

    # Top predicted decoys (likely false positives / active-like decoys)
    if "label" in df.columns:
        df_decoys = df[df["label"] == 0].sort_values(score_col, ascending=False)

        save_grid(
            df=df_decoys,
            output_path=output_dir / "top_predicted_decoys.png",
            title="Top predicted decoys",
            score_col=score_col,
            top_n=args.top_n,
        )

        # Missed actives = real actives with low predicted probability
        df_missed_actives = df[df["label"] == 1].sort_values(score_col, ascending=True)

        save_grid(
            df=df_missed_actives,
            output_path=output_dir / "missed_actives.png",
            title="Missed actives",
            score_col=score_col,
            top_n=args.top_n,
        )

        # Also useful: top predicted true actives
        df_top_actives = df[df["label"] == 1].sort_values(score_col, ascending=False)

        save_grid(
            df=df_top_actives,
            output_path=output_dir / "top_predicted_actives.png",
            title="Top predicted actives",
            score_col=score_col,
            top_n=args.top_n,
        )

    print("\nDone.")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()