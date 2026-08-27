"""
Predict P(bind) for molecules in a docked MOL2 file
This uses trained model to predict binding from specified features eg (Total Energy, VDW, HBA, HBD)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Input_output import parse_oedock_mol2, summarize_pose_uncertainty
from model import predict_pbind


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict P(bind) for all molecules in a docked MOL2 file"
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing saved receptor model, e.g. models/PXR_full",
    )

    parser.add_argument(
        "--mol2",
        required=True,
        help="Docked MOL2 file containing SMILES and docking score metadata",
    )

    parser.add_argument(
        "--output_csv",
        default="mol2_pbind_predictions.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    # 1. Parse raw MOL2 into one row per molecule/pose
    pose_df = parse_oedock_mol2(args.mol2)

    if pose_df.empty:
        raise ValueError(f"No molecules were parsed from {args.mol2}")

    # 2. Collapse multiple poses to one row per molecule
    molecule_df = summarize_pose_uncertainty(pose_df)

    # 3. Predict P(bind)
    pred_df = predict_pbind(
        model_dir=args.model_dir,
        records=molecule_df,
    )

    # 4. Save
    pred_df.to_csv(args.output_csv, index=False)

    summary_cols = [
        "name",
        "smiles",
        "receptor_name",
        "binding_probability",
        "binding_probability_std",
        "Total Energy",
        "binding_energy",
        "binding_energy_error",
    ]
    summary_cols = [c for c in summary_cols if c in pred_df.columns]

    print("\nDone.")
    print(f"Input MOL2: {args.mol2}")
    print(f"Rows scored: {len(pred_df)}")
    print(f"Saved: {args.output_csv}")

    print("\nPreview:")
    print(pred_df[summary_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()