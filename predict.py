"""CLI for predicting the metamodel coupling payload using a saved receptor-specific model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model import predict_pbind

def _record_from_single_input(smiles: str, scores_json: str | None) -> pd.DataFrame:
    record = {"smiles": smiles}
    if scores_json:
        score_dict = json.loads(scores_json)
        record.update(score_dict)
    return pd.DataFrame([record])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the coupling payload (binding probability + energy + "
        "uncertainties + per-feature contributions) for new docked molecules."
    )
    parser.add_argument("--model_dir", required=True, help="Directory containing model.joblib")
    parser.add_argument("--input_csv", help="CSV with smiles and optional docking-score columns")
    parser.add_argument("--smiles", help="Single SMILES string")
    parser.add_argument(
        "--scores_json",
        help='Optional docking scores for a single molecule, e.g. \'{"Total Energy": -9.4, "Van der Waals": -7.1}\'',
    )
    parser.add_argument("--output_csv", default="pbind_predictions.csv")
    args = parser.parse_args()

    if args.input_csv:
        records = pd.read_csv(args.input_csv)
    elif args.smiles:
        records = _record_from_single_input(args.smiles, args.scores_json)
    else:
        raise ValueError("Provide either --input_csv or --smiles.")

    if "smiles" not in records.columns:
        raise ValueError("Input must contain a 'smiles' column.")

    out = predict_pbind(args.model_dir, records)
    out.to_csv(args.output_csv, index=False)

    summary_cols = [
        "smiles",
        "receptor_name",
        "binding_probability",
        "binding_probability_std",
        "binding_energy",
        "binding_energy_error",
    ]
    summary_cols = [c for c in summary_cols if c in out.columns]

    print("Done.")
    print(out[summary_cols].to_string(index=False))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
