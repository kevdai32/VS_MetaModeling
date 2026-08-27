"""CLI for training receptor-specific P(bind) models."""

from __future__ import annotations

import argparse
from pathlib import Path

from components import DEFAULT_FEATURE_GROUPS, list_components
from Input_output import load_labeled_pose_table
from model import train_receptor_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a receptor-specific P(bind) model.")
    parser.add_argument("--receptor_name", required=True, help="Name for this receptor/model, e.g. PXR")
    parser.add_argument("--mol2", required=True, help="Docked MOL2 file")
    parser.add_argument("--actives", required=True, help="Active compounds .smi")
    parser.add_argument("--decoys", required=True, help="Decoy compounds .smi")
    parser.add_argument("--out_dir", required=True, help="Output directory for saved receptor model")
    parser.add_argument(
        "--feature_groups",
        nargs="+",
        default=DEFAULT_FEATURE_GROUPS,
        help="Feature components to use. Run --list_components to see options.",
    )
    parser.add_argument("--n_boot", type=int, default=100, help="Bootstrap models for P(bind) uncertainty")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=7)
    parser.add_argument("--calibration_cv", type=int, default=3)
    parser.add_argument("--list_components", action="store_true", help="Print available feature groups and exit")
    args = parser.parse_args()

    if args.list_components:
        for component in list_components():
            print(f"{component['name']}: {component['description']}")
            print(f"  columns: {component['columns']}")
        return

    df = load_labeled_pose_table(args.mol2, args.actives, args.decoys)
    output, metrics = train_receptor_model(
        df=df,
        receptor_name=args.receptor_name,
        feature_groups=args.feature_groups,
        out_dir=Path(args.out_dir),
        n_boot=args.n_boot,
        test_size=args.test_size,
        random_state=args.random_state,
        calibration_cv=args.calibration_cv,
    )

    print("Done.")
    print(f"Receptor: {args.receptor_name}")
    print(f"Rows scored: {len(output)}")
    print(f"Model saved to: {Path(args.out_dir) / 'model.joblib'}")
    print("Test metrics:")
    for k, v in metrics["test_set"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
