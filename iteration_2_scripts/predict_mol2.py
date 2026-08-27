"""CLI for training receptor-specific P(bind) models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from components import DEFAULT_FEATURE_GROUPS, list_components
from Input_output import load_labeled_pose_table
from model import train_receptor_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a receptor-specific P(bind) model.")
    
    parser.add_argument("--receptor_name", help="Name for this receptor/model, e.g. PXR")
    parser.add_argument("--input_dir", type=str, default=None, help="Base directory containing the input files")
    parser.add_argument("--mol2", help="Filename of the docked MOL2 file")
    parser.add_argument("--actives", help="Filename of the active compounds .smi")
    parser.add_argument("--decoys", help="Filename of the decoy compounds .smi")
    parser.add_argument("--out_dir", help="Output directory for saved receptor model")
    
    # Set defaults to None so the interactive logic can detect if they were skipped
    parser.add_argument(
        "--feature_groups",
        nargs="+",
        default=None,
        help="Feature components to use. Run --list_components to see options.",
    )
    parser.add_argument("--n_boot", type=int, default=None, help="Bootstrap models for P(bind) uncertainty")
    
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

    # --- Interactive Prompting Logic ---
    prompted = False
    
    if not args.receptor_name:
        prompted = True
        print("\n--- Receptor Training Setup ---")
        args.receptor_name = input("Receptor Name (e.g., PXR): ").strip()
        
    if args.input_dir is None:
        if not prompted: print("\n--- Receptor Training Setup ---"); prompted = True
        args.input_dir = input("Base Input Directory (press Enter for current directory): ").strip()
        
    if not args.mol2:
        if not prompted: print("\n--- Receptor Training Setup ---"); prompted = True
        args.mol2 = input("MOL2 Filename (e.g., structure.mol2): ").strip()
        
    if not args.actives:
        if not prompted: print("\n--- Receptor Training Setup ---"); prompted = True
        args.actives = input("Actives Filename (e.g., actives.smi): ").strip()
        
    if not args.decoys:
        if not prompted: print("\n--- Receptor Training Setup ---"); prompted = True
        args.decoys = input("Decoys Filename (e.g., decoys.smi): ").strip()
        
    if not args.out_dir:
        if not prompted: print("\n--- Receptor Training Setup ---"); prompted = True
        args.out_dir = input("Output Directory (e.g., ./outputs): ").strip()

    # --- New Prompts for Features and Bootstraps ---
    if args.feature_groups is None:
        if not prompted: print("\n--- Receptor Training Setup ---"); prompted = True
        default_fg_str = " ".join(DEFAULT_FEATURE_GROUPS)
        fg_input = input(f"Feature groups (space-separated) [default: {default_fg_str}]: ").strip()
        # If user types something, split it into a list. Otherwise, use defaults.
        args.feature_groups = fg_input.split() if fg_input else DEFAULT_FEATURE_GROUPS

    if args.n_boot is None:
        if not prompted: print("\n--- Receptor Training Setup ---"); prompted = True
        nb_input = input("Number of bootstrap models [default: 100]: ").strip()
        try:
            # If user types a number, use it. Otherwise, default to 100.
            args.n_boot = int(nb_input) if nb_input else 100
        except ValueError:
            print("Invalid number entered for bootstraps. Defaulting to 100.")
            args.n_boot = 100

    if prompted:
        print("-" * 31 + "\n")

    # Safety check
    if not all([args.receptor_name, args.mol2, args.actives, args.decoys, args.out_dir]):
        print("Error: You must provide all core filenames and output paths to proceed.", file=sys.stderr)
        sys.exit(1)


    # --- Path Resolution Logic ---
    base_dir = Path(args.input_dir) if args.input_dir else Path("")

    mol2_file = base_dir / args.mol2
    actives_file = base_dir / args.actives
    decoys_file = base_dir / args.decoys

    # Verify they actually exist before running the heavy lifting
    missing_files = [p for p in [mol2_file, actives_file, decoys_file] if not p.exists()]
    if missing_files:
        print("Error: Missing the following input files:", file=sys.stderr)
        for f in missing_files:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Using Structure: {mol2_file}")
    print(f"Using Actives:   {actives_file}")
    print(f"Using Decoys:    {decoys_file}")
    print(f"Features:        {', '.join(args.feature_groups)}")
    print(f"Bootstraps:      {args.n_boot}")
    print("-" * 30)
    # -----------------------------

    df = load_labeled_pose_table(str(mol2_file), str(actives_file), str(decoys_file))
    
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