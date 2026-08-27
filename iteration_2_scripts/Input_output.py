from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DOCKING_SCORE_COLS = [
    "Total Energy",
    "Electrostatic",
    "Van der Waals",
    "Ligand Polar Desolv",
    "Ligand Apolar Desolv",
    "Total Strain",
    "Max Strain",
    "Ligand Energy",
]


def parse_oedock_mol2(file_path: str | Path) -> pd.DataFrame:
    """
    Parse an OEDock-style MOL2 file with score metadata lines like:

        ########## Total Energy: -37.95

    Returns:
        pd.DataFrame with one row per molecule/pose.
    """
    file_path = Path(file_path)
    molecules: List[Dict] = []
    current: Dict = {}
    score_keys = set(DOCKING_SCORE_COLS)

    with file_path.open() as f:
        for line in f:
            if "##########" not in line:
                continue

            parts = line.split("##########", 1)[-1].strip()
            if ":" not in parts:
                continue

            key, val = parts.split(":", 1)
            key = key.strip()
            val = val.strip()

            if key == "Name":
                if current and "Total Energy" in current:
                    molecules.append(current)
                current = {"name": val}

            elif key == "SMILES":
                current["smiles"] = val

            elif key == "Rank":
                try:
                    current["rank"] = int(val)
                except ValueError:
                    current["rank"] = np.nan

            elif key in score_keys:
                try:
                    current[key] = float(val)
                except ValueError:
                    current[key] = np.nan

            # In your MOL2 output, Ligand Energy appears near the end of the score block.
            # This is a convenient point to close the current molecule/pose.
            if key == "Ligand Energy" and current:
                molecules.append(current)
                current = {}

    if current and "Total Energy" in current:
        molecules.append(current)

    return pd.DataFrame(molecules)


def parse_smi(filepath: str | Path) -> Dict[str, str]:
    """
    Parse a .smi file.

    Expected format:
        SMILES  COMPOUND_ID

    Returns:
        {compound_id: smiles}
    """
    compounds: Dict[str, str] = {}
    filepath = Path(filepath)

    with filepath.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 2:
                smiles, compound_id = parts[0], parts[1]
            else:
                smiles = parts[0]
                compound_id = parts[0]

            compounds[compound_id] = smiles

    return compounds


def add_labels(
    df: pd.DataFrame,
    active_map: Dict[str, str],
    decoy_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Add active/decoy labels using compound names.

    active = 1
    decoy = 0
    unlabeled = NaN
    """
    out = df.copy()
    active_ids = set(active_map.keys())
    decoy_ids = set(decoy_map.keys())

    out["label"] = np.where(
        out["name"].isin(active_ids),
        1,
        np.where(out["name"].isin(decoy_ids), 0, np.nan),
    )

    return out


def summarize_pose_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional:
    Collapse multiple poses to one row per molecule and summarize pose-level uncertainty.

    If only one pose is present per molecule, pose-level uncertainty columns are NaN.
    """
    required = {"name", "smiles", "Total Energy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for pose summary: {sorted(missing)}")

    rows = []

    for name, group in df.groupby("name", sort=False):
        # More negative Total Energy is treated as better.
        group = group.sort_values("Total Energy", ascending=True)

        energies = group["Total Energy"].astype(float).to_numpy()
        best_row = group.iloc[0].to_dict()

        n_poses = len(group)
        best_energy = float(energies[0])
        mean_energy = float(np.mean(energies))
        sd_energy = float(np.std(energies, ddof=1)) if n_poses > 1 else np.nan
        energy_range = float(np.max(energies) - np.min(energies)) if n_poses > 1 else np.nan
        best_second_gap = float(energies[1] - energies[0]) if n_poses > 1 else np.nan

        best_row.update(
            {
                "n_poses": n_poses,
                "binding_energy": best_energy,
                "binding_energy_mean_pose": mean_energy,
                "binding_energy_error_pose": sd_energy,
                "binding_energy_range_pose": energy_range,
                "best_second_pose_gap": best_second_gap,
            }
        )

        rows.append(best_row)

    return pd.DataFrame(rows)


def load_labeled_pose_table(
    mol2_path: str | Path,
    actives_path: str | Path,
    decoys_path: str | Path,
) -> pd.DataFrame:
    """
    Load MOL2 + active/decoy files into a labeled one-row-per-molecule table.
    """
    raw_pose_df = parse_oedock_mol2(mol2_path)
    active_map = parse_smi(actives_path)
    decoy_map = parse_smi(decoys_path)

    labeled_pose_df = add_labels(raw_pose_df, active_map, decoy_map)
    molecule_df = summarize_pose_uncertainty(labeled_pose_df)

    return molecule_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse docking MOL2 and active/decoy SMI files."
    )

    parser.add_argument(
        "--input_dir",
        default=".",
        help="Directory containing input files. Default: current directory.",
    )

    parser.add_argument(
        "--mol2",
        default="top_poses.mol2",
        help="MOL2 filename inside input_dir. Default: top_poses.mol2",
    )

    parser.add_argument(
        "--actives",
        default="actives_clusters.smi",
        help="Actives filename inside input_dir. Default: actives_clusters.smi",
    )

    parser.add_argument(
        "--decoys",
        default="decoys_clusters.smi",
        help="Decoys filename inside input_dir. Default: decoys_clusters.smi",
    )

    parser.add_argument(
        "--output_csv",
        default="parsed_pose_table.csv",
        help="Output CSV path. Default: parsed_pose_table.csv",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    mol2_path = input_dir / args.mol2
    actives_path = input_dir / args.actives
    decoys_path = input_dir / args.decoys

    for path in [mol2_path, actives_path, decoys_path]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find input file: {path}")

    df = load_labeled_pose_table(
        mol2_path=mol2_path,
        actives_path=actives_path,
        decoys_path=decoys_path,
    )

    print("\nParsed labeled pose table")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nInput files")
    print(f"MOL2:    {mol2_path}")
    print(f"Actives: {actives_path}")
    print(f"Decoys:  {decoys_path}")

    print("\nLabel counts")
    print(df["label"].value_counts(dropna=False))

    preview_cols = [
        "name",
        "smiles",
        "label",
        "Total Energy",
        "binding_energy",
        "n_poses",
        "binding_energy_error_pose",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nPreview")
    print(df[preview_cols].head())

    df.to_csv(args.output_csv, index=False)
    print(f"\nWrote parsed table to: {args.output_csv}")

if __name__ == "__main__":
    main()
