"""
Feature construction for P(bind) models
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from components import resolve_feature_columns

RDKit_AVAILABLE = True
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
except Exception:
    RDKit_AVAILABLE = False


RDKIT_FEATURE_COLS = [
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rot_bonds",
    "ring_count",
    "heavy_atoms",
    "fraction_csp3",
]


def add_rdkit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple RDKit descriptors from the `smiles` column
    """
    out = df.copy()
    if not RDKit_AVAILABLE:
        for col in RDKIT_FEATURE_COLS:
            out[col] = np.nan
        return out

    # Idempotent: if descriptors are already present, don't recompute
    if all(col in out.columns for col in RDKIT_FEATURE_COLS):
        return out

    rows = []
    for smi in out["smiles"]:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            rows.append({col: np.nan for col in RDKIT_FEATURE_COLS})
            continue

        rows.append(
            {
                "mol_wt": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "hbd": Lipinski.NumHDonors(mol),
                "hba": Lipinski.NumHAcceptors(mol),
                "rot_bonds": Lipinski.NumRotatableBonds(mol),
                "ring_count": Lipinski.RingCount(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            }
        )

    feat_df = pd.DataFrame(rows, index=out.index)
    return pd.concat([out, feat_df], axis=1)


def prepare_feature_table(
    df: pd.DataFrame,
    feature_groups: Iterable[str],
    drop_all_missing: bool = True,
    max_missing_frac: float = 0.5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add derived features and return the usable feature matrix + selected columns

    Parameters
    df:
        One-row-per-molecule dataframe
    feature_groups:
        Component names such as docking_core, vdw, electrostatic, rdkit_basic.
    drop_all_missing:
        Drop columns that are entirely missing
    max_missing_frac:
        Drop columns whose missing-rate exceeds this threshold. Defaults to 0.5
        so that pose-aggregate features are silently dropped when most molecules
        have only one pose Without this guard, median imputation fills single-
        pose rows with the population median, which leaks label information if
        pose convergence rate differs between actives and decoys
        Set to None to disable this guard
    """
    df_feat = add_rdkit_features(df)
    requested_cols = resolve_feature_columns(feature_groups)
    available_cols = [c for c in requested_cols if c in df_feat.columns]

    if drop_all_missing:
        available_cols = [c for c in available_cols if not df_feat[c].isna().all()]

    if max_missing_frac is not None:
        available_cols = [
            c for c in available_cols
            if df_feat[c].isna().mean() <= max_missing_frac
        ]

    if not available_cols:
        raise ValueError(
            "No usable feature columns found. Check feature_groups, input docking columns, "
            "and the max_missing_frac threshold"
        )

    return df_feat, available_cols


def align_to_training_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Create an X matrix with the exact columns used during training

    Missing optional docking columns are filled with NaN and handled by the
    trained imputer
    """
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[feature_columns]
