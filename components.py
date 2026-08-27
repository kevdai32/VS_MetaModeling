"""
Feature component registry for modular receptor-specific P(bind) models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class FeatureComponent:
    """
    A named group of feature columns that can be enabled/disabled
    """

    name: str
    columns: List[str]
    description: str


COMPONENTS: Dict[str, FeatureComponent] = {
    "total_energy": FeatureComponent(
        name="total_energy",
        columns=["Total Energy"],
        description="Total docking energy only Minimal feature set for P(bind); the default.",
    ),
    # "docking_core": FeatureComponent(
    #     name="docking_core",
    #     columns=["Total Energy"] #"Ligand Energy"],
    #     description="Total docking energy plus ligand internal energy.",
    # ),
    "docking_rank": FeatureComponent(
        name="docking_rank",
        columns=["rank"],
        description=(
            "Docking pose rank. Off by default: rank is derived from Total Energy "
            "within a single screen and does not generalize to compounds outside it."
        ),
    ),
    "vdw": FeatureComponent(
        name="vdw",
        columns=["Van der Waals"],
        description="Van der Waals docking score contribution.",
    ),
    "electrostatic": FeatureComponent(
        name="electrostatic",
        columns=["Electrostatic"],
        description="Electrostatic docking score contribution.",
    ),
    "desolvation": FeatureComponent(
        name="desolvation",
        columns=["Ligand Polar Desolv", "Ligand Apolar Desolv"],
        description="Polar and apolar desolvation docking terms.",
    ),
    "strain": FeatureComponent(
        name="strain",
        columns=["Total Strain", "Max Strain"],
        description="Ligand strain terms from docking.",
    ),
    "pose_uncertainty": FeatureComponent(
        name="pose_uncertainty",
        columns=[
            "n_poses",
            "binding_energy_mean_pose",
            "binding_energy_error_pose",
            "binding_energy_range_pose",
            "best_second_pose_gap",
        ],
        description="Pose-level energy dispersion Only meaningful when multiple poses per molecule are available",
    ),
    "rdkit_basic": FeatureComponent(
        name="rdkit_basic",
        columns=[
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
        description="Basic SMILES-derived RDKit descriptors.",
    ),
}


# Minimal default: P(bind) from Total Energy alone. The other components stay
# in the registry as optional add-ons enabled via --feature_groups
DEFAULT_FEATURE_GROUPS = ["total_energy"]


def list_components() -> List[dict]:
    """
    Return component metadata for printing or UI display
    """
    return [
        {"name": c.name, "columns": c.columns, "description": c.description}
        for c in COMPONENTS.values()
    ]


def resolve_feature_columns(feature_groups: Iterable[str]) -> List[str]:
    """
    Expand feature group names into an ordered, de-duplicated feature-column list
    """
    cols: List[str] = []
    unknown = []

    for group in feature_groups:
        if group not in COMPONENTS:
            unknown.append(group)
            continue
        for col in COMPONENTS[group].columns:
            if col not in cols:
                cols.append(col)

    if unknown:
        valid = ", ".join(sorted(COMPONENTS))
        raise ValueError(f"Unknown feature group(s): {unknown}. Valid groups: {valid}")

    return cols
