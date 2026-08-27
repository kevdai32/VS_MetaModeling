# Iteration 2

Receptor-specific P(bind) model training and prediction scripts for docking data.

## Contents

- `train_receptor.py` trains a receptor-specific binding probability model from a MOL2 file plus active/decoy `.smi` files.
- `predict.py` scores molecules from a CSV or a single SMILES input using a saved model.
- `predict_mol2.py` parses a docked MOL2 file, collapses poses, and scores molecules with a saved model.
- `model.py`, `features.py`, `components.py`, and `Input_output.py` contain the model, feature, component, and parsing utilities.
- `models/` and `test/` contain saved outputs and example prediction artifacts.
- `iteration_2_scripts/all_poses_20.mol2` is tracked with Git LFS because it exceeds GitHub's normal 100 MB file limit.

## Requirements

Core Python dependencies include:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `matplotlib`
- `rdkit` for RDKit-derived features and molecule visualizations

## Example Usage

Train a model:

```bash
python train_receptor.py \
  --receptor_name PXR \
  --input_dir iteration_2_scripts \
  --mol2 all_poses_20.mol2 \
  --actives actives_clusters.smi \
  --decoys decoys_clusters.smi \
  --out_dir models/PXR_octant
```

Predict from a MOL2 file:

```bash
python predict_mol2.py \
  --model_dir models/PXR_octant \
  --mol2 iteration_2_scripts/all_poses_20.mol2 \
  --output_csv mol2_pbind_predictions.csv
```
