#!/usr/bin/env python3
"""
Evaluation script for generated molecule distributions.

This script computes various metrics on a dataframe containing generated molecules
with columns: smiles, qed, sa

Usage:
    python evaluate_molecules.py --input_df path/to/dataframe.csv --output_path results.json
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
import safe as sf
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional

from moose.utils.utils_chem import (
    safes_to_smiles_and_mols,
    uniqueness,
    internal_diversity,
    fragment_diversity,
    scaffold_diversity,
    mol_to_fp,
    smiles_to_mol,
    mol_to_canonical_smiles,
    loose_novelty,
    snn,
    load_fps,
)


def smiles_to_safe(smiles: str) -> Optional[str]:
    """
    Convert SMILES string to SAFE string.

    Args:
        smiles: SMILES string to convert.

    Returns:
        SAFE string if successful, None otherwise.
    """
    try:
        safe_str = sf.encode(smiles, canonical=True)
        return safe_str
    except Exception:
        return None


def compute_metrics(
    df: pd.DataFrame,
    train_smiles: Optional[list] = None,
    train_fps: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics on the input dataframe.
    If no training SMILES -- no novelty
    If no training SMILES and no training fingerprints -- no SNN

    Args:
        df: DataFrame with columns 'smiles', 'qed', 'sa'
        train_smiles: Optional list of training set SMILES for novelty computation
        train_fps: Optional list of training set fingerprints for SNN computation

    Returns:
        Dictionary of computed metrics.
    """
    results = {}

    # Extract SMILES and convert to canonical
    smiles_list = df["smiles"].tolist()
    canonical_smiles = []
    valid_mols = []

    print("Converting SMILES to canonical format and RDKit Mol objects...")
    for smi in smiles_list:
        mol = smiles_to_mol(smi)
        if mol is not None:
            can_smi = mol_to_canonical_smiles(mol)
            canonical_smiles.append(can_smi)
            valid_mols.append(mol)
        else:
            # Keep invalid ones as None for tracking
            canonical_smiles.append(None)
            valid_mols.append(None)

    valid_canonical = [s for s in canonical_smiles if s is not None]
    valid_mols_filtered = [m for m in valid_mols if m is not None]

    results["num_total"] = len(smiles_list)
    results["num_valid"] = len(valid_canonical)
    results["validity"] = len(valid_canonical) / len(smiles_list) if smiles_list else 0.0

    if not valid_canonical:
        print("Warning: No valid molecules found. Returning empty results.")
        return results

    # 1. Uniqueness
    print("Computing uniqueness...")
    results["uniqueness"] = uniqueness(valid_canonical)

    # 2. Internal Diversity
    print("Computing internal diversity...")
    results["internal_diversity"] = internal_diversity(valid_mols_filtered)

    # 3. Fragment Diversity (requires SAFE strings)
    print("Converting SMILES to SAFE for fragment diversity...")
    safe_strings = []
    for smi in valid_canonical:
        safe_str = smiles_to_safe(smi)
        if safe_str is not None:
            safe_strings.append(safe_str)

    if safe_strings:
        print("Computing fragment diversity...")
        global_frag_div, mean_unique_per_mol = fragment_diversity(safe_strings)
        results["fragment_diversity_global"] = global_frag_div
        results["fragment_diversity_mean_unique_per_mol"] = mean_unique_per_mol
    else:
        print("Warning: Could not convert any SMILES to SAFE for fragment diversity.")
        results["fragment_diversity_global"] = 0.0
        results["fragment_diversity_mean_unique_per_mol"] = 0.0

    # 4. Scaffold Diversity
    print("Computing scaffold diversity...")
    scaffold_div, num_unique_scaffolds = scaffold_diversity(valid_mols_filtered)
    results["scaffold_diversity"] = scaffold_div
    results["num_unique_scaffolds"] = num_unique_scaffolds

    # 5. Property statistics (QED and SA)
    print("Computing property statistics...")

    # Match valid molecules with their properties
    valid_indices = [i for i, s in enumerate(canonical_smiles) if s is not None]
    valid_qed = df.iloc[valid_indices]["qed"].values if "qed" in df.columns else None
    valid_sa = df.iloc[valid_indices]["sa"].values if "sa" in df.columns else None

    if valid_qed is not None:
        results["qed_mean"] = float(np.mean(valid_qed))
        results["qed_std"] = float(np.std(valid_qed))
        results["qed_min"] = float(np.min(valid_qed))
        results["qed_max"] = float(np.max(valid_qed))

    if valid_sa is not None:
        results["sa_mean"] = float(np.mean(valid_sa))
        results["sa_std"] = float(np.std(valid_sa))
        results["sa_min"] = float(np.min(valid_sa))
        results["sa_max"] = float(np.max(valid_sa))

    # 6. Novelty (if training set provided)
    if train_smiles is not None and train_fps is None:
        ## Compute and cache the train_fps
        print("Computing and caching training fingerprints...")
        train_mols = [smiles_to_mol(smi) for smi in train_smiles]
        train_mols = [mol for mol in train_mols if mol is not None]  # Filter out None values
        train_fps = [mol_to_fp(mol) for mol in train_mols]
        with open("train_fps.pkl", "wb") as f:
            pickle.dump(train_fps, f)

    elif train_smiles is not None:
        print("Computing novelty...")

        # Convert training SMILES to canonical
        train_canonical = []
        for smi in train_smiles:
            mol = smiles_to_mol(smi)
            if mol is not None:
                train_canonical.append(mol_to_canonical_smiles(mol))
        results["novelty"] = loose_novelty(valid_canonical, train_canonical)

    # 7. SNN (Similarity to Nearest Neighbor) if training fingerprints provided
    if train_fps is not None:
        print("Computing SNN...")
        gen_fps = [mol_to_fp(mol) for mol in valid_mols_filtered]
        snn_frac, max_sims = snn(gen_fps, train_fps)
        results["snn_fraction"] = snn_frac
        results["snn_mean_max_similarity"] = float(np.mean(max_sims)) if max_sims else 0.0

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated molecule distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-input-df",
        type=str,
        required=True,
        help="Path to input CSV file with columns: smiles, qed, sa",
    )
    parser.add_argument(
        "-output-path",
        type=str,
        default="evaluation_results.json",
        help="Path to folder to house outputs ",
    )
    parser.add_argument(
        "-train-smiles",
        type=str,
        default=None,
        help="Optional path to CSV file with training SMILES (for novelty computation)",
    )
    parser.add_argument(
        "-train-fps",
        type=str,
        default=None,
        help="Optional path to pickled training fingerprints (for SNN computation)",
    )
    parser.add_argument(
        "--cache-fps", action="store_true", help="Cache the generated molecules as fingerprints?"
    )

    args = parser.parse_args()

    # Load input dataframe
    print(f"Loading input dataframe from {args.input_df}...")
    df = pd.read_csv(args.input_df)

    if "smiles" not in df.columns:
        raise ValueError(f"Missing required columns: smiles")

    ## Load the generative model's training data if provided
    train_smiles = None
    if args.train_smiles:
        print(f"Loading training SMILES from {args.train_smiles}...")
        train_df = pd.read_csv(args.train_smiles)
        if "smiles" in train_df.columns:
            train_smiles = train_df["smiles"].tolist()
        else:
            print(
                f"Warning: 'smiles' column not found in {args.train_smiles}. Skipping novelty computation."
            )
    else:
        print(f"No training data provided. Skipping novelty computation.")

    train_fps = None
    if args.train_fps:
        print(f"Loading training fingerprints from {args.train_fps}...")
        train_fps = load_fps(args.train_fps)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.cache_fps:
        print("Caching the generated molecules as fingerprints")
        fps = []
        for smiles in tqdm(df["smiles"]):
            mol = smiles_to_mol(smiles)
            if mol is not None:
                fps.append(mol_to_fp(mol, radius=2, n_bits=2048))
        with open(output_path / "generated_fps.pkl", "wb") as f:
            pickle.dump(fps, f)
        print(f"Fingerprints cached to {output_path / 'generated_fps.pkl'}")

    # Compute metrics
    print("\n" + "=" * 50)
    print("Computing evaluation metrics...")
    print("=" * 50 + "\n")

    results = compute_metrics(df, train_smiles=train_smiles, train_fps=train_fps)

    # Save results

    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
