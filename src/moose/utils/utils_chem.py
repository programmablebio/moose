# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
from tqdm import tqdm
import pickle
import numpy as np
import safe as sf
import datamol as dm
from contextlib import suppress
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
from typing import List, Set, Optional, Iterable, Tuple, Generator


# https://github.com/datamol-io/safe/blob/main/safe/sample.py
# https://github.com/jensengroup/GB_GA/blob/master/crossover.py
def safe_to_smiles(safe_str: str, fix: bool = True) -> Optional[str]:
    if fix:
        safe_str = ".".join(
            [
                frag
                for frag in safe_str.split(".")
                if sf.decode(frag, ignore_errors=True) is not None
            ]
        )
    return sf.decode(safe_str, canonical=True, ignore_errors=True)


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Return kekulized RDKit Mol object or None if invalid.

    Args:
        smiles: SMILES string to convert to RDKit Mol object.

    Returns:
        Kekulized RDKit Mol object or None if invalid.
    """

    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        # If kekulization fails, return None
        return None
    return mol


def mol_to_canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def mol_to_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def cache_fps(
    mols: List[Chem.Mol], radius: int = 2, n_bits: int = 2048, save_path: str = "fps.pkl"
) -> List[np.ndarray]:
    """
    Cache fingerprints of molecules.

    Args:
        mols: List of RDKit Mol objects.
        radius: Radius for Morgan fingerprint.
        n_bits: Number of bits for Morgan fingerprint.
        save_path: Path to save the cached fingerprints.
    """

    fps = [mol_to_fp(mol, radius, n_bits) for mol in mols]
    with open(save_path, "wb") as f:
        pickle.dump(fps, f)


def load_fps(save_path: str = "fps.pkl") -> List[np.ndarray]:
    """
    Load cached fingerprints of molecules.

    Args:
        save_path: Path to load the cached fingerprints.

    Returns:
        List of cached fingerprints.
    """

    with open(save_path, "rb") as f:
        fps = pickle.load(f)
    return fps


def safes_to_smiles_and_mols(
    safes: Iterable[str],
) -> Tuple[List[str], List[Chem.Mol]]:
    """
    Batch convert SAFE strings to canonical SMILES + Mol objects.
    Skips any that fail decoding.

    Args:
        safes: Iterable of SAFE strings to convert.

    Returns:
        Tuple of lists: [canonical SMILES, RDKit Mol objects].
    """

    smiles_list = []
    mols = []

    for s in safes:
        smi = safe_to_smiles(s)
        if smi is None:
            continue
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        can = mol_to_canonical_smiles(mol)
        smiles_list.append(can)
        mols.append(mol)

    return smiles_list, mols


def filter_by_substructure(sequences: List[str], substruct: str) -> List[str]:
    substruct = sf.utils.standardize_attach(substruct)
    substruct = Chem.DeleteSubstructs(Chem.MolFromSmarts(substruct), Chem.MolFromSmiles("*"))
    substruct = Chem.MolFromSmarts(Chem.MolToSmiles(substruct))
    return sf.utils.filter_by_substructure_constraints(sequences, substruct)


def mix_sequences(
    prefix_sequences: List[str],
    suffix_sequences: List[str],
    prefix: str,
    suffix: str,
    num_samples: int = 1,
) -> List[str]:

    mol_linker_slicer = sf.utils.MolSlicer(require_ring_system=False)

    prefix_linkers = []
    suffix_linkers = []
    prefix_query = dm.from_smarts(prefix)
    suffix_query = dm.from_smarts(suffix)

    for x in prefix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, prefix_query)
            prefix_linkers.append(out[1])

    for x in suffix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, suffix_query)
            suffix_linkers.append(out[1])

    n_linked = 0
    linked = []
    linkers = prefix_linkers + suffix_linkers
    linkers = [x for x in linkers if x is not None]
    for n_linked, linker in enumerate(linkers):
        linked.extend(mol_linker_slicer.link_fragments(linker, prefix, suffix))
        if n_linked > num_samples:
            break
        linked = [x for x in linked if x]
    return linked[:num_samples]


def cut(smiles: str) -> Set[str]:
    def cut_nonring(mol: Chem.Mol) -> Optional[List[Chem.Mol]]:
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
            return None

        bis = random.choice(
            mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]"))
        )  # single bond not in ring
        bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

        try:
            return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        except ValueError:
            return None

    mol = Chem.MolFromSmiles(smiles)
    frags = set()
    # non-ring cut
    for _ in range(3):
        frags_nonring = cut_nonring(mol)
        if frags_nonring is not None:
            frags |= set([Chem.MolToSmiles(f) for f in frags_nonring])
    return frags


class Slicer:
    def __call__(self, mol: Chem.Mol) -> Generator[List[int], None, None]:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        # non-ring single bonds
        bonds = mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]"))
        for bond in bonds:
            yield bond


def uniqueness(smiles_list: List[str]) -> float:
    """
    Num unique SMILES / Num total SMILES.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Uniqueness score as float.
    """

    if not smiles_list:
        return 0.0
    return len(set(smiles_list)) / len(smiles_list)


def loose_novelty(
    gen_smiles: List[str],
    train_smiles: Iterable[str],
) -> float:
    """
    Novel here means SMILES string that is not in training set.
    Loose definition.

    Args:
        gen_smiles: canonical SMILES from generated set.
        train_smiles: canonical SMILES in the training/reference set.

    Returns:
        Novelty score as float.
    """

    if not gen_smiles:
        return 0.0

    train_set = set(train_smiles)
    novel = [s for s in gen_smiles if s not in train_set]

    return len(novel) / len(gen_smiles)


def snn(gen_fps: List, train_fps: Iterable, threshold: float = 0.4) -> Tuple[float, List[float]]:
    """
    Similarity to Nearest Neighbor (SNN).

    'Novel' here means structurally disjoint from training set according to
    a fingerprint similarity threshold.

    Threshold (0.4) is from Wengong Jin's multi-objective molecule generation paper:
    https://people.csail.mit.edu/tommi/papers/JBJ_ICML2020b.pdf

    Args:
        gen_fps: RDKit fingerprints for generated molecules.
        train_fps: RDKit fingerprints for training/reference molecules.
        threshold: Threshold for structural disjointness.

    Returns:
        passed_frac: fraction of generated molecules whose max similarity to
                     any training molecule is < threshold.
        max_sims: list of max similarities (one per generated molecule).
    """
    if not gen_fps:
        return 0.0, []

    train_fps = list(train_fps)

    if len(train_fps) == 0:
        return 1.0, [0.0] * len(gen_fps)

    passed = 0
    max_sims: List[float] = []

    for fp in tqdm(gen_fps, desc="Calculating SNN"):
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        maxsim = max(sims) if sims else 0.0
        max_sims.append(maxsim)
        if maxsim < threshold:
            passed += 1

    passed_frac = passed / len(gen_fps)

    return passed_frac, max_sims


def internal_diversity(
    mols: List[Chem.Mol],
    sample_pairs: int = 50_000,
    radius: int = 2,
    n_bits: int = 2048,
    seed: int = 0,
) -> float:
    """
    Approximate internal diversity as average Tanimoto distance over
    ~ ~ ~random pairs~ ~ ~ of Morgan fingerprints.

    Args:
        mols: List of RDKit Mol objects.
        sample_pairs: Number of random pairs to sample.
        radius: Radius for Morgan fingerprint.
        n_bits: Number of bits for Morgan fingerprint.
        seed: Random seed.

    Returns:
        Internal diversity score as float.
    """

    n = len(mols)
    if n < 2:
        return 0.0

    fps = [mol_to_fp(m, radius=radius, n_bits=n_bits) for m in mols]

    total_pairs = n * (n - 1) // 2
    if sample_pairs is None or sample_pairs >= total_pairs:
        ## Exhaustive
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:
        random.seed(seed)
        pairs = set()
        while len(pairs) < sample_pairs:
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pairs.add((i, j))
        pairs = list(pairs)

    distances = []
    for i, j in pairs:
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        distances.append(1.0 - sim)

    return sum(distances) / len(distances)


## Fragment diversity operates on SAFEs
def fragment_diversity(safes: Iterable[str]) -> Tuple[float, float]:
    """
    Finds the golbal fragment diversity and mean # of unique frags per mol
    Takes SAFEs!

    Args:
        safes: Iterable of SAFE strings.

    Returns:
        global_frag_div: Global fragment diversity.
        mean_unique_per_mol: Mean # of unique fragments per mol.
    """

    safes = list(safes)
    if not safes:
        return 0.0, 0.0

    all_tokens = []
    unique_per_mol = []

    for s in safes:
        tokens = s.strip().split(".")
        if not tokens:
            continue
        all_tokens.extend(tokens)
        unique_per_mol.append(len(set(tokens)))

    if not all_tokens:
        return 0.0, 0.0

    global_frag_div = len(set(all_tokens)) / len(all_tokens)
    mean_unique_per_mol = sum(unique_per_mol) / len(unique_per_mol)

    return global_frag_div, mean_unique_per_mol


def mol_to_scaffold(mol: Chem.Mol) -> Optional[str]:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def scaffold_diversity(mols: List[Chem.Mol]) -> Tuple[float, int]:
    """
    Args:
        mols: List of RDKit Mol objects.

    Returns:
        scaffold_div: Scaffold diversity.
        num_unique_scaffolds: Number of unique scaffolds.
    """

    scaffolds = []
    for m in mols:
        scaf = mol_to_scaffold(m)
        if scaf is not None:
            scaffolds.append(scaf)

    if not scaffolds:
        return 0.0, 0

    num_mols_with_scaffold = len(scaffolds)
    unique_scaffolds = set(scaffolds)
    scaffold_div = len(unique_scaffolds) / num_mols_with_scaffold

    return scaffold_div, len(unique_scaffolds)
