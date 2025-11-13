# Changes Made: Converting TR2-D2 from Peptides to Molecules

This document summarizes all changes made to convert TR2-D2 from working with peptides to working with molecules (SAFE format).

## Overview

The codebase has been converted to support molecule generation using:
- **GenMol's BERT backbone** instead of RoFormer
- **SAFE tokenizer** instead of SMILES/peptide tokenizer
- **Molecule scoring functions** (QED, SA) instead of peptide properties
- **Dynamic scoring** based on `score_func_names` instead of hardcoded metrics
- **Dictionary-based returns** instead of fixed tuple returns

---

## Changes to `Diffusion` Class (`diffusion.py`)

### 1. **Initialization (`__init__`)**

#### Tokenizer Changes
- **Replaced**: `SMILES_SPE_Tokenizer` with GenMol's `get_tokenizer()` from `genmol.utils.utils_data`
- **Added**: Tokenizer attributes (`bos_index`, `eos_index`) extracted from SAFE tokenizer
- **Kept**: Optional tokenizer injection via constructor parameter

#### Backbone Model Changes
- **Replaced**: `roformer.Roformer` with `BertForMaskedLM` from HuggingFace Transformers
- **Model Configuration**: Uses `BertConfig.from_dict(dict(self.config.model))` with proper error handling
- **Freezing Logic**: Updated to work with BERT architecture:
  - Replaced `backbone.freeze_model()` and `backbone.unfreeze_n_layers()` (RoFormer-specific)
  - Implemented manual parameter freezing using `requires_grad = False/True`
  - Fixed layer access: `self.backbone.bert.encoder.layer[-i-1]` (BERT has nested structure)
  - Handles models with fewer than 8 layers using `min(unfreeze_layers, total_layers)`

#### Validation Analyzer
- **Replaced**: `PeptideAnalyzer` with `SafeAnalyzer` class
- **SafeAnalyzer**: New class that validates SAFE strings by decoding to SMILES
  - Uses `safe.decode(safe_str, canonical=True, ignore_errors=False)`
  - Returns `True` if decoding succeeds, `False` on exceptions

#### Training Scheme Support
- **Added**: Dual training scheme support via `config.training.scheme`
- **TR2-D2 Scheme** (default): Uses existing TR2-D2 diffusion logic
- **GenMol Scheme**: Instantiates MDLM (Masked Diffusion Language Model) components:
  - `AntitheticUniformTimeDistribution` or `UniformTimeDistribution` for time sampling
  - `DiscreteMaskedPrior` for prior distribution
  - `LogLinearExpNoiseTransform` for noise schedule
  - `MDLM` object that handles GenMol's training loop

#### EMA Support
- **Added**: Exponential Moving Average (EMA) support from GenMol
- **Conditional**: Only initialized if `config.training.ema > 0`
- **Integration**: EMA state saved/loaded in checkpoint hooks

### 2. **Forward Method**

#### Backbone Integration
- **Updated**: Now calls `self.backbone(input_ids=zt, attention_mask=attn_mask)` instead of RoFormer
- **Output Handling**: Extracts `.logits` from BERT output (dict vs tensor)
- **Device Management**: Ensures all tensors are on correct device
- **Autocast**: Uses `torch.cuda.amp.autocast(dtype=torch.float32)` for mixed precision

#### SUBS Parameterization
- **Unchanged**: Still applies SUBS (Substitution) parameterization:
  - Sets mask token probabilities to `-infinity`
  - Ensures unmasked tokens remain unchanged (log-prob = 0 for correct token, -inf for others)
- **Compatible**: Works correctly with BERT logits format

### 3. **Training Step**

#### Dual Training Schemes
- **TR2-D2 Scheme**: Uses existing `_compute_loss` method with TR2-D2's diffusion loss
- **GenMol Scheme**: New `_training_step_genmol` method that:
  - Samples time using `self.mdlm.sample_time()`
  - Corrupts input using `self.mdlm.forward_process()`
  - Computes loss using `self.mdlm.loss()` with optional global mean
  - Supports `global_mean_loss` config option

#### Training Step Routing
- **Method**: `training_step` now checks `self.training_scheme`
- **Routing**: Calls `_training_step_genmol` if scheme is "genmol", otherwise uses standard TR2-D2 path

### 4. **Sampling Methods**

#### `sample_finetuned_with_rnd`
- **Reward Model Integration**: 
  - Changed from array-based to dictionary-based reward model
  - Converts dict to array: `np.array([score_dict[name] for name in reward_model.score_func_names]).T`
  - Shape: `(num_sequences, num_objectives)` for MCTS compatibility
- **Validation**: Uses `self.analyzer.is_valid()` instead of `is_peptide()`
- **Edge Case Handling**: Properly handles empty valid sequences (returns zeros)

#### `sample_finetuned`
- **Return Format**: Changed from tuple of individual metrics to dictionary
  - **Old**: `(x_rollout, affinity, sol, hemo, nf, permeability, valid_fraction, df)`
  - **New**: `(x_rollout, scores_dict, valid_fraction, df)` where `scores_dict = {name: average_score}`
- **Dynamic Scoring**: 
  - Extracts `score_func_names` from `reward_model.score_func_names`
  - Computes averages: `scores = {name: np.mean(score_dict[name]) for name in score_func_names}`
- **DataFrame**:
  - Column: "SAFE Sequence" instead of "Peptide Sequence"
  - Dynamic columns based on `score_func_names`
  - Handles empty sequences correctly
- **Validation**: Uses `self.analyzer.is_valid()` for molecule validation

### 5. **Invalid Loss Computation**

#### `compute_invalid_loss`
- **Updated**: Changed from `is_peptide()` to `is_valid()` for molecule validation
- **Penalty Logic**: Unchanged - still penalizes invalid sequences based on token probabilities

### 6. **Checkpoint Hooks**

#### `on_load_checkpoint`
- **Added**: EMA state loading if EMA is enabled
- **Existing**: Fast-forward epoch/batch tracking

#### `on_save_checkpoint`
- **Added**: EMA state saving if EMA is enabled
- **Existing**: Checkpoint cleaning and sampler state

### 7. **Optimizer Step**

#### `optimizer_step`
- **Added**: EMA update if EMA is enabled
- **Existing**: Standard optimizer step and gradient clipping

---

## Changes to `MolScoringFunctions` Class (`scoring/scoring_functions.py`)

### Class Definition
- **Added**: `Dict` import from `typing`
- **Removed**: `prot_seqs` parameter (not needed for molecules)
- **Type Hints**: Added proper type hints for `forward` method: `List[str] -> Dict[str, np.ndarray]`

### Initialization
- **Removed**: `prot_seqs` parameter and related logic
- **Simplified**: Only requires `score_func_names` and optional `device`
- **Oracle Functions**: Maps `'qed'` and `'sa'` to `OracleQED()` and `OracleSA()`

### Forward Method
- **Return Type**: Returns dictionary `{score_func_name: np.array([scores...])}`
- **Shape**: Each value is array of shape `(num_sequences,)` - one score per sequence
- **Ordering**: Dictionary keys match `score_func_names` order

---

## Changes to `MCTS` Class (`peptide_mcts.py`)

### Imports
- **Removed**: `PeptideAnalyzer` import
- **Changed**: `ScoringFunctions` import to `MolScoringFunctions`

### Initialization
- **Analyzer**: Uses `policy_model.analyzer` (SafeAnalyzer) instead of `PeptideAnalyzer()`
- **Reward Function**: Uses `MolScoringFunctions(score_func_names, device=args.device)`
- **Dynamic Logging**: Replaced hardcoded logs (`affinity1_log`, `sol_log`, etc.) with `self.score_logs = {name: [] for name in score_func_names}`
- **Score Function Names**: Stores `self.score_func_names` as instance variable

### Reset Method
- **Dynamic Reset**: Replaces hardcoded log resets with `self.score_logs = {name: [] for name in self.score_func_names}`

### Expand Method
- **Validation**: Changed `self.analyzer.is_peptide(childSeq)` to `self.analyzer.is_valid(childSeq)`
- **Score Conversion**: 
  - Converts dict from `reward_model` to array: `np.array([score_dict[name] for name in self.score_func_names]).T`
  - Shape: `(num_sequences, num_objectives)`
- **Dynamic Logging**: 
  - Computes average scores: `average_scores = np.mean(score_vectors, axis=0)`
  - Logs dynamically: `for idx, name in enumerate(self.score_func_names): self.score_logs[name].append(average_scores[idx])`
- **Buffer Update**: Uses `validSequences` instead of `childSequences` for buffer operations
- **Comments**: Removed "FOR PEPTIDES ONLY" and "END OF FOR PEPTIDES ONLY" comments

---

## Changes to `finetune` Function (`finetune_peptides.py`)

### Function Signature
- **Unchanged**: Still accepts `prot_name` parameter (for backward compatibility, can be `None`)

### Initialization
- **Dynamic Logging**: Replaced hardcoded metric lists with `score_logs = {name: [] for name in reward_model.score_func_names}`
- **Score Function Names**: Extracted from `reward_model.score_func_names`

### Training Loop
- **Return Value Handling**: 
  - Changed from: `x_eval, affinity, sol, hemo, nf, permeability, valid_fraction = ...`
  - Changed to: `x_eval, scores_dict, valid_fraction = ...`
- **Dynamic Logging**: 
  - Iterates over `score_func_names`: `for name in score_func_names: score_logs[name].append(scores_dict[name])`
- **Dynamic Printing**: 
  - Builds score string dynamically: `score_str = " ".join([f"{name} {scores_dict[name]:.4f}" for name in score_func_names])`
- **Dynamic Wandb Logging**: 
  - Updates wandb dict: `wandb_dict.update({name: scores_dict[name] for name in score_func_names})`

### Logging and Plotting
- **Save Logs**: Updated `save_logs_to_file` call to pass `score_logs` dict
- **Dynamic Plotting**: 
  - Iterates over `score_func_names` to create plots: `for name in score_func_names: plot_data_with_distribution_seaborn(...)`
  - Uses dynamic labels: `label1=f"Average {name.upper()} Score"`
- **DataFrame**: Uses dynamic column names from `score_func_names`

### `save_logs_to_file` Function
- **Signature**: Changed from individual list parameters to `save_logs_to_file(valid_fraction_log, score_logs, output_path)`
- **DataFrame Creation**: 
  - Creates dict dynamically: `log_data = {"Iteration": ..., "Valid Fraction": valid_fraction_log, **{name: score_logs[name] for name in score_logs.keys()}}`
  - Removed hardcoded column names

---

## Changes to `finetune.py`

### Protein Sequences
- **Removed**: All protein sequence definitions (amhr, tfr, gfap, glp1, glast, ncam, cereblon, ligase, skp2)
- **Removed**: Logic for selecting protein based on `args.prot_seq`

### Run Name and Filename
- **Changed**: Uses `args.name` for filename instead of `prot_name`
- **Default**: `filename = args.name if args.name else "molecules"`
- **Run Name**: Based on `run_name_prefix = filename` instead of protein name

### Model Initialization
- **Unchanged**: Still loads from checkpoint using `Diffusion.load_from_checkpoint()`

### MCTS and Reward Model
- **MCTS**: Instantiated with `prot_seqs=None` instead of `prot_seqs=[prot]`
- **Reward Model**: `MolScoringFunctions` instantiated without `prot_seqs` parameter
- **Function Calls**: `finetune` called with `prot_name=None`

---

## Changes to `SafeAnalyzer` Class (`diffusion.py`)

### New Class
- **Purpose**: Validates SAFE strings by attempting to decode them to SMILES
- **Method**: `is_valid(safe_str: str) -> bool`
- **Implementation**: 
  - Tries to decode using `safe.decode(safe_str, canonical=True, ignore_errors=False)`
  - Catches `sf.DecoderError` and `ValueError` exceptions
  - Returns `True` if decoding succeeds, `False` otherwise

---

## New Files Created

### `scoring/functions/mol_functions/mol_oracles.py`
- **OracleQED**: Wraps TDC's QED oracle for SAFE sequences
  - Invalid score: `0.0`
  - Converts SAFE to SMILES, calls TDC oracle
- **OracleSA**: Wraps TDC's SA (Synthetic Accessibility) oracle for SAFE sequences
  - Invalid score: `10.0`
  - Converts SAFE to SMILES, calls TDC oracle
- **Base Class**: `_BaseOracle` provides common SAFE-to-SMILES conversion logic

---

## Key Architectural Changes

### 1. **Backbone Model**
- **From**: RoFormer (peptide-specific transformer)
- **To**: BERT (HuggingFace Transformers, used by GenMol)
- **Impact**: All forward passes now use BERT API

### 2. **Tokenizer**
- **From**: SMILES SPE Tokenizer (peptide-specific)
- **To**: SAFE tokenizer (from GenMol)
- **Impact**: All tokenization and decoding now uses SAFE format

### 3. **Validation**
- **From**: `PeptideAnalyzer.is_peptide()` (checks peptide bond validity)
- **To**: `SafeAnalyzer.is_valid()` (checks SAFE string decodability)
- **Impact**: All validation checks now verify molecule validity

### 4. **Scoring Functions**
- **From**: Peptide properties (binding affinity, solubility, hemolysis, etc.)
- **To**: Molecule properties (QED, SA) via TDC oracles
- **Impact**: All reward computation now uses molecule-specific metrics

### 5. **Return Formats**
- **From**: Fixed tuples with hardcoded metric names
- **To**: Dictionaries keyed by `score_func_names`
- **Impact**: Code is now generic and works with any set of scoring functions

### 6. **Training Schemes**
- **Added**: Support for GenMol's MDLM training scheme
- **Existing**: TR2-D2's original diffusion training scheme
- **Impact**: Can switch between training schemes via config

---

## Configuration Changes Needed

### Required Config Updates
1. **Model Configuration**: Must provide BERT config in `config.model` (not RoFormer config)
2. **Training Scheme**: Set `config.training.scheme = "genmol"` to use GenMol training, or `"tr2d2"` for original
3. **EMA**: Set `config.training.ema > 0` to enable EMA (optional)
4. **Score Functions**: Pass `score_func_names = ["qed", "sa"]` or other molecule oracles

### Optional Config Updates
1. **Global Mean Loss**: Set `config.training.global_mean_loss = True` for GenMol scheme (optional)
2. **Antithetic Sampling**: Set `config.training.antithetic_sampling = True` for GenMol scheme (optional)

---

## Breaking Changes

### API Changes
1. **`sample_finetuned` Return Value**: 
   - **Old**: `(x_rollout, affinity, sol, hemo, nf, permeability, valid_fraction, df)`
   - **New**: `(x_rollout, scores_dict, valid_fraction, df)`
   - **Migration**: Access scores via `scores_dict["qed"]`, `scores_dict["sa"]`, etc.

2. **`MolScoringFunctions` Return Value**:
   - **Old**: N/A (new class)
   - **New**: Dictionary `{score_func_name: np.array([scores...])}`
   - **Migration**: Access scores via dictionary keys

3. **`MCTS` Logging**:
   - **Old**: `mcts.affinity1_log`, `mcts.sol_log`, etc.
   - **New**: `mcts.score_logs["qed"]`, `mcts.score_logs["sa"]`, etc.
   - **Migration**: Access logs via dictionary keys

### Removed Functionality
1. **Peptide-Specific Scoring**: Binding affinity, solubility, hemolysis, nonfouling, permeability
2. **Protein Sequences**: No longer needed for molecule generation
3. **Peptide Validation**: `is_peptide()` method no longer used

---

## Testing Recommendations

1. **Validate Tokenizer**: Test that SAFE tokenizer correctly encodes/decodes molecules
2. **Validate Backbone**: Test that BERT backbone produces correct logits shape
3. **Validate Scoring**: Test that QED and SA oracles return correct scores
4. **Validate Sampling**: Test that `sample_finetuned` returns correct dictionary format
5. **Validate MCTS**: Test that MCTS works with dictionary-based rewards
6. **Validate Training**: Test both TR2-D2 and GenMol training schemes

---

## Files Modified

1. `diffusion.py` - Core Diffusion class updated for molecules
2. `scoring/scoring_functions.py` - MolScoringFunctions class fixed
3. `peptide_mcts.py` - MCTS class updated for molecules
4. `finetune_peptides.py` - Finetune function updated for dynamic scoring
5. `finetune.py` - Main script updated to remove protein dependencies

## Files Created

1. `scoring/functions/mol_functions/mol_oracles.py` - TDC oracle wrappers for molecules

---

## Summary of Diffusion Class Changes

The `Diffusion` class underwent significant changes to support molecule generation:

1. **Backbone**: Switched from RoFormer to BERT (GenMol's backbone)
2. **Tokenizer**: Switched from SMILES/peptide tokenizer to SAFE tokenizer
3. **Validation**: Switched from peptide validation to SAFE validation
4. **Training**: Added support for GenMol's MDLM training scheme alongside TR2-D2's scheme
5. **Sampling**: Updated to return dictionary-based scores instead of fixed tuples
6. **EMA**: Added Exponential Moving Average support from GenMol
7. **Forward Pass**: Updated to use BERT API instead of RoFormer API
8. **SUBS Parameterization**: Maintained compatibility with new backbone

All changes maintain backward compatibility where possible, with the main breaking change being the return format of `sample_finetuned` (now returns dictionary instead of tuple).

