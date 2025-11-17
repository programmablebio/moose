import numpy as np
import safe as sf
from tdc import Oracle, Evaluator


class _BaseOracle:
    """
    SAFE-to-SMILES adapter for TDC oracle scorers.
    Base class for all TDC oracles.
    """

    oracle_name: str = ""
    invalid_score: float = 0.0

    def __init__(self):
        if not self.oracle_name:
            raise ValueError("oracle_name must be set on subclasses.")
        self._oracle = Oracle(self.oracle_name)

    @staticmethod
    def _safe_to_smiles(sequence: str):
        """
        Decode SAFE string to SMILES, returning None on failure.

        Args:
            sequence: SAFE string to decode.

        Returns:
            SMILES string if successful, None otherwise.
        """

        try:
            smiles = sf.decode(sequence, canonical=True, ignore_errors=False)
        # except (sf.DecoderError, ValueError):
        except (sf.SAFEDecodeError, ValueError):
            return None
        return smiles

    def __call__(self, input_seqs):
        scores = np.full(len(input_seqs), self.invalid_score, dtype=np.float32)

        smiles_batch = []
        valid_indices = []
        for idx, sequence in enumerate(input_seqs):
            smiles = self._safe_to_smiles(sequence)
            if smiles is None:
                continue
            smiles_batch.append(smiles)
            valid_indices.append(idx)

        if smiles_batch:
            oracle_scores = self._oracle(smiles_batch)
            for idx, value in zip(valid_indices, oracle_scores):
                scores[idx] = np.float32(value)

        return scores


class OracleQED(_BaseOracle):
    """Wraps TDC's QED oracle for SAFE sequences."""

    oracle_name = "qed"
    invalid_score = 0.0


class OracleSA(_BaseOracle):
    """
    Wraps TDC's synthetic accessibility oracle for SAFE sequences.
    
    SA scores are normalized to [0, 1] where higher is better:
    1. Raw SA scores are clipped to max 6.0
    2. Normalized as: (6 - SA_score) / 5

    Thus SA = 1 (easy to synthesize) becomes 1.0, SA = 6 (hard) becomes 0.0
    """

    oracle_name = "sa"
    invalid_score = 6.0

    def __call__(self, input_seqs):

        raw_scores = super().__call__(input_seqs)
        
        ## 6 is not max theoretically but it is clipped in PMO
        clipped_scores = np.clip(raw_scores, a_min=1, a_max=6.0)
        normalized_scores = (6.0 - clipped_scores) / 5.0
        
        return normalized_scores.astype(np.float32)
