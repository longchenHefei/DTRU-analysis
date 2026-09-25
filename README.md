# Correlated Decryption Errors in DTRU

Reproduction package for the paper

> *Correlated Decryption Errors in DTRU: Revisiting Failure Rates and Failure Boosting for Lattice KEMs with Repetition-Coded Blocks*

We recompute the decryption failure rates of all seven DTRU parameter sets with the real Algorithm-2 failure event and the correct block covariance, then evaluate classical/quantum failure-boosting costs under the model of D'Anvers et al. (ePrint 2018/1089, 2019/1399, 2021/193).

## Contents

| Path | Description |
|------|-------------|
| `paper/main.tex`, `paper/main.pdf` | Draft paper (LaTeX + PDF) |
| `scripts/` | Python reproduction scripts |
| `results/` | JSON outputs from the scripts |
| `reports/` | Chinese technical reports (full analysis + DTRU-2048 note) |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `mpmath`

```bash
pip install numpy scipy mpmath
```

## Reproduce

From `scripts/` (paths in the scripts assume they live next to each other):

```bash
cd scripts

# Three-ring toy validation (real Enc/Dec vs models)
python3 dfr_all_sets.py toy

# All 7 parameter sets (40 samples by default; set DTRU_SAMPLES=200 for the paper numbers)
DTRU_SAMPLES=200 python3 dfr_all_sets.py full
python3 dfr_all_sets.py merge

# Half-space predicate vs Algorithm 2; toy spectral recovery; n=2048 directions
python3 recover.py check
python3 recover.py toy
python3 recover.py full

# Optional: ball-spectrum / CGF checks
python3 failure_boost.py
python3 failure_boost.py tail
python3 ml_decoder_check.py toy
python3 ml_decoder_check.py full
python3 exact_cgf_check.py full
```

Headline numbers for DTRU-2048 (200 ciphertexts) match those in `results/dfr_all_sets.json` and Section 6 of the paper: median failure rate about \(2^{-123}\), sample mean about \(2^{-111}\).

## License

Code and data: MIT. The DTRU specification itself is not redistributed here; cite the NGCC submission.

## Disclaimer

This is cryptanalysis of a submitted KEM. No end-to-end key recovery on the full parameters is claimed. Numbers are for academic evaluation.