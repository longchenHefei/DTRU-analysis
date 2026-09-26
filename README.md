# Decryption Failures in NGCC Lattice KEMs

Reproduction package for the paper

> *Decryption Failures in NGCC Lattice KEMs: Correlated Blocks, Omitted Compression Noise, and Failure Boosting under a Query Cap*

We recompute decryption failure rates and failure-boosting costs for Class I / Class II first-round NGCC lattice KEMs:

- **Part I — DTRU.** Real Algorithm-2 failure region and block covariance for all seven sets.
- **Part II — Cheetah.** Exact one-dimensional tilted-FFT tails including public-key compression noise.
- **Part III — Rudraksh2 / Scabbard.** Exact pair tails for B2-Minal decoding under a \(2^{64}\) / \(2^{80}\) query cap.

No end-to-end key recovery is claimed.

## Contents

| Path | Description |
|------|-------------|
| `paper/main.tex`, `paper/main.pdf` | Draft paper (LaTeX + PDF) |
| `scripts/` | Python reproduction scripts (DTRU + Cheetah + Minal) |
| `results/` | JSON outputs cited in the paper |
| `reports/` | Chinese technical reports |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `mpmath`

```bash
pip install numpy scipy mpmath
```

## Reproduce

From `scripts/`:

```bash
cd scripts

# --- DTRU (Part I) ---
python3 dfr_all_sets.py toy
DTRU_SAMPLES=200 python3 dfr_all_sets.py full
python3 dfr_all_sets.py merge
python3 recover.py check
python3 recover.py toy
python3 recover.py full

# --- Cheetah + Minal (Parts II–III) ---
# Writes exact_tails.json next to the script; paper numbers live in ../results/
python3 exact_tails.py
python3 cheetah_toy.py

# Optional screens (Gaussian / Chernoff, not the paper's exact figures)
python3 boost_screen.py
python3 query80_cost.py
```

Headline numbers:

- DTRU-2048 (200 ciphertexts): sample mean about \(2^{-111}\) — `results/dfr_all_sets.json`
- Cheetah128 / 256 exact DFR \(2^{-79.1}\) / \(2^{-51.8}\) — `results/exact_tails.json`, `results/cheetah_toy.json`
- Rudraksh2-128-I / Scabbard-128 exact DFR \(2^{-103.9}\) / \(2^{-102.1}\) — `results/exact_tails.json`

## License

Code and data: MIT. The NGCC specifications themselves are not redistributed here; cite the submissions (DTRU, Cheetah, Rudraksh2, MORNING-Scabbard).

## Disclaimer

This is cryptanalysis of submitted KEMs. No end-to-end key recovery on the full parameters is claimed. Numbers are for academic evaluation.
