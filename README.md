# Forgetting as Probe: The Geometry of Catastrophic Forgetting

**The Geometry of Catastrophic Forgetting: Scale and Batch Size Reveal a Two-Dimensional Behavioral Shift in Language Models**

*Yi Zhang — Independent Researcher*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19446077.svg)](https://doi.org/10.5281/zenodo.19446077)

---

## Abstract

Catastrophic forgetting — the degradation of previously learned knowledge during fine-tuning on new domains — is universally treated as a problem to minimize. We propose treating its magnitude as a structured signal instead. Through controlled experiments across model scales (0.5B to 14B parameters) and batch sizes (1 to 16), we discover that forgetting magnitude forms a two-dimensional surface with non-trivial geometry: monotonically decreasing with scale, and exhibiting a V-shaped curve with respect to batch size at larger scales. A qualitative behavioral shift occurs between 3B and 7B parameters, where the relationship between batch size and forgetting changes character. This shift coincides with the scale range where emergent abilities have been empirically observed in prior work, suggesting a possible connection worth further investigation. Our findings imply that optimal batch size is a predictable function of model scale, potentially replacing expensive grid search with a principled selection rule.

---

## Key Results

### Scale Curve (batch=4, fixed)

| Scale | PPL before | PPL after | Forgetting |
|-------|-----------|-----------|------------|
| 0.5B  | 2.342     | 3.572     | +52.51%    |
| 1.5B  | 2.411     | 3.096     | +28.41%    |
| 3B    | 2.123     | 2.536     | +19.45%    |
| 7B    | 1.678     | 1.910     | +13.79%    |
| 14B   | 1.825     | 1.916     | +4.98%     |

### Full 2D Surface (math → literature, 200 steps)

|       | bs=1    | bs=2    | bs=4    | bs=8    | bs=16   |
|-------|---------|---------|---------|---------|---------|
| 0.5B  | +11.65% | +18.62% | +52.56% | +140.04%| +282.78%|
| 1.5B  | +4.15%  | +11.17% | +36.41% | +61.85% | +134.52%|
| 3B    | +2.23%  | +6.29%  | +13.69% | +14.41% | +44.90% |
| 7B    | +3.73%  | +6.88%  | +14.24% | +10.44% | +8.79%  |
| 14B   | +3.02%  | +5.67%  | +4.98%  | +5.07%  | +1.50%  |

The V-shape emerges at 7B (peak at bs=4) and the peak shifts left to bs=2 at 14B. Smaller models show monotonic increase — the V-shape is absent below the 3B–7B behavioral shift.

---

## Reproducing the Experiments

### Requirements

```bash
pip install transformers datasets torch accelerate bitsandbytes
```

### Core Experiments

**1. Initial 3-domain probe (0.5B, math/code/literature)**
```bash
python forgetting_probe.py
```

**2. Controlled scale sweep (batch=4, 0.5B–7B)**
```bash
python forgetting_controlled.py        # single GPU
python server_run_all_controlled.py    # multi-GPU server
```

**3. Batch size sweep (7B, bs=1–16)**
```bash
python batchsweep_7b.py               # single GPU
python server_batchsweep_7b.py        # multi-GPU server
```

**4. Full 2D surface (0.5B–3B, all batch sizes)**
```bash
python batchscale_2d.py               # single GPU
python server_batchscale_2d.py        # multi-GPU server
```

**5. 14B batch sweep**
```bash
python forgetting_14b.py              # single GPU
python server_forgetting_14b.py       # multi-GPU server
```

### Hardware Used

- Core experiments (0.5B–7B): 4× RTX 5090 (consumer-grade)
- Extended experiments (14B): 4× RTX Pro 6000
- Total compute cost: ~$13 USD (~90 RMB)

---

## Results Files

| File | Contents |
|------|----------|
| `results.json` | Initial 3-domain probe results |
| `results_controlled.json` | Token-controlled replication |
| `results_0p5B_v2.json` | 0.5B controlled scale sweep |
| `results_1p5B_v2.json` | 1.5B controlled scale sweep |
| `results_3B_v2.json` | 3B controlled scale sweep |
| `results_7B_v2.json` | 7B controlled scale sweep |
| `results_7b_batchsweep.json` | 7B batch size sweep (bs=1–16) |
| `results_batchscale_2d.json` | Full 2D surface (0.5B–3B) |
| `results_14b_batchsweep.json` | 14B batch size sweep |

---

## Protocol

All controlled experiments use identical hyperparameters:

- **Model family:** Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B)
- **Domain pair:** Math (lighteval/MATH) → Literature (wikitext-103-raw-v1)
- **Token budget:** 50,000 tokens per domain (exact)
- **Steps:** 200 global steps (fixed compute budget)
- **Learning rate:** 2e-5
- **Max length:** 256
- **Seed:** 42
- **Optimizer:** AdamW fp32
- **Gradient checkpointing:** None

Note: Fixing steps rather than epochs means epoch count varies with batch size by construction (batch=1: ~1.1 epochs; batch=16: ~20 epochs). This reflects standard practice in LLM training where compute budgets are measured in gradient updates.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{zhang2026forgetting,
  title={The Geometry of Catastrophic Forgetting: Scale and Batch Size Reveal a Two-Dimensional Behavioral Shift in Language Models},
  author={Zhang, Yi},
  year={2026},
  doi={10.5281/zenodo.19446077},
  url={https://doi.org/10.5281/zenodo.19446077}
}
```

---

## License

**Code:** MIT License — see [LICENSE](LICENSE)

**Paper:** CC BY 4.0 — the accompanying paper on Zenodo is licensed under Creative Commons Attribution 4.0 International.
