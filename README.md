# Enhancing Electric Grid Energy Flexibility using AI

**Physics-Guided Graph Learning and Forecast-Driven Storage Scheduling for Flexible and Voltage-Secure Power Systems**

- **Candidate:** Engr. Saad Hayat (`336-GUELE-22`)
- **Supervisor:** Engr. Dr. Aamir Nawaz, Assistant Professor, Department of Electrical Engineering
- **Institution:** Department of Electrical Engineering, Faculty of Engineering & Technology, Gomal University, Dera Ismail Khan, KP, Pakistan
- **Permanent Zenodo Archive:** [DOI: 10.5281/zenodo.23440240](https://doi.org/10.5281/zenodo.23440240)
- **Solver Stack:** Python `3.10.12` + `pandapower v2.13.1` (cross-validated against **MATLAB R2025a + MATPOWER v8.1**)

---

## Canonical Repository Structure (Aligned with Dissertation Appendix D, Table D.2)

```text
Enhancing-Electric-Grid-Energy-Flexibility-AI/
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml                         # Conda spec (Appendix D, Listing D.1)
├── requirements.txt                        # Pip spec
├── run_all.py                              # Master pipeline runner (--fast or full 10-seed)
├── utils/
│   ├── __init__.py
│   ├── seed_utils.py                       # 10-seed deterministic protocol (Appendix D, Listing D.2)
│   └── download_datasets.py                # Strictly read-only CSV downloader (no permission writes)
├── pandapower/
│   ├── sim1_layer1_forecasting.py          # Layer 1: 12-Model SOTA Benchmark, Optuna, 10-Seed Stats, TreeSHAP
│   ├── sim2_layer2_bess_rainflow.py        # Layer 2: PyTorch LSTM BESS, Outer-Loop SoC, ASTM E1049-85 Rainflow, IEEE 33-Bus, LCOS/NPV
│   ├── sim3_layer3_pigat_fvsi.py           # Layer 3a: Leak-Free LHS N-1 Graph Dataset, PI-GAT/TGCN/GCN/GAT + MC Dropout + Ablation
│   └── sim4_closed_loop_acopf.py           # Layer 3b: MERRA-2 NOCT/Hellmann RES, Closed-Loop AC OPF, 0-60% Hosting Sweep, 14-300 Bus Scaling
├── matpower/
│   ├── cross_validate_ieee14_matpower.m    # MATLAB R2025a + MATPOWER v8.1 base-case solver parity
│   └── closed_loop_acopf_matpower.m        # MATLAB R2025a + MATPOWER v8.1 IEEE 14-bus AC OPF verification
├── data/
│   └── DATA_CARDS.md                       # Complete Gebru et al. Data Cards for all 8 CSVs + Ember + Grids
└── checkpoints/
    └── README.md                           # Saved trained model weights (.pt / .json)
```

---

## Quickstart & Independent Reproduction

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate grid-flexibility-ai
# Or via pip:
pip install -r requirements.txt
```

### 2. Download Verified Benchmark Datasets (Read-Only)
```bash
python utils/download_datasets.py --data-dir data
```

### 3. Execute the Full Three-Layer Pipeline
```bash
# Quick end-to-end verification (2 seeds, reduced epochs/LHS states)
python run_all.py --fast

# Full dissertation 10-seed reproduction (GPU recommended for PyTorch / PyG)
python run_all.py
```

Or run each layer independently:
```bash
python pandapower/sim1_layer1_forecasting.py --data-dir data --out-dir outputs
python pandapower/sim2_layer2_bess_rainflow.py --data-dir data --out-dir outputs
python pandapower/sim3_layer3_pigat_fvsi.py --out-dir outputs
python pandapower/sim4_closed_loop_acopf.py --data-dir data --out-dir outputs
```

---

## Master Reproducibility Mapping (Appendix D, Table D.2)

| Script Path | Datasets & Grids | Dissertation Outputs Generated |
| :--- | :--- | :--- |
| `pandapower/sim1_layer1_forecasting.py` | `D1a/b`, `D2a/b`, `D3a/b`, `D5` | Layer 1 12-model SOTA benchmark (`Physics-XGBoost`, `PatchTST`, `TimesNet`, `FEDformer`, `Autoformer`, `Informer`, `Transformer`, `LSTM`, `GRU`, `1D-CNN`, `Prophet`, `SARIMA`), Optuna TPE tuning, 10-seed Diebold-Mariano / Wilcoxon / Friedman / Block-Bootstrap tests, multi-climate & noise/surge stress tests, and exact `TreeSHAP` kW attributions (Tables 5.1–5.7, Figs. 5.1–5.5). |
| `pandapower/sim2_layer2_bess_rainflow.py` | `D1a` & IEEE 33-Bus (`pn.case33bw()`) | Layer 2 PyTorch `LSTM` predictive BMS + outer-loop `SoC ∈ [10%, 90%]` recursion + ASTM E1049-85 3-point Rainflow cycle counting (`FEC`, `D_cycle`, `L_est`) + 10-year `LCOS`/`NPV`/`IRR`/`SPP` ledger + IEEE 33-bus voltage/loss audit (Table 3.5, Table 5.8, Fig. 5.6). |
| `pandapower/sim3_layer3_pigat_fvsi.py` | `D6` (IEEE 14, 57, 118-Bus LHS `N-1`) | Programmatic leak-free pre-dispatch assertion, 5,000-state LHS + `N-1` graph dataset, real `PI-GAT` (impedance edge attention + physics loss + 50-pass MC Dropout), `TGCN`, `GCN`, `GAT`, `GraphSAGE`, `ChebNet`, classical ML baselines, 11-step reactive load ramp, and ablation study (Tables 5.9–5.13, Figs. 5.7–5.8). |
| `pandapower/sim4_closed_loop_acopf.py` | `D2a`, `D4a/b` & IEEE 14/18/33/57/118/300-Bus | MERRA-2 NOCT solar + Hellmann wind conversion, IEEE Std 1547-2018 `net.sgen` + `net.storage` closed-loop AC OPF (`pp.runopp`) with fallback relaxation, 0%–60% hosting capacity sweep, 4-scenario audit, and 14–300 bus latency scaling benchmark (Tables 5.14–5.16, Figs. 5.9–5.11). |

---

## Associated Peer-Reviewed Publications
1. S. Hayat, A. Nawaz, *et al.*, "Enhancing Power System Flexibility Using AI-Based Forecasting Techniques: A Comparative Study," *Battery Energy* (Wiley), vol. 5, no. 1, e70079, 2026. DOI: `10.1002/bte2.70079`
2. S. Hayat, A. Nawaz, *et al.*, "Dynamic Load Modelling and Optimised Battery Management Using LSTM for Enhanced Power System Flexibility," *The Journal of Engineering* (IET/Wiley), vol. 2025, no. 1, e70126, 2025. DOI: `10.1049/tje2.70126`
3. S. Hayat, A. Nawaz, *et al.*, "Optimization of Power System Flexibility Through AI-Driven Dynamic Load Management and Renewable Integration," *Battery Energy* (Wiley), vol. 4, e70026, 2025. DOI: `10.1002/bte2.20250009`
4. S. Hayat, A. Nawaz, *et al.*, "Predicting Voltage Stability Using Graph Convolutional Networks," *Journal of Engineering* (Wiley), vol. 2026, 8588270, 2026. DOI: `10.1155/je/8588270`
