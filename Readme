# Enhancing Electric Grid Energy Flexibility using AI

Official PhD Dissertation Code, Simulation Suite (`pandapower` & `MATPOWER`), and Standardized Dataset Data Cards for **Engr. Saad Hayat** (`336-GUELE-22`), supervised by **Engr. Dr. Aamir Nawaz**, Department of Electrical Engineering, Faculty of Engineering and Technology, **Gomal University, Dera Ismail Khan, Pakistan**.

* **Final Tentative Dissertation Title:** *Enhancing Electric Grid Energy Flexibility using AI*
* **Alternative Technical Subtitle:** *Physics-Guided Graph Learning and Forecast-Driven Storage Scheduling for Flexible and Voltage-Secure Power Systems*
* **Candidate:** Engr. Saad Hayat (`336-GUELE-22`)
* **Supervisor:** Engr. Dr. Aamir Nawaz
* **External Examiner / Reviewer Guidance Incorporated:** Prof. Dr. Laiq Khan (COMSATS University Islamabad, Attock Campus)

## 1. Unified Three-Layer Architecture & End-to-End Data Chain

To address power grid flexibility end-to-end rather than as isolated tasks, this repository implements a vertically integrated three-layer pipeline connecting kilowatt-scale building forecasts to megawatt-scale distribution and transmission operations:

| Layer | Module Name | Inputs (with Units) | Core Algorithms | Outputs (with Units) | Benchmark Systems |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | **Demand & Renewable Forecasting** | Historical load ($\text{kW}$), calendar encodings, and TMY3 weather ($\text{°C}, \text{\%}, \text{W/m}^2, \text{m/s}$) | `XGBoost`, `Transformer`, `GRU`, `LSTM`, `1D-CNN`, `Prophet`, `SARIMA`, `Informer`, `Autoformer`, `FEDformer`, `PatchTST`, `TimesNet` + `TreeSHAP` / Attention XAI | 24-hour building load forecast ($\hat{P}_{\text{load},t}$ in $\text{kW}$), nodal load ($\text{MW}/\text{MVAr}$), and available PV/Wind ($\hat{P}_{\text{avail},t}^{\text{RES}}$ in $\text{MW}$) | Phoenix Zone 2B, Alabama Zone 3A, & 7-Country Ember Grids |
| **Layer 2** | **Forecast-Driven Storage Scheduling** | Aggregated feeder/bus load forecast ($\text{kW}/\text{MW}$), tariff signals ($\$/\text{kWh}$), and previous $\text{SoC}(t-1)$ | LSTM Receding-Horizon Controller + Outer-Loop $\text{SoC}$ Recursion ($\text{SoC} \in [10\%, 90\%]$, $\eta_{\text{ch}}=\eta_{\text{dis}}=0.95$) + Rainflow Full Equivalent Cycle (`FEC`) degradation model | Signed BESS active/reactive set-points ($P_{\text{BESS},t}, Q_{\text{BESS},t}$ in $\text{MW}/\text{MVAr}$) and updated $\text{SoC}(t)$ | Commercial BESS ($1\text{ MWh}/50\text{ kW}$), IEEE 33-Bus ($1\text{ MWh}/250\text{ kW}$), & Substation BESS ($20\text{ MWh}/10\text{ MW}$) |
| **Layer 3** | **Physics-Guided Graph Security Screening & AC-OPF Execution** | Candidate pre-dispatch bus injections ($P_i, Q_i, V_i$ set-points) and network topology ($R_{ij}, X_{ij}, B_{ij}$, line status) | **PI-GAT** (Impedance-Guided Graph Attention), `TGCN` (`GCN+LSTM`), `GCN`, `GAT`, `GraphSAGE`, `ChebNet` + `pandapower` (`runopp`) & `MATPOWER` (`runpf`/`runopf`) | Pre-dispatch branch $\text{FVSI} < 0.80$ pass/fail screen, optimal generator/BESS dispatch, `sgen` curtailment ($P_{\text{curt},t}$), and bus voltages ($V_i \in [0.95, 1.06]\text{ p.u.}$) | IEEE 14-Bus, IEEE 33-Bus, IEEE 57-Bus, IEEE 118-Bus, & IEEE 300-Bus (inference scaling) |

## 2. Repository Directory Structure

```text
Saad-PhD-Thesis/
├── README.md                                  # Master PhD thesis overview, architecture, and Data Cards
├── LICENSE                                    # MIT Open-Source License
├── environment.yml                            # Conda environment specification (exact pinned versions)
├── requirements.txt                           # pip package dependencies
├── data/
│   ├── DATA_CARDS.md                          # Standardized Data Cards for Datasets 1–4 (Appendix C)
│   ├── predicted_loads.csv                    # Transformer-predicted bus load scaling factors
│   ├── phoenix_zone2b_commercial_sample.csv   # Dataset 1: NREL/OpenEI Phoenix Zone 2B hourly load & weather
│   ├── alabama_zone3a_restaurant_sample.csv   # Dataset 2: OpenEI Alabama G0100030 restaurant load & weather
│   └── ieee14_5000_contingency_samples.csv    # Dataset 4: Pre-dispatch injection-to-FVSI benchmark samples
├── matpower/
│   ├── fourthpaper.m                          # Unified MATPOWER 8.1 single-window tabbed dashboard (Figs 6–10, Tables 4–8)
│   └── cross_validate_ieee14_matpower.m       # Baseline IEEE 14-bus MATPOWER vs. pandapower verification script
└── pandapower/
    ├── layer1_forecasting_benchmarks.py       # Layer 1: XGBoost, Transformer, GRU, LSTM & SOTA benchmarks + TreeSHAP
    ├── layer2_bess_rainflow_scheduler.py      # Layer 2: LSTM BESS scheduling, outer-loop SoC recursion & Rainflow FEC
    ├── layer3_pigat_tgcn_fvsi_screening.py    # Layer 3a: PI-GAT & TGCN pre-dispatch FVSI security screening + GNNExplainer
    └── layer3_pandapower_closed_loop_opf.py   # Layer 3b: Closed-loop pandapower AC-OPF across IEEE 14, 33, 57 & 118-bus
```

## 3. Standardized Dataset Data Cards (Reviewer Comment 10 & Appendix C)

To ensure 100% reproducibility and eliminate leakage, all time-series datasets use strictly chronological **70% Training / 15% Validation / 15% Testing** splits. Feature scalers (`MinMaxScaler` and `StandardScaler`) are fitted **strictly on the training partition only** and applied to validation and test partitions without refitting.

### Data Card 1: Phoenix Commercial Building Dataset (ASHRAE Climate Zone 2B — Hot-Dry)

* **Source & DOI:** NREL / OpenEI Commercial and Residential Hourly Load Profiles for TMY3 Locations in the United States ([OpenEI Submission 153](https://www.google.com/search?q=https%3A%2F%2Fdata.openei.org%2Fsubmissions%2F153), DOI: `10.25984/1788456`).
* **Data Type & Resolution:** EnergyPlus-simulated DOE Commercial Prototype Building (Outpatient Healthcare / Commercial Facility in Phoenix, Arizona, USA) paired with TMY3 meteorological variables; hourly resolution ($\Delta t = 1\text{ h}$).
* **Variables:** Total facility electricity ($\text{kW}$), cooling ($\text{kW}$), heating ($\text{kW}$), interior lighting ($\text{kW}$), interior equipment ($\text{kW}$), fans ($\text{kW}$), gas heating ($\text{kW}$), dry-bulb temperature ($\text{°C}$), relative humidity ($\text{\%}$), and solar irradiance ($\text{W/m}^2$).
* **Ordered Preprocessing Steps:**
  1. Timestamp standardization to UTC-7 and removal of duplicate timestamps.
  2. Linear interpolation for missing gaps $\le 6\text{ h}$; removal of contiguous gaps $> 6\text{ h}$.
  3. Outlier detection via $Z$-score filtering ($|Z| > 3$) and `IsolationForest` ($\text{contamination} = 0.01$).
  4. Cyclical calendar encoding ($\sin/\cos$ of `hour`, `day_of_week`, `month`, `is_weekend`) and 24-hour autoregressive lag stack ($t-1, \dots, t-24$).
  5. Chronological 70% / 15% / 15% split; `MinMaxScaler` ($[0, 1]$) fitted on the 70% training split only.

### Data Card 2: Alabama Full-Service Restaurant Dataset (ASHRAE Climate Zone 3A — Warm-Humid)

* **Source & DOI:** NREL / OpenEI End-Use Load Profiles for the U.S. Building Stock (Full-Service Restaurant, Alabama State, Geographic Field Code `G0100030`, year 2018; DOI: `10.25984/1788456`).
* **Data Type & Resolution:** Hourly load and meteorological records ($8,760\text{ hours/year}$).
* **Electrical Load Features (Table 1):** `Electricity: Facility [kW]`, `Electricity: HVAC [kW]`, `Electricity: Interior equipment [kW]`, `Electricity: Interior lights [kW]`, `Electricity: Exterior lights [kW]`, and `Electricity: Refrigeration [kW]`.
* **Weather Features (Table 2):** `Temperature [C]`, `Humidity [%]`, `Wind speed [m/s]`, `Dew point [C]`, `Solar radiation [W/m2]`, `Precipitation [mm]`, `Pressure [Pa]`, and `Cloud cover [%]`.
* **Engineered Features:** `hour`, `day_of_week`, `month`, `is_weekend`, `lagged_load`, and `lagged_temp`.

### Data Card 3: Multi-Country Electricity Demand Dataset (Cross-Climate Generalization)

* **Source:** [Ember Global Electricity Data](https://www.google.com/search?q=https%3A%2F%2Fember-energy.org%2Fdata%2Fyearly-electricity-data%2F) spanning 7 countries across cold European and tropical climate zones.
* **Purpose:** Evaluates cross-climate generalization (Case Study 1) of the Layer 1 forecasting models (`XGBoost`, `GRU`, `LSTM`, `Transformer`, `PatchTST`, `TimesNet`, `Informer`, `Autoformer`, `FEDformer`).

### Data Card 4: Pre-Dispatch Voltage-Stability Graph Dataset (`pandapower` / `MATPOWER`)

* **Systems Covered:** IEEE 14-bus ($20$ branches), IEEE 57-bus ($80$ branches), and IEEE 118-bus ($186$ branches), plus the 11-step reactive load ramp benchmark ($10\text{ MVAr}$ steps to $\text{FVSI} \ge 1.0$).
* **Sample Count & Generation:** $5,000$ converged operating points per system generated via uniform random $\pm 30\%$ active/reactive load and renewable variations across intact and standardized $N-1$ contingency topologies (`Line 2–4`, `Line 4–9`, `Line 12–13`).
* **Zero Label-Leakage Input Specification (Reviewer Comment 18 / C-6):**
  * **Node Inputs ($X_V$):** Pre-dispatch bus active demand $P_{d,i}$, reactive demand $Q_{d,i}$, scheduled active generation $P_{g,i}$, and voltage set-point $V_{g,i}$.
  * **Edge Inputs ($X_E$):** Physical branch resistance $R_{ij}$, reactance $X_{ij}$, susceptance $B_{ij}$, reactance-to-resistance ratio $X_{ij}/R_{ij}$, and binary line status $u_{ij} \in \{0, 1\}$.
  * **Excluded Variables:** Post-solution branch power flows ($PF, QF, PT, QT$) are **strictly excluded** from the GNN inputs so that the model performs genuine pre-dispatch security screening rather than algebraic curve-fitting.
  * **Target Label ($Y_E$):** Ground-truth branch $\text{FVSI}_{ij} = \frac{4 Z_{ij}^2 Q_j}{V_i^2 X_{ij}}$ (with operational security threshold $\text{FVSI} < 0.80$).

## 4. Why Both `pandapower` and `MATPOWER` Are Included (Reviewer Comment 22 & M-20)

The initial simulations in our published *Battery Energy* (2025) study were developed in **MATLAB / MATPOWER**, whereas the unified three-layer closed-loop pipeline in the revised dissertation is implemented in **Python / `pandapower`**. Both toolchains are provided in this repository and cross-validated on the IEEE 14-bus base case:

* **Direct Python Integration:** `pandapower` couples directly with `PyTorch`, `PyTorch Geometric`, `XGBoost`, and `Optuna` in memory without slow CSV/MAT file handoffs.
* **Native Storage & Renewable Elements:** `pandapower` provides dedicated `net.storage` elements (supporting signed active/reactive power with outer-loop $\text{SoC}$ recursion) and `net.sgen` static generators (supporting $\text{IEEE Std 1547-2018}$ reactive power bounds $Q_{\text{RES}} \le 0.44 P_{\text{rated}}$ and active power curtailment $0 \le P_{\text{curt}}(t) \le P_{\text{avail}}(t)$).
* **Cross-Validation Agreement:** Running `matpower/cross_validate_ieee14_matpower.m` and `pandapower/layer3_pandapower_closed_loop_opf.py` confirms identical IEEE 14-bus base-case bus voltages ($< 10^{-4}\text{ p.u.}$ difference) and active power losses ($13.39\text{ MW/h}$, or $5.17\%$ of the $259.0\text{ MW}$ system load).

## 5. Hardware & Software Reproducibility Specification (Reviewer Comment 15)

### Standardized Hardware Environment

* **Local Workstation (Preprocessing, `pandapower` / `MATPOWER` Simulations & CPU Latency):** HP EliteBook 1040 G8 (Intel Core i7-1185G7 @ 3.00 GHz, 32 GB DDR4 RAM, Windows 11 64-bit).
* **Cloud GPU Accelerator (Deep/Graph Model Training & GPU Latency):** Google Colab Pro (NVIDIA Tesla T4 GPU, 16 GB GDDR6 VRAM).
* **Deterministic Random Seeds:** `SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]` (10 independent runs per model for paired **Diebold–Mariano**, **Wilcoxon signed-rank**, and **Friedman/Nemenyi** statistical tests).

### Software Installation (`environment.yml` / `pip`)

```bash
# Option A: Create the exact Conda environment
conda env create -f environment.yml
conda activate grid-flexibility-ai

# Option B: Install via pip (Python 3.10.12)
pip install pandapower==2.13.1 torch==2.1.0 torch-geometric==2.4.0 xgboost==2.0.2 \
            scikit-learn==1.3.2 optuna==3.5.0 scipy==1.11.4 pandas==2.1.3 \
            numpy==1.24.4 shap==0.44.0 matplotlib==3.8.2
```

### Running the MATPOWER Tabbed Dashboard (`matpower/fourthpaper.m`)

1. Install [MATPOWER 8.1](https://www.google.com/search?q=https%3A%2F%2Fmatpower.org%2Fdownload%2F) in MATLAB R2025a via `install_matpower(1, 1)`.
2. Open MATLAB, navigate to the `matpower/` directory, and run:
   ```matlab
   fourthpaper
   ```
3. A single standalone window with **9 interactive tabs** opens automatically, displaying **Figures 6–10** and **Tables 4–8** from the published *Battery Energy* (2025) paper.

## 6. Related Publications & Citation

If you use this code or dataset pipeline in your work, please cite our PhD dissertation and peer-reviewed journal publications:

1. **S. Hayat**, A. Nawaz, A. A. Almani, Z. Javid, and W. Holderbaum, "Optimization of Power System Flexibility Through AI-Driven Dynamic Load Management and Renewable Integration," *Battery Energy* (Wiley), vol. 4, p. e70026, 2025. DOI: [10.1002/bte2.20250009](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.1002%2Fbte2.20250009)
2. **S. Hayat** et al., "Dynamic Load Modeling and Optimized Battery Management Using LSTM for Enhanced Power Flexibility," *The Journal of Engineering* (IET), 2025. DOI: `10.1049/tje2.70126`
3. **S. Hayat** et al., "Physics-Guided Graph Attention and Spatiotemporal Graph Convolutional Networks for Voltage Stability Assessment," *Journal of Engineering* (Wiley), 2026. DOI: `10.1155/je/8588270`

```bibtex
@phdthesis{hayat2026enhancing,
  author = {Hayat, Saad},
  title  = {Enhancing Electric Grid Energy Flexibility using {AI}},
  school = {Department of Electrical Engineering, Gomal University},
  address= {Dera Ismail Khan, Pakistan},
  year   = {2026}
}

@article{hayat2025optimization,
  author  = {Hayat, Saad and Nawaz, Aamir and Almani, Aftab Ahmed and Javid, Zahid and Holderbaum, William},
  title   = {Optimization of Power System Flexibility Through {AI}-Driven Dynamic Load Management and Renewable Integration},
  journal = {Battery Energy},
  volume  = {4},
  pages   = {e70026},
  year    = {2025},
  doi     = {10.1002/bte2.20250009}
}
```
