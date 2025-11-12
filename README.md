# lasso_with_bootstrapped_sampling
**Lasso-BSS** (Bootstrap Stability Selection for Lasso) is a lightweight Python package for evaluating the robustness of Lasso regression models using bootstrap resampling.
It is designed for scientific workflows (e.g., ecology, geoscience, or data-driven modeling) that require transparent model selection and reproducible statistics.


---
## Installation
```bash
git clone https://github.com/<your_username>/lasso-bss.git
cd lasso-bss
pip install -e .
```
---

## Example Usage
### Windows
```bat
examples\run.bat
```

### Linux / macOS
```bash
bash examples/run.sh
```
This example will:
1. Load data from `examples/data/`
2. Run 50 bootstrap (in default) iterations with 5-fold cross-validation
3. Output results and plots to `examples/outputs/`
---
## Input csv
1. siteID column
2. feature columns
3. response variable  `y`
   
---
## Directory Structure
```csharp
lasso-bss/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ lasso_with_bss/
│     ├─ __init__.py
│     ├─ io.py          # Data loading & cleaning
│     ├─ lasso.py       # Bootstrap Lasso core
│     ├─ cli.py         # Command-line interface
│     └─ utils.py       # Helper functions (optional)
├─ examples/
│  ├─ data/             # Example CSVs
│  ├─ outputs/          # Generated outputs
│  └─ run.bat / run.sh  # Example scripts
```
---

## Outputs
Running the example produces
`alpha_mae_curves.png` - MAE vs alpha plot (stability visualization + alpha range validation)
`coef_matrix.csv` - coefficients from each bootstrap iteration
`coef_summary.csv` - feature-wise mean, std, quartiles and vote counts

## License
Mit License (C) 2025 Jiacheng Ling
