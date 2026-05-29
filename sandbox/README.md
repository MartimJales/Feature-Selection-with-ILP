# PADTAI Sandbox

Quick testing environment for PADTAI with clusters 0 and 1.

## Quick Start

```bash
bash sandbox/run_padtai_sandbox.sh
```

This will:
1. Create a timestamped output folder `sandbox/output_YYYYMMDD_HHMMSS/`
2. Preprocess the cluster CSVs (dropna, sanitize columns, coerce types)
3. Run PADTAI on both clusters with `--intcols auto` (automatic numeric column detection)
4. Extract rules from PADTAI output
5. Generate `padtai_rules.json` for each cluster
6. **Automatically evaluate clusters** and save metrics to `evaluation_results.csv`

Each run creates a **new timestamped folder**, preserving all previous iterations.

## Output Structure

```
sandbox/
├── cluster_0/           # Original input data (unchanged)
│   └── padtai_input.csv
├── cluster_1/           # Original input data (unchanged)
│   └── padtai_input.csv
│
└── output_20260529_143022/  # Timestamped run directory
    ├── cluster_0/
    │   ├── padtai_input.csv                 (copy of original)
    │   ├── padtai_input.prepared.csv        (after dropna, sanitize, coerce)
    │   ├── padtai_input.sample.csv          (first 5 rows)
    │   ├── padtai_stdout.txt                (PADTAI stdout)
    │   ├── padtai_stderr.txt                (PADTAI stderr)
    │   ├── padtai_returncode.txt            (exit code)
    │   ├── padtai_rules.json                (extracted rules)
    │   ├── run.log                          (detailed wrapper logs)
    │   └── padtai_output/                   (PADTAI's prolog artifacts)
    │
    ├── cluster_1/
    │   └── [same structure as cluster_0]
    │
    └── evaluation_results.csv               (metrics for all clusters)
```

## Command Options

```bash
python3 sandbox/run_padtai_sandbox.py \
  --clusters 0 1 \           # which clusters to run (default: 0, 1)
  --timeout 600 \            # PADTAI timeout in seconds (default: 600)
  --intcols auto \           # auto-detect numeric columns (default: none)
  --no-eval                  # skip automatic evaluation (optional)
```

`--intcols` modes:
- `none` - treat all columns as binary (default in previous runs)
- `auto` - auto-detect count-like columns with >2 distinct values
- `4,5,6` - specify exact column indices as numeric

## Evaluation Output

The script automatically evaluates each cluster and saves `evaluation_results.csv` with:
- `cluster_id` - cluster identifier
- `n_samples` - number of samples in the dataset
- `n_rules` - total number of extracted rules
- `n_label1_rules` - rules mentioning attr_label_1 (positive)
- `n_label0_rules` - rules mentioning attr_label_0 (negative)
- `accuracy` - classification accuracy
- `recall` - true positive rate
- `precision` - positive predictive value
- `f1` - F1-score
- `tp`, `fp`, `tn`, `fn` - confusion matrix values

## Examples

```bash
# Run with numeric column detection (default)
bash sandbox/run_padtai_sandbox.sh

# Run specific cluster(s) with custom timeout
python3 sandbox/run_padtai_sandbox.py --clusters 0 --timeout 300

# Run without numeric detection
python3 sandbox/run_padtai_sandbox.py --clusters 0 1 --intcols none

# Run PADTAI only, skip evaluation
python3 sandbox/run_padtai_sandbox.py --clusters 0 1 --no-eval
```

## Viewing Results

```bash
# List all timestamped runs
ls -la sandbox/output_*/

# View latest run
ls -la sandbox/output_$(ls -t sandbox/output_* | head -1 | xargs -n1 basename)/

# Compare metrics from multiple runs
cat sandbox/output_20260529_143022/evaluation_results.csv
cat sandbox/output_20260529_150530/evaluation_results.csv
```
