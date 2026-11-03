# Error-Correcting Codebook Design

This module generates and optimizes error-correcting codebooks for multiplexed imaging experiments.

The workflow consists of two steps:

1. **Initial codebook generation**  
   Generate the complete set of 4-of-N candidate codes.

2. **Error-correcting codebook optimization**  
   Select a subset of candidate codes satisfying a user-defined minimum Hamming-distance constraint.

## Workflow

```text
generate_initial_codebook.py
        ↓
initial_codebook/
    codebook_F{F}N{N}.csv
        ↓
optimize_error_correcting_codebook.py
        ↓
error_correcting_codebook/
    error_correcting_codebook_*.csv
    error_correcting_codebook_*.json
```
## Codebook Parameters

Codebooks are defined by the number of imaging rounds (`N`), number of fluorophores (`F`), number of signal imaging rounds per barcode (`A`), and minimum Hamming distance (`D`).

In this workflow, the initial codebook is generated using `A = 4`, corresponding to a 4-of-N design.

## Requirements

### Required

* Python >= 3.9

### Tested with

* Python 3.13.2
* PyTorch 2.8.0
* CUDA 12.8

The initial codebook generation script and the CPU implementation of the optimization pipeline use only the Python standard library.

### Optional

* PyTorch
* CUDA-compatible GPU

PyTorch is only required when GPU acceleration is used. If PyTorch or CUDA is unavailable, the optimization pipeline can run on CPU.

---

# 1. Initial Codebook Generation

Script:

```text
generate_initial_codebook.py
```

This script generates a complete **4-of-N initial codebook**.

Each code contains exactly four non-zero positions. For each active position, the symbol can take one of `F` possible values (`1 ... F`), while `0` represents an inactive position.

The total number of generated codes is:

```text
C(N, 4) × F^4
```

where:

- `F` = number of fluorophores
- `N` = number of imaging rounds

## Usage

Basic usage:

```bash
python generate_initial_codebook.py --F 1 --N 8
```

By default, the generated codebook is saved to:

```text
initial_codebook/codebook_F1N8.csv
```

A custom output directory can be specified using `--output-dir`:

```bash
python generate_initial_codebook.py --F 1 --N 8 --output-dir example/output
```

The output file will then be saved as:

```text
example/output/codebook_F1N8.csv
```

## Output

The output filename follows the format:

```text
codebook_F{F}N{N}.csv
```

If `--output-dir` is not specified, the default output directory is:

```text
initial_codebook/
```

For example, with:

```text
F = 1
N = 8
```

the script generates:

```text
C(8, 4) × 1^4 = 70
```

candidate codes.

The generated codebook is automatically validated for:

- duplicate codes
- incorrect numbers of active positions
- all-zero codes
- consistency with the expected theoretical codebook size

---

# 2. Error-Correcting Codebook Optimization

Script:

```text
optimize_error_correcting_codebook.py
```

This script selects a large error-correcting subset from the initial candidate codebook.

The optimization procedure combines:

1. simulated annealing for initial seed selection
2. progressive Hamming-distance relaxation
3. greedy codebook extension
4. repeated independent searches
5. final minimum pairwise Hamming-distance validation

The search starts from `D_start` and progressively relaxes the distance constraint until `D_min`.

If `--d-start` is not specified:

```text
D_start = N
```

---

## Input Modes

Two input modes are supported.

### Option 1: Specify F and N

This is the recommended usage.

```bash
python optimize_error_correcting_codebook.py --F 1 --N 8 --d-min 4 --threshold 14
```

The script automatically searches for:

```text
initial_codebook/codebook_F1N8.csv
```

A more complete example is:

```bash
python optimize_error_correcting_codebook.py --F 1 --N 8 --d-min 4 --threshold 14 --time-limit 7200 --max-iterations 100 --workers 8 --device auto
```

### Option 2: Specify the input CSV directly

```bash
python optimize_error_correcting_codebook.py --input initial_codebook/codebook_F1N8.csv --d-min 4 --threshold 14
```

A more complete example is:

```bash
python optimize_error_correcting_codebook.py --input initial_codebook/codebook_F1N8.csv --d-min 4 --threshold 14 --time-limit 7200 --max-iterations 100 --workers 8 --device auto
```

---

## Main Parameters

### `--F`

Number of fluorophores.

Used together with `--N` to automatically locate the corresponding initial codebook.

Example:

```text
--F 1
```

### `--N`

Total number of imaging rounds.

Example:

```text
--N 8
```

### `--d-min`

Required minimum pairwise Hamming distance in the final codebook.

Example:

```text
--d-min 4
```

### `--d-start`

Starting Hamming-distance constraint.

This parameter is optional. If omitted:

```text
D_start = N
```

For example, when `N = 8`, the search starts from `D = 8`.

To override the default:

```text
--d-start 6
```

### `--threshold`

Target number of codes.

If threshold-based stopping is enabled, the search terminates once a codebook containing at least this number of codes is found.

Example:

```text
--threshold 14
```

### `--time-limit`

Maximum search time in seconds.

Default:

```text
7200
```

### `--max-iterations`

Maximum number of independent search runs.

Default:

```text
100
```

### `--workers`

Number of CPU worker processes.

Use:

```text
--workers 1
```

for single-process CPU execution.

Higher values enable multiprocessing.

### `--device`

Execution mode:

```text
auto
cpu
gpu
```

Default:

```text
auto
```

In `auto` mode, CUDA GPU acceleration is used when available; otherwise, the script falls back to CPU execution.

### `--seed`

Optional random seed for improved reproducibility.

Example:

```text
--seed 260904
```

If no seed is provided, independent searches use randomized initial states.

### `--output-dir`

Output directory for the optimized codebook and metadata.

Default:

```text
error_correcting_codebook/
```

A custom directory can be specified, for example:

```bash
python optimize_error_correcting_codebook.py --F 1 --N 8 --d-min 4 --threshold 14 --output-dir example/output
```

---

## Stopping Conditions

Three stopping conditions are supported:

- target codebook size
- elapsed search time
- maximum iteration count

By default, all three are enabled.

They can be disabled individually using:

```text
--no-stop-by-threshold
--no-stop-by-time
--no-stop-by-iteration
```

At least one stopping condition must remain enabled.

---

## Output

The optimized codebook is saved by default to:

```text
error_correcting_codebook/
```

Example output:

```text
error_correcting_codebook/
├── error_correcting_codebook_F1N8_Dstart8_Dmin4_count14.csv
└── error_correcting_codebook_F1N8_Dstart8_Dmin4_count14.json
```

### CSV output

The CSV file contains the final selected error-correcting codebook.

### JSON metadata

The JSON file records information about the optimization run, including:

- input file
- F
- N
- initial code count
- selected code count
- `D_start`
- `D_min`
- observed minimum pairwise Hamming distance
- target threshold
- time limit
- maximum iterations
- completed iterations
- execution time
- CPU / GPU mode
- number of CPU workers
- random seed
- progressive search trace

---

## Recommended Example

First generate the initial codebook:

```bash
python generate_initial_codebook.py --F 1 --N 8
```

Then optimize the error-correcting codebook:

```bash
python optimize_error_correcting_codebook.py --F 1 --N 8 --d-min 4 --threshold 14 --time-limit 7200 --max-iterations 100 --workers 8 --device auto
```

The complete workflow is:

```text
F, N
 ↓
4-of-N initial codebook
 ↓
progressive Hamming-distance optimization
 ↓
error-correcting codebook
```
