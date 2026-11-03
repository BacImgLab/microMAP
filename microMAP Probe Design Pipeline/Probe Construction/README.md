# 16S rRNA Probe Design and Construction

This module designs genus-targeted 16S rRNA primary probes, constructs barcode-encoded combined probes, assigns error-correcting codes, and generates helper probes for downstream multiplexed FISH experiments.

The workflow is implemented in R and consists of four main steps:

```text
Step 1. Primary probe design
        ↓
Step 2. Primary probe specificity filtering
        ↓
Step 3. Combined probe construction and code assignment
        ↓
Step 4. Helper probe design and filtering
```

The provided input files are reduced example datasets intended to demonstrate the workflow and code structure rather than reproduce the complete probe collection used in the study.

---

## Workflow

### Step 1. Primary probe design

`main_design_probes.R`

Candidate primary probes are designed from input 16S rRNA reference sequences using [DECIPHER](https://bioconductor.org/packages/release/bioc/html/DECIPHER.html).

The workflow:

1. Reads the input 16S rRNA FASTA file.
2. Generates candidate sequence regions using `DECIPHER::TileSeqs()`.
3. Designs candidate probes using `DECIPHER::DesignProbes()`.
4. Filters probes by:
   - probe length
   - GC content
   - probe binding free energy
5. Assigns unique probe IDs within each target.
6. Exports candidate probe sequences for downstream BLAST analysis.

Run:

```bash
Rscript main_design_probes.R
```

Main outputs:

```text
outputs/01_design_probes/
├── candidate_primary_probes_numbering.Rdata
└── candidate_primary_probes.fasta
```

---

### Step 2. Primary probe specificity filtering

`main_pri_probes_filter.R`

Candidate primary probes are screened by BLAST against the reference 16S rRNA sequence database.

The workflow:

1. Builds a local nucleotide BLAST database from the input reference 16S rRNA sequences.
2. Runs BLASTN for candidate primary probes.
3. Parses BLAST XML output and calculates:
   - total matched bases
   - matched GC bases
   - maximum consecutive matches
   - maximum consecutive matched GC bases
4. Treats same-genus hits as expected target hits and evaluates different-genus hits as potential off-targets.
5. Removes probes with excessive off-target matching.
6. Summarizes the abundance represented by same-genus BLAST hits.
7. Prioritizes probes by represented abundance and probe efficiency.
8. Selects non-overlapping probes while enforcing a minimum spacing between selected probes.

Run:

```bash
Rscript main_pri_probes_filter.R
```

Main outputs:

```text
outputs/
├── ref_16S_db/
│   └── ref_16S_db.*
│
└── 02_primary_filter/
    ├── blast/
    ├── filter_primary_probes.Rdata
    └── fin_primary_probes.Rdata
```

The BLAST database in `outputs/ref_16S_db/` is generated from the input reference FASTA and reused in the subsequent combined-probe filtering step.

---

### Step 3. Combined probe construction and code assignment

`main_combine_probes.R`

Final primary probes are combined with barcode-specific secondary/readout probe sequences, followed by BLAST filtering and code assignment.

The workflow:

1. Extracts short sequence-derived spacer regions adjacent to each primary probe.
2. Duplicates primary probes into paired probe-combination groups.
3. Converts the codebook to long format and maps active imaging rounds to the corresponding secondary/readout probes.
4. Expands primary probes across candidate codes.
5. Generates valid two-group partitions of active codebook readouts.
6. Assigns readouts within each group to left and right positions without fluorophore-specific positional preference.
7. Constructs candidate combined probes by placing the primary probe between the corresponding left and right readout sequences.
8. Runs BLASTN specificity filtering on candidate combined probes.
9. Retains complete probe-code combinations that pass BLAST filtering.
10. Assigns one unique available code to each target genus using a greedy search.
11. Selects one valid probe combination for each assigned genus-code pair.

Run:

```bash
Rscript main_combine_probes.R
```

Main outputs:

```text
outputs/03_combine_probes/
├── blast/
├── candidate_combined_probes.fasta
├── fin_comb_probes.fasta
├── fin_assigned_probes_core.csv
├── fin_grouped_codebook_info.csv
├── assigned_codes.csv
└── available_probe_set.csv
```

The error-correcting codebook used in this step is generated separately. Details of codebook generation and optimization are provided in the corresponding codebook-design module.

---

### Step 4. Helper probe design and filtering

`main_helper_probes.R`

Helper probes are designed adjacent to the final primary probes and filtered for positional conflicts and sequence similarity.

The workflow:

1. Designs left and right helper probes adjacent to each final primary probe.
2. Maintains a defined interval between helper and primary probe binding regions.
3. Identifies neighboring primary probes that are too close to accommodate independent helper regions.
4. Removes helper probes located between conflicting neighboring primary probes.
5. Builds a local BLAST database from the final primary probe sequences.
6. Runs BLASTN to evaluate helper-probe similarity to primary probes.
7. Filters helper probes using:
   - total matched bases
   - matched GC bases
   - maximum consecutive matches
   - maximum consecutive matched GC bases
8. Retains helper probes with no BLAST hits or acceptable BLAST hits.
9. Deduplicates identical helper sequences while retaining information about shared parent probes and reference targets.

Run:

```bash
Rscript main_helper_probes.R
```

Main outputs:

```text
outputs/04_helper_probes/
├── primary_probes_db/
├── intermediate/
├── blast/
├── helper_probes_final.csv
├── helper_probes_final.Rdata
└── helper_probes_final.fasta
```

---

## Project Structure

```text
.
├── main_design_probes.R
├── main_pri_probes_filter.R
├── main_combine_probes.R
├── main_helper_probes.R
│
├── config/
│   └── params.json
│
├── example/
│   ├── example_16S.fasta
│   ├── example_abundance.csv
│   ├── example_codebook.csv
│   └── example_sec_probes.csv
│
├── functions/
│   └── *.R
│
└── outputs/
    ├── ref_16S_db/
    ├── 01_design_probes/
    ├── 02_primary_filter/
    ├── 03_combine_probes/
    └── 04_helper_probes/
```

---

## Configuration

Pipeline paths and input files are defined in:

```text
config/params.json
```

The configuration includes:

- project working directory
- input 16S rRNA FASTA
- abundance table
- error-correcting codebook
- secondary/readout probe table
- BLAST executable paths
- reference 16S BLAST database path
- step-specific output paths

Example:

```json
{
  "global": {
    "work_dir": ".",
    "base_functions_dir": "functions",
    "input_fasta": "example/example_16S.fasta",
    "abundance_file": "example/example_abundance.csv",
    "codebook_path": "example/example_codebook.csv",
    "sec_probes_path": "example/example_sec_probes.csv",
    "ref_16S_db": "outputs/ref_16S_db/ref_16S_db"
  }
}
```

Paths to `blastn` and `makeblastdb` should be adjusted according to the local BLAST+ installation.

---

## Input Files

### 16S rRNA reference FASTA

The FASTA file contains reference 16S rRNA sequences used for primary probe design and specificity analysis.

Sequence identifiers should correspond to the taxon names used in the abundance table.

Example:

```text
>Duncaniella sp1
ACGT...
>Duncaniella sp2
ACGT...
```

### Abundance table

The abundance table contains taxonomic information and abundance values used for primary-probe prioritization and code assignment.

A `taxon_name` field is constructed from genus and species information and is used to match abundance records with FASTA/BLAST sequence identifiers.

Relevant fields include:

```text
Genus
Species
taxon_name
Abundance
rank
```

### Error-correcting codebook

The codebook contains one barcode per row and imaging-round assignments across `Round*` columns.

A value of `0` indicates no signal in that imaging round, while non-zero values specify the fluorophore/readout identity.

The codebook is generated separately using the codebook-design workflow. See the corresponding codebook module for codebook generation, Hamming-distance optimization, and related parameters.

### readout probe table

The readout probe table contains the fluorescent readout probe sequences, probe names, matching identifiers, and sequence modification information used for combined-probe construction.


---

## Requirements and Tested Environment

The workflow was tested with the following environment:

```text
R             4.5.1

jsonlite      2.0.0
DBI           1.3.0
tools         4.5.1
RSQLite       3.53.3
Biostrings    2.76.0
DECIPHER      3.4.0
dplyr         1.2.1
tibble        3.3.1
tidyr         1.3.2
stringr       1.6.0
stringi       1.8.7
purrr         1.2.2
readr         2.2.0
XML           3.99.0.23
seqinr        4.2.44

NCBI BLAST+   2.15.0+
```

The workflow uses the following BLAST+ executables:

```text
blastn
makeblastdb
```

`tools` is included with R.

---

## Running the Example Workflow

Run the scripts from the project root directory in the following order:

```bash
Rscript main_design_probes.R
Rscript main_pri_probes_filter.R
Rscript main_combine_probes.R
Rscript main_helper_probes.R
```

All output files are written to the corresponding directories under `outputs/`.
