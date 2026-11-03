# microMAP Probe Design

This module contains the computational workflows used to prepare the mouse gut 16S rRNA reference database, generate error-correcting codebooks, and design barcode-encoded 16S rRNA probes for microMAP.

The module is organized into three submodules. Reference database preparation and codebook design provide the sequence resources and imaging codes used in downstream probe design and construction.

## Submodules

| Directory                                                | Submodule | Main purpose | Implementation |
|----------------------------------------------------------| --- | --- | --- |
| [`Reference Database/`](01_reference_database/README.md) | 16S rRNA Reference Database Construction | Process PacBio full-length 16S ASVs into a taxonomy-annotated, abundance-linked reference database | Python |
| [`Codebook Design/`](02_codebook_design/README.md)       | Error-Correcting Codebook Design | Generate candidate codes and select codebooks satisfying a minimum Hamming-distance constraint | Python |
| [`Probe Construction/`](03_probe_construction/README.md) | 16S rRNA Probe Design and Construction | Design and filter primary probes, construct barcode-encoded probes, assign codes, and generate helper probes | R |

Each submodule contains its own `README.md` with dependencies, input formats, configuration, execution instructions, and output descriptions.

## Workflow Overview

### 1. Reference database construction

This workflow starts from **processed full-length 16S ASV sequences and an ASV abundance table**. Raw PacBio read processing and ASV inference are outside the scope of the supplied pipeline.

The pipeline performs BLAST-based annotation, sequence orientation, taxonomy retrieval with TaxonKit, and abundance integration. Its main outputs are:

- `Species_summary_abundance.csv`: species-level abundance summary.
- `Genus_summary_abundance.csv`: genus-level abundance summary.
- `global_all_taxa_consensus.fasta`: merged 16S reference FASTA containing ASV sequences eligible for export.

Unassigned sequences can be retained as background sequences for downstream specificity screening.

### 2. Error-correcting codebook design

This workflow generates a complete set of 4-of-N candidate codes and selects a subset satisfying the requested minimum pairwise Hamming distance.

Codebooks are described by:

- `N`: number of imaging rounds.
- `F`: number of fluorophores.
- `A`: number of signal imaging rounds per barcode; the current generator uses `A = 4`.
- `D`: minimum pairwise Hamming distance.

The main outputs are an optimized codebook in CSV format and optimization metadata in JSON format. Codebook generation can run independently of reference database construction.

### 3. Probe design and construction

This workflow uses 16S reference sequences, abundance information, an error-correcting codebook, and a secondary/readout probe table. It consists of four sequential steps:

1. Design candidate primary probes using DECIPHER.
2. Filter primary probes by BLAST specificity and prioritize them using represented abundance and probe efficiency.
3. Construct barcode-encoded combined probes, filter their specificity, and assign a unique available code to each target genus.
4. Design and filter helper probes adjacent to the selected primary probes.

The main outputs include final combined-probe sequences, genus-to-code assignments, probe and codebook information tables, and helper-probe sequences.

## Example Data and Connections Between Submodules

**The supplied examples are independent demonstrations, not a single end-to-end dataset.** Reduced example inputs are provided for reference database construction and probe design. They illustrate the workflows and do not reproduce the complete reference database or probe collection used in the study.

When applying these workflows to a complete dataset:

- Prepare the reference FASTA and abundance table using the outputs of the reference database construction workflow, following the input formats required for probe design and construction.
- Prepare the codebook using the error-correcting codebook design workflow, ensuring that its imaging-round columns and fluorophore symbols match the secondary/readout probe table.
- Provide the readout probe table separately, including the required sequences and identifiers.

## Getting Started

1. Open the README for the submodule you want to run.
2. Install its dependencies and prepare any required external databases. The NCBI reference and taxonomy databases used by submodule 1 are not included.
3. Configure the input, output, and executable paths as described in that README.
4. Run commands from the corresponding **submodule directory**. In submodule 3, execute the four R scripts in their documented order.

Dependencies and tested versions are documented separately for each submodule. PyTorch and a CUDA-compatible GPU are optional for codebook optimization; CPU execution is supported.
