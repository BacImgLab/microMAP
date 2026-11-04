# microMAP

Code Package for microMAP

This repository contains the computational workflows for microMAP, covering barcode-encoded 16S rRNA probe design, multi-round fluorescence image processing, bacterial identification and segmentation, and spatial microbiome analysis.

## Modules

| Module | Main purpose | Implementation |
| --- | --- | --- |
| [Probe Design](#probe-design) | Construct 16S rRNA reference databases, generate error-correcting codebooks, and design barcode-encoded probes | Python and R |
| [Spatial Microbiome Analysis](#spatial-microbiome-analysis) | Process microscopy images, segment and decode bacterial cells, reconstruct spatial maps, and analyze microbial spatial organization | MATLAB and Python |
| [BacUNet](#bacunet) | Train and apply a deep learning model for bacterial identification and segmentation from multi-channel fluorescence images | Python |

Each module has its own README describing dependencies, input formats, configuration, execution instructions, and outputs.

## Probe Design

The probe design module contains three submodules:

1. **Reference Database Construction:** Process full-length 16S ASV sequences and abundance tables into a taxonomy-annotated reference database using BLAST and TaxonKit.
2. **Error-Correcting Codebook Design:** Generate candidate barcodes and select codebooks satisfying a minimum pairwise Hamming-distance constraint.
3. **Probe Design and Construction:** Design primary probes using DECIPHER, filter their specificity, construct barcode-encoded probes, assign codes to target genera, and design helper probes.

Main outputs include reference sequences, taxonomic abundance summaries, codebooks, final probe sequences, and target-to-code assignments.

## Spatial Microbiome Analysis

The spatial analysis module provides an image-processing and analysis workflow comprising:

- Image extraction, ROI cropping, multi-round registration, and background subtraction.
- Cellpose-based bacterial instance segmentation and codebook-based barcode decoding.
- Optional tissue-boundary annotation and distance analysis.
- Spatial interaction, self-aggregation, and mesoscale neighborhood analysis.

Main outputs include decoded bacterial images, stitched spatial maps, cell-level coordinate tables, tissue-boundary distances, and spatial analysis results.

## BacUNet

BacUNet provides a deep learning workflow for bacterial identification and segmentation from multi-channel fluorescence images. Its progressive training strategy includes single-bacterial pre-training, category-based curriculum learning, and mixed-bacterial training. Application to real microbial-community data requires the corresponding data-loader adaptation and fine-tuning.

The module includes dataset loading, distributed training, validation, checkpoint saving, and inference. Main outputs include trained model checkpoints, predicted bacterial masks, probability maps, and evaluation metrics.

## Workflow Connections

Probe design supplies the sequences and barcode assignments used for microMAP experiments. After hybridization and imaging, the acquired fluorescence images are processed for bacterial identification and spatial analysis.

Two analysis routes are available:

- **Cellpose and barcode decoding:** Segment bacterial instances using the universal-probe images, then identify cells from their multi-round fluorescence signals and the experimental codebook.
- **BacUNet:** Use a trained network to predict bacterial masks and identities from multi-channel fluorescence images.

Before using BacUNet predictions in the downstream reconstruction and spatial analysis scripts, adapt the output format, filenames, and bacterial identifiers to the requirements documented in the spatial analysis module.

## Getting Started

1. Select the module required for your task and read its README.
2. Install the module-specific dependencies and prepare the required input data, databases, codebooks, or model checkpoints.
3. Update local paths and experiment-specific settings, including imaging rounds, channel order, target identities, and pixel size where applicable.
4. Run the scripts in the order documented for that module.

The modules use separate software environments. Probe design uses Python and R; spatial analysis uses MATLAB and Python; BacUNet training uses PyTorch with CUDA and distributed training through `torchrun`. Detailed requirements are provided in the respective module READMEs.

## Data and Configuration Notes

- The reference database and probe design examples are independent demonstrations, rather than a single end-to-end dataset. They do not reproduce the complete reference database or probe collection used in the study.
- Raw sequencing-read processing and ASV inference are outside the supplied reference database workflow.
- External reference databases and experiment-specific assets must be prepared as described in each module.
- Keep codebook assignments, image-channel order, model output classes, and downstream bacterial identifiers consistent across the selected workflow.
