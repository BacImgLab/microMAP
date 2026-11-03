###############################################
# main_design_probes.R
#
# Step 1: Design candidate primary probes
# ==============================================================================
# Primary Probe Design Pipeline
# ==============================================================================
#
# Purpose:
#     Designs candidate primary probes from 16S rRNA reference sequences
#     for downstream taxon-specific probe construction.
#
# Core Method:
#     1. Reads input 16S rRNA sequences from FASTA.
#     2. Generates candidate sequence regions using DECIPHER::TileSeqs().
#     3. Designs candidate probes using DECIPHER::DesignProbes() under
#        defined hybridization conditions.
#     4. Filters probes by length, GC content, and binding free energy.
#     5. Assigns unique probe IDs within each target and exports candidate
#        probe sequences for downstream BLAST specificity analysis.
#
# Inputs:
#     - 16S rRNA reference FASTA
#     - Pipeline configuration file: config/params.json
#
# Execution:
#     Run from the project root directory.
#
# Example Usage:
#     Rscript main_design_probes.R
#
# Outputs:
#     - outputs/01_design_probes/candidate_primary_probes_numbering.Rdata
#     - outputs/01_design_probes/candidate_primary_probes.fasta
#
# ==============================================================================
rm(list = ls())


# ============================================================
# 0. Load required packages
# ============================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(DBI)
  library(tools)
  library(RSQLite)
  library(Biostrings)
  library(DECIPHER)
  library(dplyr)
  library(purrr)
})


# ============================================================
# 1. Locate the script and parameter file
# ============================================================

# Detect the directory containing this script.
# This allows the default config file to be located without
# depending on the current working directory.

command_args_full <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", command_args_full, value = TRUE)

if (length(script_arg) > 0) {
  script_path <- sub("^--file=", "", script_arg[1])
  script_dir <- dirname(
    normalizePath(script_path, winslash = "/", mustWork = TRUE)
  )
} else {
  # Interactive mode (e.g. RStudio)
  script_dir <- normalizePath(
    getwd(),
    winslash = "/",
    mustWork = TRUE
  )
}


# Command-line usage:
# Rscript main_design_probes.R config/params.json

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  params_file <- args[1]
  message("[info] Using parameter file from command line: ", params_file)
} else {
  params_file <- file.path(script_dir, "config", "params.json")
  message("[info] No parameter file provided.")
  message("[info] Using default parameter file: ", params_file)
}

if (!file.exists(params_file)) {
  stop("Parameter file does not exist: ", params_file)
}

params_file <- normalizePath(
  params_file,
  winslash = "/",
  mustWork = TRUE
)

params <- jsonlite::fromJSON(params_file)


# ============================================================
# 2. Helper functions for path handling
# ============================================================

is_absolute_path <- function(path) {
  grepl("^(/|[A-Za-z]:[/\\\\])", path)
}

resolve_path <- function(path, root_dir) {
  
  if (is.null(path) || !nzchar(path)) {
    stop("An empty or undefined path was found in the parameter file.")
  }
  
  if (is_absolute_path(path)) {
    return(
      normalizePath(
        path,
        winslash = "/",
        mustWork = FALSE
      )
    )
  }
  
  normalizePath(
    file.path(root_dir, path),
    winslash = "/",
    mustWork = FALSE
  )
}


# ============================================================
# 3. Initialize project directories
# ============================================================

if (is.null(params$global$work_dir) ||
    !nzchar(params$global$work_dir)) {
  stop(
    "'global.work_dir' is missing or empty in the parameter file."
  )
}

work_dir <- resolve_path(
  params$global$work_dir,
  script_dir
)

if (!dir.exists(work_dir)) {
  stop("Working directory does not exist: ", work_dir)
}


# Directory containing custom functions

if (is.null(params$global$base_functions_dir) ||
    !nzchar(params$global$base_functions_dir)) {
  stop(
    "'global.base_functions_dir' is missing or empty in the parameter file."
  )
}

base_functions_dir <- resolve_path(
  params$global$base_functions_dir,
  work_dir
)

if (!dir.exists(base_functions_dir)) {
  stop(
    "Custom function directory does not exist: ",
    base_functions_dir
  )
}


# Output directory for Step 1

if (is.null(params$step1_design_probes$output_dir) ||
    !nzchar(params$step1_design_probes$output_dir)) {
  stop(
    "'step1_design_probes.output_dir' is missing or empty."
  )
}

output_dir <- resolve_path(
  params$step1_design_probes$output_dir,
  work_dir
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 4. Load required custom functions
# ============================================================

required_function_files <- c(
  "process_probes_design.R",
  "process_probes_filter.R",
  "numbering_probes.R"
)

for (f in required_function_files) {
  
  function_path <- file.path(
    base_functions_dir,
    f
  )
  
  if (!file.exists(function_path)) {
    stop(
      "Required function file does not exist: ",
      function_path
    )
  }
  
  source(function_path)
  
  message(
    "[source] ",
    f
  )
}

# ============================================================
# 5. Initialize input and output paths
# ============================================================

# Input FASTA

if (is.null(params$global$input_fasta) ||
    !nzchar(params$global$input_fasta)) {
  stop(
    "'global.input_fasta' is missing or empty."
  )
}

input_fasta <- resolve_path(
  params$global$input_fasta,
  work_dir
)

if (!file.exists(input_fasta)) {
  stop(
    "Input FASTA file does not exist: ",
    input_fasta
  )
}


# Candidate probe output

if (is.null(params$step1_design_probes$candidate_probes) ||
    !nzchar(params$step1_design_probes$candidate_probes)) {
  stop(
    "'step1_design_probes.candidate_probes' is missing or empty."
  )
}

candidate_probes_path <- resolve_path(
  params$step1_design_probes$candidate_probes,
  work_dir
)


# Candidate probe FASTA output

if (is.null(params$step1_design_probes$candidate_probes_fasta) ||
    !nzchar(params$step1_design_probes$candidate_probes_fasta)) {
  stop(
    "'step1_design_probes.candidate_probes_fasta' is missing or empty."
  )
}

candidate_probes_fasta_path <- resolve_path(
  params$step1_design_probes$candidate_probes_fasta,
  work_dir
)


# ============================================================
# 6. Design candidate probes
# ============================================================

message("============================================================")
message("[step 1] Designing candidate primary probes")
message("============================================================")


# A temporary SQLite database is used by DECIPHER during
# probe design and is removed automatically after completion.

db_file <- tempfile(
  pattern = "probe_design_",
  fileext = ".sqlite"
)

on.exit(
  unlink(db_file),
  add = TRUE
)


output_probes <- process_probes_design(
  input_fasta = input_fasta,
  db_file     = db_file,
  
  # Candidate tile length
  minLength_tiles = 28,
  maxLength_tiles = 34,
  
  # Probe design settings
  minLength_dp    = 28,
  maxLength_dp    = 34,
  maxPermutations = 1,
  
  # Hybridization conditions
  hybTemp = 37,      # Temperature in degrees Celsius
  P       = 100e-9,  # Probe concentration: 100 nM
  Na      = 0.3,     # Monovalent ion concentration: 0.3 M
  FA      = 50       # Formamide concentration: 50% (v/v)
)


# ============================================================
# 7. Filter candidate probes by GC content and delta G
# ============================================================

message("============================================================")
message("[step 2] Filtering candidate probes")
message("============================================================")


candidate_primary_probes <- process_probes_filter(
  probes = output_probes,
  
  minLength_dp = 28,
  maxLength_dp = 34,
  
  # Sequence and thermodynamic filters
  min_GC = 0.40,
  max_GC = 0.70,
  max_dG = -13.0,   # kcal/mol
  
  # Hybridization conditions
  temp = 37,
  P    = 100e-9,
  ions = 0.3,
  FA   = 50,
  
  batchSize = 1000
)


# ============================================================
# 8. Assign probe identifiers and export FASTA
# ============================================================

message("============================================================")
message("[step 3] Assigning probe identifiers")
message("============================================================")


numbering_res <- numbering_probes(
  probes_df         = candidate_primary_probes,
  probes_fasta_path = candidate_probes_fasta_path,
  seq_col           = "probe",
  id_col            = "identifier",
  probe_id_col      = "probe_id"
)

candidate_primary_probes_numbering <- numbering_res$probes_numbered


# ============================================================
# 9. Save results
# ============================================================

save(
  candidate_primary_probes_numbering,
  file = candidate_probes_path
)


message("============================================================")
message("[done] Candidate primary probe design completed.")
message("[output] Probe table: ", candidate_probes_path)
message("[output] Probe FASTA: ", candidate_probes_fasta_path)
message("============================================================")