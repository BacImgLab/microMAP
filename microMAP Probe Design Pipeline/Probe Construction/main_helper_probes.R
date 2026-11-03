###############################################
# main_helper_probes.R
#
# Step 4: Design and filter helper probes
# ==============================================================================
# Helper Probe Design and Filtering Pipeline
# ==============================================================================
#
# Purpose:
#     Designs and filters helper probes adjacent to the final primary probes
#     to support downstream 16S rRNA probe hybridization.
#
# Core Method:
#     1. Designs left and right helper probes from 16S rRNA regions adjacent
#        to each final primary probe.
#     2. Marks neighboring primary probes that are too close to accommodate
#        independent helper regions.
#     3. Removes helper probes located between conflicting neighboring
#        primary probes.
#     4. Builds a local BLAST database from the final primary probe sequences.
#     5. Runs BLASTN to evaluate helper-probe matches against primary probes.
#     6. Filters helper probes using total, GC, and consecutive matching
#        thresholds.
#     7. Retains helper probes with no BLAST hits or acceptable BLAST hits.
#     8. Deduplicates identical helper sequences while preserving information
#        about shared parent probes and reference targets.
#
# Inputs:
#     - Final primary probes from Step 2
#     - 16S rRNA reference FASTA
#     - Pipeline configuration file: config/params.json
#
# Execution:
#     Run from the project root directory after Step 2.
#     Step 3 is not required for helper-probe design.
#
# Example Usage:
#     Rscript main_helper_probes.R
#
# Outputs:
#     - outputs/04_helper_probes/primary_probes_db/
#     - outputs/04_helper_probes/intermediate/
#     - outputs/04_helper_probes/blast/
#     - outputs/04_helper_probes/helper_probes_final.csv
#     - outputs/04_helper_probes/helper_probes_final.Rdata
#     - outputs/04_helper_probes/helper_probes_final.fasta
#
# ==============================================================================

rm(list = ls())


# ============================================================
# 0. Load required packages
# ============================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(readr)
  
  library(Biostrings)
  library(XML)
  library(stringi)
})


# ============================================================
# 1. Locate the script and parameter file
# ============================================================

# Detect the directory containing this script so that the
# default configuration file can be located independently of
# the current working directory.

command_args_full <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", command_args_full, value = TRUE)

if (length(script_arg) > 0) {
  
  script_path <- sub("^--file=", "", script_arg[1])
  
  script_dir <- dirname(
    normalizePath(
      script_path,
      winslash = "/",
      mustWork = TRUE
    )
  )
  
} else {
  
  # Interactive mode, e.g. running from RStudio
  script_dir <- normalizePath(
    getwd(),
    winslash = "/",
    mustWork = TRUE
  )
}


# Command-line usage:
# Rscript main_helper_probes.R config/params.json

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  
  params_file <- args[1]
  
  message(
    "[info] Using parameter file from command line: ",
    params_file
  )
  
} else {
  
  params_file <- file.path(
    script_dir,
    "config",
    "params.json"
  )
  
  message("[info] No parameter file provided.")
  message(
    "[info] Using default parameter file: ",
    params_file
  )
}


if (!file.exists(params_file)) {
  
  stop(
    "Parameter file does not exist: ",
    params_file
  )
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
  
  grepl(
    "^(/|[A-Za-z]:[/\\\\])",
    path
  )
}


resolve_path <- function(path, root_dir) {
  
  if (is.null(path) || !nzchar(path)) {
    
    stop(
      "An empty or undefined path was found in the parameter file."
    )
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
    file.path(
      root_dir,
      path
    ),
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
  
  stop(
    "Working directory does not exist: ",
    work_dir
  )
}


# Directory containing custom R functions

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


# ============================================================
# 4. Initialize Step 4 output directories
# ============================================================

if (is.null(params$step4_helper_probes$output_dir) ||
    !nzchar(params$step4_helper_probes$output_dir)) {
  
  stop(
    "'step4_helper_probes.output_dir' is missing or empty."
  )
}


output_dir <- resolve_path(
  params$step4_helper_probes$output_dir,
  work_dir
)


dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# Intermediate files generated during helper probe design

intermediate_dir <- file.path(
  output_dir,
  "intermediate"
)


blast_dir <- file.path(
  output_dir,
  "blast"
)


primary_probe_db_dir <- file.path(
  output_dir,
  "primary_probes_db"
)


dir.create(
  intermediate_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


dir.create(
  blast_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


dir.create(
  primary_probe_db_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 5. Load required custom functions
# ============================================================

required_function_files <- c(
  "create_blast_database.R",
  "design_helper_probes.R",
  "mark_close_primary_probes.R",
  "filter_helper_overlaps.R",
  "run_blastn.R",
  "xml_extract.R",
  "filter_helper_blast_hits.R",
  "finalize_helper_probes_by_blast.R",
  "deduplicate_helper_probes.R"
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
# 6. Initialize input paths
# ============================================================

# Step 2 final primary probes

if (is.null(params$step2_pri_filter$fin_primary_probes) ||
    !nzchar(params$step2_pri_filter$fin_primary_probes)) {
  
  stop(
    "'step2_pri_filter.fin_primary_probes' is missing or empty."
  )
}


fin_primary_probes_rdata <- resolve_path(
  params$step2_pri_filter$fin_primary_probes,
  work_dir
)


# Input 16S rRNA sequences

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


# ============================================================
# 7. Initialize external tool paths
# ============================================================

if (is.null(params$tools$blastn) ||
    !nzchar(params$tools$blastn)) {
  
  stop(
    "'tools.blastn' is missing or empty."
  )
}


if (is.null(params$tools$makeblastdb) ||
    !nzchar(params$tools$makeblastdb)) {
  
  stop(
    "'tools.makeblastdb' is missing or empty."
  )
}


blastn      <- params$tools$blastn
makeblastdb <- params$tools$makeblastdb


# ============================================================
# 8. Initialize final output paths
# ============================================================

required_step4_params <- c(
  "helper_probes_final_csv",
  "helper_probes_final_rdata",
  "helper_probes_final_fasta"
)


for (x in required_step4_params) {
  
  if (is.null(params$step4_helper_probes[[x]]) ||
      !nzchar(params$step4_helper_probes[[x]])) {
    
    stop(
      "'step4_helper_probes.",
      x,
      "' is missing or empty."
    )
  }
}


helper_probes_final_csv <- resolve_path(
  params$step4_helper_probes$helper_probes_final_csv,
  work_dir
)


helper_probes_final_rdata <- resolve_path(
  params$step4_helper_probes$helper_probes_final_rdata,
  work_dir
)


helper_probes_final_fasta <- resolve_path(
  params$step4_helper_probes$helper_probes_final_fasta,
  work_dir
)


# ============================================================
# 9. Check required files and external tools
# ============================================================

required_files <- c(
  fin_primary_probes_rdata,
  input_fasta,
  blastn,
  makeblastdb
)


for (f in required_files) {
  
  if (!file.exists(f)) {
    
    stop(
      "Required file does not exist: ",
      f
    )
  }
}


# ============================================================
# 10. Load primary probes and 16S sequences
# ============================================================

message("============================================================")
message("[step 1] Loading primary probes and 16S sequences")
message("============================================================")


step2_env <- new.env()


load(
  fin_primary_probes_rdata,
  envir = step2_env
)


if (!exists(
  "fin_primary_probes",
  envir = step2_env,
  inherits = FALSE
)) {
  
  stop(
    "Object 'fin_primary_probes' was not found in: ",
    fin_primary_probes_rdata
  )
}


fin_primary_probes <- get(
  "fin_primary_probes",
  envir = step2_env
)


rm(step2_env)


if (!is.data.frame(fin_primary_probes)) {
  
  stop(
    "'fin_primary_probes' is not a data.frame or tibble."
  )
}


# Normalize the end-position column.
#
# Some probe-design outputs use ".end", whereas others may
# contain "end". Helper-probe functions use ".end".

if (!".end" %in% colnames(fin_primary_probes) &&
    "end" %in% colnames(fin_primary_probes)) {
  
  fin_primary_probes$.end <- fin_primary_probes$end
}


required_pri_cols <- c(
  "probe_id",
  "probe",
  "identifier",
  "start",
  ".end"
)


missing_pri_cols <- setdiff(
  required_pri_cols,
  colnames(fin_primary_probes)
)


if (length(missing_pri_cols) > 0) {
  
  stop(
    "'fin_primary_probes' is missing required column(s): ",
    paste(
      missing_pri_cols,
      collapse = ", "
    )
  )
}


seq_16S <- Biostrings::readDNAStringSet(
  filepath = input_fasta
)


message(
  "[info] Number of primary probes: ",
  nrow(fin_primary_probes)
)


message(
  "[info] Number of 16S sequences: ",
  length(seq_16S)
)


# ============================================================
# 11. Check BLAST tools
# ============================================================

message("============================================================")
message("[step 2] Checking BLAST tools")
message("============================================================")


message(
  "[info] blastn: ",
  blastn
)

system2(
  blastn,
  args = "-version"
)


message(
  "[info] makeblastdb: ",
  makeblastdb
)

system2(
  makeblastdb,
  args = "-version"
)


# ============================================================
# 12. Build a BLAST database from primary probes
# ============================================================

# This database is specific to Step 4 and is used to detect
# potentially problematic complementarity between helper
# probes and selected primary probes.

message("============================================================")
message("[step 3] Building primary-probe BLAST database")
message("============================================================")


primary_probes_fasta <- file.path(
  primary_probe_db_dir,
  "primary_probes_db.fasta"
)


primary_probe_db_prefix <- file.path(
  primary_probe_db_dir,
  "primary_probes_db"
)


primary_probe_set <- Biostrings::DNAStringSet(
  as.character(
    fin_primary_probes$probe
  )
)


names(primary_probe_set) <- as.character(
  fin_primary_probes$probe_id
)


Biostrings::writeXStringSet(
  primary_probe_set,
  filepath = primary_probes_fasta
)


message(
  "[info] Primary probes written to BLAST database: ",
  length(primary_probe_set)
)


primary_probe_db <- create_blast_database(
  makeblastdb = makeblastdb,
  query_db    = primary_probes_fasta,
  db_out      = primary_probe_db_prefix
)


# ============================================================
# 13. Design candidate helper probes
# ============================================================

message("============================================================")
message("[step 4] Designing candidate helper probes")
message("============================================================")


helper_probes <- design_helper_probes(
  fin_primary_probes = fin_primary_probes,
  seq_16S            = seq_16S,
  helper_length      = 30,
  interval           = 3
)


message(
  "[info] Candidate helper probes generated: ",
  nrow(helper_probes)
)


# ============================================================
# 14. Remove helper probes affected by nearby primary probes
# ============================================================

# Primary probes located too close to neighboring primary
# probes may not have sufficient space for valid helper probes.

message("============================================================")
message("[step 5] Filtering helper probes by primary-probe proximity")
message("============================================================")


fin_primary_probes_marked <- mark_close_primary_probes(
  fin_primary_probes = fin_primary_probes,
  helper_length      = 30,
  interval           = 3
)


message(
  "[info] Primary probes with helper overlap flag: ",
  sum(
    fin_primary_probes_marked$helper_overlap_flag,
    na.rm = TRUE
  )
)


message(
  "[info] Primary probes too close to next probe: ",
  sum(
    fin_primary_probes_marked$too_close_to_next,
    na.rm = TRUE
  )
)


message(
  "[info] Primary probes too close to previous probe: ",
  sum(
    fin_primary_probes_marked$too_close_to_prev,
    na.rm = TRUE
  )
)


helper_overlap_res <- filter_helper_overlaps(
  helper_probes             = helper_probes,
  fin_primary_probes_marked = fin_primary_probes_marked
)


helper_probes_marked <- helper_overlap_res$helper_probes_marked

helper_probes_removed_by_overlap <-
  helper_overlap_res$removed_helper_probes

helper_probes_filtered_by_overlap <-
  helper_overlap_res$helper_probes_filtered


message(
  "[info] Helper probes before overlap filtering: ",
  nrow(helper_probes)
)


message(
  "[info] Helper probes removed by overlap filtering: ",
  nrow(helper_probes_removed_by_overlap)
)


message(
  "[info] Helper probes after overlap filtering: ",
  nrow(helper_probes_filtered_by_overlap)
)


if (nrow(helper_probes_filtered_by_overlap) == 0) {
  
  stop(
    "No helper probes remained after primary-probe proximity filtering."
  )
}


# ============================================================
# 15. Write overlap-filtered helper probes to FASTA
# ============================================================

helper_probes_filtered_by_overlap_fasta <- file.path(
  intermediate_dir,
  "helper_probes_filtered_by_overlap.fasta"
)


helper_probe_filtered_set <- Biostrings::DNAStringSet(
  as.character(
    helper_probes_filtered_by_overlap$helper_probe
  )
)


names(helper_probe_filtered_set) <- as.character(
  helper_probes_filtered_by_overlap$helper_id
)


Biostrings::writeXStringSet(
  helper_probe_filtered_set,
  filepath = helper_probes_filtered_by_overlap_fasta
)


message(
  "[info] Helper probes written for BLAST filtering: ",
  length(helper_probe_filtered_set)
)


# ============================================================
# 16. BLAST helper probes against primary probes
# ============================================================

# Helper probes are searched against the selected primary
# probes to remove sequences with potentially problematic
# complementarity.

message("============================================================")
message("[step 6] Running helper-to-primary BLASTN")
message("============================================================")


helper_to_primary_probe_db_blast_res <- run_blastn(
  blastn          = blastn,
  query_blast     = helper_probes_filtered_by_overlap_fasta,
  db_out          = primary_probe_db,
  out_dir         = blast_dir,
  blast_tag       = "helper_to_primary_probe_db",
  word_size       = 5,
  evalue          = "1",
  outfmt          = 5,
  max_hsps        = 1,
  max_target_seqs = 1000,
  strand          = "both",
  overwrite       = TRUE
)


# ============================================================
# 17. Parse and filter helper-probe BLAST hits
# ============================================================

message("============================================================")
message("[step 7] Filtering helper probes by BLAST matches")
message("============================================================")


helper_blast_info <- xml_extract(
  helper_to_primary_probe_db_blast_res$blast_out,
  keep_seqs = TRUE
)


message(
  "[info] Parsed helper-probe BLAST hits: ",
  nrow(helper_blast_info)
)


helper_blast_info_filter <- filter_helper_blast_hits(
  result_df      = helper_blast_info,
  max_match      = 18,
  max_gc         = 12,
  max_con_match  = 16,
  max_con_gc     = 5
)


message(
  "[info] BLAST hits passing helper filtering criteria: ",
  nrow(helper_blast_info_filter)
)


helper_probes_final_filter <- finalize_helper_probes_by_blast(
  helper_probes           = helper_probes_filtered_by_overlap,
  helper_blast_info       = helper_blast_info,
  helper_blast_info_filter = helper_blast_info_filter,
  helper_id_col           = "helper_id",
  blast_query_col         = "query_def"
)


message(
  "[info] Helper probes before BLAST filtering: ",
  nrow(helper_probes_filtered_by_overlap)
)


message(
  "[info] Helper probes with BLAST hits: ",
  dplyr::n_distinct(
    helper_blast_info$query_def
  )
)


message(
  "[info] Helper probes retained after BLAST filtering: ",
  nrow(helper_probes_final_filter)
)


message(
  "[info] Helper probes removed by BLAST filtering: ",
  nrow(helper_probes_filtered_by_overlap) -
    nrow(helper_probes_final_filter)
)


# ============================================================
# 18. Deduplicate final helper probes
# ============================================================

message("============================================================")
message("[step 8] Deduplicating helper probes")
message("============================================================")


helper_probes_final_dedup <- deduplicate_helper_probes(
  helper_probes        = helper_probes_final_filter,
  sequence_col         = "helper_probe",
  helper_id_col        = "helper_id",
  parent_probe_id_col  = "parent_probe_id",
  identifier_col       = "identifier"
)


message(
  "[info] Helper probes before deduplication: ",
  nrow(helper_probes_final_filter)
)


message(
  "[info] Helper probes after deduplication: ",
  nrow(helper_probes_final_dedup)
)


message(
  "[info] Duplicate helper sequences removed: ",
  nrow(helper_probes_final_filter) -
    nrow(helper_probes_final_dedup)
)


# ============================================================
# 19. Prepare final helper probe table
# ============================================================

required_final_cols <- c(
  "helper_id",
  "helper_probe"
)


missing_final_cols <- setdiff(
  required_final_cols,
  colnames(helper_probes_final_dedup)
)


if (length(missing_final_cols) > 0) {
  
  stop(
    "Final helper probe table is missing required column(s): ",
    paste(
      missing_final_cols,
      collapse = ", "
    )
  )
}


helper_probes_final <- helper_probes_final_dedup %>%
  dplyr::mutate(
    helper_id = as.character(
      .data$helper_id
    ),
    helper_probe = as.character(
      .data$helper_probe
    )
  ) %>%
  dplyr::filter(
    !is.na(
      .data$helper_id
    ),
    .data$helper_id != "",
    !is.na(
      .data$helper_probe
    ),
    .data$helper_probe != ""
  )


# Sort final helper probes when the expected annotation
# columns are available.

if (all(
  c(
    "identifier",
    "primary_start",
    "helper_side"
  ) %in% colnames(helper_probes_final)
)) {
  
  helper_probes_final <- helper_probes_final %>%
    dplyr::arrange(
      .data$identifier,
      .data$primary_start,
      .data$helper_side
    )
}


message(
  "[info] Final helper probes: ",
  nrow(helper_probes_final)
)


# ============================================================
# 20. Save final outputs
# ============================================================

message("============================================================")
message("[step 9] Saving final helper probes")
message("============================================================")


readr::write_csv(
  helper_probes_final,
  helper_probes_final_csv
)


save(
  helper_probes_final,
  file = helper_probes_final_rdata
)


helper_probes_final_set <- Biostrings::DNAStringSet(
  as.character(
    helper_probes_final$helper_probe
  )
)


names(helper_probes_final_set) <- as.character(
  helper_probes_final$helper_id
)


Biostrings::writeXStringSet(
  helper_probes_final_set,
  filepath = helper_probes_final_fasta
)


message("============================================================")
message("[done] Helper probe design completed.")
message("[output] Primary-probe BLAST database: ", primary_probe_db_dir)
message("[output] BLAST results: ", blast_dir)
message("[output] Final helper probe CSV: ", helper_probes_final_csv)
message("[output] Final helper probe RData: ", helper_probes_final_rdata)
message("[output] Final helper probe FASTA: ", helper_probes_final_fasta)
message("[info] Number of final helper probes: ", length(helper_probes_final_set))
message("============================================================")