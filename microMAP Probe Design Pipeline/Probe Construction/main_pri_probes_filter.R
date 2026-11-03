###############################################
# main_pri_probes_filter.R
#
# Step 2: Filter primary probes by BLAST specificity
# ==============================================================================
# Primary Probe Specificity Filtering Pipeline
# ==============================================================================
#
# Purpose:
#     Filters candidate primary probes by BLAST specificity and selects
#     non-overlapping final primary probes for each target.
#
# Core Method:
#     1. Reads candidate primary probes generated in Step 1.
#     2. Builds a local nucleotide BLAST database from the 16S rRNA
#        reference sequences when the database is not already available.
#     3. Runs BLASTN to evaluate candidate probe specificity.
#     4. Parses BLAST XML output and calculates total, GC, and consecutive
#        sequence-matching statistics.
#     5. Retains probes whose off-target hits satisfy the specified matching
#        thresholds while requiring expected same-genus target hits.
#     6. Summarizes the abundance represented by same-genus BLAST hits.
#     7. Selects non-overlapping probes by abundance and probe efficiency,
#        while enforcing the required minimum spacing between probes.
#
# Inputs:
#     - Candidate primary probes from Step 1
#     - 16S rRNA reference FASTA
#     - Taxon abundance table
#     - Pipeline configuration file: config/params.json
#
# Execution:
#     Run from the project root directory after Step 1.
#
# Example Usage:
#     Rscript main_pri_probes_filter.R
#
# Outputs:
#     - outputs/ref_16S_db/
#     - outputs/02_primary_filter/blast/
#     - outputs/02_primary_filter/filter_primary_probes.Rdata
#     - outputs/02_primary_filter/fin_primary_probes.Rdata
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
  library(Biostrings)
  library(XML)
  library(seqinr)
})


# ============================================================
# 1. Locate the script and parameter file
# ============================================================

# Detect the directory containing this script.
# This allows the default configuration file to be located
# without depending on the current working directory.

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
# Rscript main_pri_probes_filter.R config/params.json

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


# Step 2 output directory

if (is.null(params$step2_pri_filter$output_dir) ||
    !nzchar(params$step2_pri_filter$output_dir)) {
  
  stop(
    "'step2_pri_filter.output_dir' is missing or empty."
  )
}


output_dir <- resolve_path(
  params$step2_pri_filter$output_dir,
  work_dir
)


dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# BLAST output generated during primary probe filtering

blast_dir <- file.path(
  output_dir,
  "blast"
)


dir.create(
  blast_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 4. Load required custom functions
# ============================================================

required_function_files <- c(
  "create_blast_database.R",
  "run_blastn.R",
  "xml_extract.R",
  "blast_match_filter.R",
  "dedup_primary_blast_by_query.R",
  "select_nonoverlap_probes_by_priority.R"
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
# 5. Initialize Step 1 input paths
# ============================================================

if (is.null(params$step1_design_probes$candidate_probes) ||
    !nzchar(params$step1_design_probes$candidate_probes)) {
  
  stop(
    "'step1_design_probes.candidate_probes' is missing or empty."
  )
}


if (is.null(params$step1_design_probes$candidate_probes_fasta) ||
    !nzchar(params$step1_design_probes$candidate_probes_fasta)) {
  
  stop(
    "'step1_design_probes.candidate_probes_fasta' is missing or empty."
  )
}


candidate_probes_rdata <- resolve_path(
  params$step1_design_probes$candidate_probes,
  work_dir
)


candidate_probes_fasta <- resolve_path(
  params$step1_design_probes$candidate_probes_fasta,
  work_dir
)


# ============================================================
# 6. Initialize global input paths
# ============================================================

if (is.null(params$global$abundance_file) ||
    !nzchar(params$global$abundance_file)) {
  
  stop(
    "'global.abundance_file' is missing or empty."
  )
}


if (is.null(params$global$input_fasta) ||
    !nzchar(params$global$input_fasta)) {
  
  stop(
    "'global.input_fasta' is missing or empty."
  )
}


abundance_csv <- resolve_path(
  params$global$abundance_file,
  work_dir
)


query_db_fasta <- resolve_path(
  params$global$input_fasta,
  work_dir
)


# ============================================================
# 7. Initialize external tool paths
# ============================================================

if (is.null(params$tools$makeblastdb) ||
    !nzchar(params$tools$makeblastdb)) {
  
  stop(
    "'tools.makeblastdb' is missing or empty."
  )
}


if (is.null(params$tools$blastn) ||
    !nzchar(params$tools$blastn)) {
  
  stop(
    "'tools.blastn' is missing or empty."
  )
}


makeblastdb <- params$tools$makeblastdb
blastn      <- params$tools$blastn


# ============================================================
# 8. Initialize shared BLAST database path
# ============================================================

# This local 16S BLAST database is generated from the input
# FASTA and is reused by multiple downstream pipeline steps.

if (is.null(params$global$ref_16S_db) ||
    !nzchar(params$global$ref_16S_db)) {
  
  stop(
    "'global.ref_16S_db' is missing or empty."
  )
}


ref_16S_db_out <- resolve_path(
  params$global$ref_16S_db,
  work_dir
)


dir.create(
  dirname(ref_16S_db_out),
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 9. Initialize Step 2 output paths
# ============================================================

if (is.null(params$step2_pri_filter$filter_primary_probes) ||
    !nzchar(params$step2_pri_filter$filter_primary_probes)) {
  
  stop(
    "'step2_pri_filter.filter_primary_probes' is missing or empty."
  )
}


if (is.null(params$step2_pri_filter$fin_primary_probes) ||
    !nzchar(params$step2_pri_filter$fin_primary_probes)) {
  
  stop(
    "'step2_pri_filter.fin_primary_probes' is missing or empty."
  )
}


filter_primary_probes_path <- resolve_path(
  params$step2_pri_filter$filter_primary_probes,
  work_dir
)


fin_primary_probes_path <- resolve_path(
  params$step2_pri_filter$fin_primary_probes,
  work_dir
)


dir.create(
  dirname(filter_primary_probes_path),
  recursive = TRUE,
  showWarnings = FALSE
)


dir.create(
  dirname(fin_primary_probes_path),
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 10. Check required files and external tools
# ============================================================

if (!file.exists(candidate_probes_rdata)) {
  
  stop(
    "Candidate probe RData file does not exist: ",
    candidate_probes_rdata
  )
}


if (!file.exists(candidate_probes_fasta)) {
  
  stop(
    "Candidate probe FASTA file does not exist: ",
    candidate_probes_fasta
  )
}


if (!file.exists(abundance_csv)) {
  
  stop(
    "Abundance table does not exist: ",
    abundance_csv
  )
}


if (!file.exists(query_db_fasta)) {
  
  stop(
    "Input FASTA file does not exist: ",
    query_db_fasta
  )
}


if (!file.exists(makeblastdb)) {
  
  stop(
    "makeblastdb executable does not exist: ",
    makeblastdb
  )
}


if (!file.exists(blastn)) {
  
  stop(
    "blastn executable does not exist: ",
    blastn
  )
}


# ============================================================
# 11. Load candidate probes from Step 1
# ============================================================

message("============================================================")
message("[step 1] Loading candidate primary probes")
message("============================================================")


step1_env <- new.env()


load(
  candidate_probes_rdata,
  envir = step1_env
)


if (!exists(
  "candidate_primary_probes_numbering",
  envir = step1_env,
  inherits = FALSE
)) {
  
  stop(
    "Object 'candidate_primary_probes_numbering' was not found in: ",
    candidate_probes_rdata
  )
}


candidate_primary_probes_numbering <- get(
  "candidate_primary_probes_numbering",
  envir = step1_env
)


rm(step1_env)


# ============================================================
# 12. Read the abundance table
# ============================================================

message("============================================================")
message("[step 2] Reading abundance information")
message("============================================================")


abundance_table <- read.csv(
  abundance_csv,
  stringsAsFactors = FALSE
)


required_abundance_columns <- c(
  "Genus",
  "Species",
  "Abundance"
)


missing_columns <- setdiff(
  required_abundance_columns,
  colnames(abundance_table)
)


if (length(missing_columns) > 0) {
  
  stop(
    "The abundance table is missing required column(s): ",
    paste(
      missing_columns,
      collapse = ", "
    )
  )
}


abundance_table <- abundance_table %>%
  dplyr::mutate(
    rank = dplyr::dense_rank(
      dplyr::desc(
        .data$Abundance
      )
    ),
    taxon_name = paste(
      .data$Genus,
      .data$Species
    )
  )


# ============================================================
# 13. Build the shared local 16S BLAST database
# ============================================================

# The database is constructed from the input 16S FASTA and
# associated abundance information.
#
# It is used here for primary probe specificity filtering and
# can also be reused by subsequent pipeline steps.

message("============================================================")
message("[step 3] Building shared local 16S BLAST database")
message("============================================================")


create_blast_database(
  makeblastdb     = makeblastdb,
  db_out          = ref_16S_db_out,
  query_db        = query_db_fasta,
  abundance_table = abundance_table,
  rank_threshold  = NULL,
  min_abundance   = NULL,
  taxon_col       = "taxon_name"
)


# ============================================================
# 14. Run BLASTN for candidate primary probes
# ============================================================

message("============================================================")
message("[step 4] Running BLASTN specificity search")
message("============================================================")


primary_blast_res <- run_blastn(
  blastn      = blastn,
  query_blast = candidate_probes_fasta,
  db_out      = ref_16S_db_out,
  out_dir     = blast_dir,
  blast_tag   = "candidate_primary",
  strand      = "minus"
)


# ============================================================
# 15. Parse BLAST XML output
# ============================================================

primary_blast_info <- xml_extract(
  primary_blast_res$blast_out,
  keep_seqs = TRUE
)


# ============================================================
# 16. Filter BLAST matches
# ============================================================

# Candidate probes are filtered according to the following
# sequence-matching criteria:
#
#   max_match     = 18
#   max_con_match = 16
#   max_con_gc    = 5
#
# A stricter contiguous-match requirement is applied to the
# expected self hit.

message("============================================================")
message("[step 5] Filtering BLAST matches")
message("============================================================")


primary_blast_filtered_df <- blast_match_filter(
  result_df                 = primary_blast_info,
  max_match                 = 18,
  max_con_match             = 16,
  max_con_gc                = 5,
  enforce_strict_self_match = TRUE,
  strict_self_con_match     = 28
)


n_before <- length(
  unique(
    primary_blast_info$query_def
  )
)


n_after <- length(
  unique(
    primary_blast_filtered_df$query_def
  )
)


message(
  "[info] Candidate probes before BLAST filter: ",
  n_before
)


message(
  "[info] Candidate probes after BLAST filter:  ",
  n_after
)


# ============================================================
# 17. Summarize BLAST hits by probe
# ============================================================

# BLAST hits are summarized by query probe.
#
# Abundance information from matched target taxa is used to
# calculate abundance_sum for each probe.

primary_probe_dedup_df <- dedup_primary_blast_by_query(
  blast_filtered_df   = primary_blast_filtered_df,
  abundance_table     = abundance_table,
  abundance_key_col   = "taxon_name",
  abundance_value_col = "Abundance"
)


# ============================================================
# 18. Retain the expected self hit for each probe
# ============================================================

# Probe identifiers contain a final suffix corresponding to
# the probe number.
#
# Removing this suffix recovers the target identifier, which
# is compared with the BLAST hit title.
#
# Only the expected self hit is retained for each query.

primary_blast_filtered_uni_df <- primary_blast_filtered_df %>%
  dplyr::mutate(
    query_base = sub(
      "_[^_]+$",
      "",
      .data$query_def
    )
  ) %>%
  dplyr::filter(
    .data$query_base == .data$hit_title
  ) %>%
  dplyr::group_by(
    .data$query_id
  ) %>%
  dplyr::slice(1) %>%
  dplyr::ungroup() %>%
  dplyr::select(
    -query_base
  )


# ============================================================
# 19. Add abundance information to self-hit records
# ============================================================

primary_blast_filtered_uni_df <- primary_blast_filtered_uni_df %>%
  dplyr::left_join(
    primary_probe_dedup_df %>%
      dplyr::select(
        query_def,
        abundance_sum
      ) %>%
      dplyr::distinct(),
    by = "query_def"
  )


# ============================================================
# 20. Retain candidate probes passing the BLAST filter
# ============================================================

# Candidate probes are retained only when a valid self hit
# remains after BLAST filtering.
#
# abundance_sum and target_genus are transferred from the
# BLAST-filtered results.
#
# Duplicate probe sequences are removed while retaining the
# first occurrence.

filter_primary_probes <- candidate_primary_probes_numbering %>%
  dplyr::left_join(
    primary_blast_filtered_uni_df %>%
      dplyr::select(
        query_def,
        abundance_sum,
        target_genus
      ) %>%
      dplyr::distinct(),
    by = c(
      "probe_id" = "query_def"
    )
  ) %>%
  dplyr::filter(
    !is.na(
      .data$abundance_sum
    )
  ) %>%
  dplyr::relocate(
    probe_id,
    probe,
    efficiency,
    abundance_sum
  ) %>%
  dplyr::distinct(
    probe,
    .keep_all = TRUE
  )


message(
  "[info] Probes passing BLAST specificity filtering: ",
  nrow(filter_primary_probes)
)


# ============================================================
# 21. Select non-overlapping primary probes
# ============================================================

# Probes are selected independently for each target identifier.
#
# Priority is determined lexicographically using:
#
#   1. abundance_sum
#   2. probe efficiency
#
# Selected probes must have a minimum gap of 3 nt.
# At most two probes are retained for each target identifier.

message("============================================================")
message("[step 6] Selecting non-overlapping primary probes")
message("============================================================")


fin_primary_probes <- select_nonoverlap_probes_by_priority(
  probes_df      = filter_primary_probes,
  id_col         = "identifier",
  start_col      = "start",
  len_col        = "len",
  eff_col        = "efficiency",
  abundance_col  = "abundance_sum",
  probe_id_col   = "probe_id",
  min_gap        = 3,
  max_n_per_id   = 2,
  method         = "lexicographic"
)


message(
  "[info] Final number of selected primary probes: ",
  nrow(fin_primary_probes)
)


# ============================================================
# 22. Save results
# ============================================================

save(
  filter_primary_probes,
  file = filter_primary_probes_path
)


save(
  fin_primary_probes,
  file = fin_primary_probes_path
)


message("============================================================")
message("[done] Primary probe specificity filtering completed.")
message("[output] Shared 16S BLAST database: ", ref_16S_db_out)
message("[output] BLAST results: ", blast_dir)
message("[output] Filtered probes: ", filter_primary_probes_path)
message("[output] Final primary probes: ", fin_primary_probes_path)
message("============================================================")
