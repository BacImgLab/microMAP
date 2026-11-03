###############################################
# main_combine_probes.R
#
# Step 3: Combine primary probes and assign codes
# ==============================================================================
# Combined Probe Construction and Code Assignment Pipeline
# ==============================================================================
#
# Purpose:
#     Constructs barcode-encoded combined probes from final primary probes
#     and assigns valid error-correcting codes to target genera.
#
# Core Method:
#     1. Adds short sequence-derived spacers adjacent to each primary probe
#        for separation between primary and readout probe regions.
#     2. Duplicates primary probes into paired probe-combination groups.
#     3. Maps non-zero codebook entries to secondary/readout probes.
#     4. Expands each primary probe across all candidate codes.
#     5. Generates all valid two-group partitions of active codebook readouts
#        and assigns left/right readout positions without fluorophore-specific
#        positional preference.
#     6. Constructs combined probes by placing the primary probe between
#        the corresponding left and right readout sequences.
#     7. Runs BLASTN specificity filtering on candidate combined probes.
#     8. Retains only complete probe-code combinations that pass BLAST
#        filtering.
#     9. Assigns one unique available code to each target genus using a
#        greedy search prioritized initially by abundance rank.
#    10. Selects one valid probe combination for each assigned genus-code pair.
#
# Inputs:
#     - Final primary probes from Step 2
#     - 16S rRNA reference FASTA
#     - Error-correcting codebook
#     - Secondary/readout probe table
#     - Taxon abundance table
#     - Shared 16S BLAST database generated in Step 2
#     - Pipeline configuration file: config/params.json
#
# Execution:
#     Run from the project root directory after Step 2.
#
# Example Usage:
#     Rscript main_combine_probes.R
#
# Outputs:
#     - outputs/03_combine_probes/candidate_combined_probes.fasta
#     - outputs/03_combine_probes/blast/
#     - outputs/03_combine_probes/fin_comb_probes.fasta
#     - outputs/03_combine_probes/fin_assigned_probes_core.csv
#     - outputs/03_combine_probes/fin_grouped_codebook_info.csv
#     - outputs/03_combine_probes/assigned_codes.csv
#     - outputs/03_combine_probes/available_probe_set.csv
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
# Rscript main_combine_probes.R config/params.json

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


# Step 3 output directory

if (is.null(params$step3_combine_probes$output_dir) ||
    !nzchar(params$step3_combine_probes$output_dir)) {
  
  stop(
    "'step3_combine_probes.output_dir' is missing or empty."
  )
}


output_dir <- resolve_path(
  params$step3_combine_probes$output_dir,
  work_dir
)


dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


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
  "calculate_regions.R",
  "add_flank_regions_to_probe.R",
  "duplicate_probe_combinations.R",
  "map_codebook_to_secondary_probes.R",
  "expand_probes_by_code.R",
  "generate_grouped_codebook_combinations.R",
  "assign_left_right_positions.R",
  "combine_probes.R",
  "run_blastn.R",
  "xml_extract.R",
  "blast_match_filter.R",
  "prepare_available_probe_set.R",
  "assign_codes_greedy.R",
  "select_assigned_probe_sets.R"
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
# 5. Initialize input paths
# ============================================================

# Step 2 primary probe output

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


# Global input files and shared BLAST database

required_global_params <- c(
  "codebook_path",
  "sec_probes_path",
  "input_fasta",
  "abundance_file",
  "ref_16S_db"
)


for (x in required_global_params) {
  
  if (is.null(params$global[[x]]) ||
      !nzchar(params$global[[x]])) {
    
    stop(
      "'global.",
      x,
      "' is missing or empty."
    )
  }
}


codebook_path <- resolve_path(
  params$global$codebook_path,
  work_dir
)

sec_probes_path <- resolve_path(
  params$global$sec_probes_path,
  work_dir
)

input_fasta <- resolve_path(
  params$global$input_fasta,
  work_dir
)

abundance_csv <- resolve_path(
  params$global$abundance_file,
  work_dir
)

ref_16S_db_out <- resolve_path(
  params$global$ref_16S_db,
  work_dir
)


# ============================================================
# 6. Initialize external tool paths
# ============================================================

if (is.null(params$tools$blastn) ||
    !nzchar(params$tools$blastn)) {
  
  stop(
    "'tools.blastn' is missing or empty."
  )
}


blastn <- params$tools$blastn


# ============================================================
# 7. Initialize Step 3 output paths
# ============================================================

required_step3_params <- c(
  "candidate_combined_probes_fasta",
  "fin_comb_probes_fasta",
  "fin_assigned_probes_core",
  "fin_grouped_codebook_info",
  "assigned_codes",
  "available_probe_set"
)


for (x in required_step3_params) {
  
  if (is.null(params$step3_combine_probes[[x]]) ||
      !nzchar(params$step3_combine_probes[[x]])) {
    
    stop(
      "'step3_combine_probes.",
      x,
      "' is missing or empty."
    )
  }
}


candidate_combined_probes_fasta <- resolve_path(
  params$step3_combine_probes$candidate_combined_probes_fasta,
  work_dir
)

fin_comb_probes_fasta <- resolve_path(
  params$step3_combine_probes$fin_comb_probes_fasta,
  work_dir
)

fin_assigned_probes_core_csv <- resolve_path(
  params$step3_combine_probes$fin_assigned_probes_core,
  work_dir
)

fin_grouped_codebook_info_csv <- resolve_path(
  params$step3_combine_probes$fin_grouped_codebook_info,
  work_dir
)

assigned_codes_csv <- resolve_path(
  params$step3_combine_probes$assigned_codes,
  work_dir
)

available_probe_set_csv <- resolve_path(
  params$step3_combine_probes$available_probe_set,
  work_dir
)


# ============================================================
# 8. Check required files and BLAST database
# ============================================================

required_files <- c(
  fin_primary_probes_rdata,
  codebook_path,
  sec_probes_path,
  input_fasta,
  abundance_csv,
  blastn
)


for (f in required_files) {
  
  if (!file.exists(f)) {
    
    stop(
      "Required file does not exist: ",
      f
    )
  }
}


# Check the shared nucleotide BLAST database generated in Step 2.

ref_db_files <- paste0(
  ref_16S_db_out,
  c(".nhr", ".nin", ".nsq")
)


if (!all(file.exists(ref_db_files))) {
  
  stop(
    "Shared 16S BLAST database is incomplete: ",
    ref_16S_db_out,
    "\nExpected files:\n",
    paste(
      ref_db_files,
      collapse = "\n"
    )
  )
}


# ============================================================
# 9. Read abundance information
# ============================================================

message("============================================================")
message("[step 1] Reading abundance information")
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
    taxon_name = paste(
      .data$Genus,
      .data$Species
    )
  )


genus_abundance <- abundance_table %>%
  dplyr::group_by(
    .data$Genus
  ) %>%
  dplyr::summarise(
    Genus_Abundance = sum(
      .data$Abundance,
      na.rm = TRUE
    ),
    n_species = dplyr::n_distinct(
      .data$Species
    ),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(
      .data$Genus_Abundance
    ),
    .data$Genus
  ) %>%
  dplyr::mutate(
    rank = dplyr::dense_rank(
      dplyr::desc(
        .data$Genus_Abundance
      )
    )
  )


# ============================================================
# 10. Read and prepare secondary probes
# ============================================================

message("============================================================")
message("[step 2] Reading secondary probes and codebook")
message("============================================================")


sec_probes_df <- read.csv(
  sec_probes_path,
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)


required_sec_columns <- c(
  "Name",
  "Sequence",
  "modified"
)


missing_sec_columns <- setdiff(
  required_sec_columns,
  colnames(sec_probes_df)
)


if (length(missing_sec_columns) > 0) {
  
  stop(
    "The secondary probe table is missing required column(s): ",
    paste(
      missing_sec_columns,
      collapse = ", "
    )
  )
}


sec_probes_df$rc_sequence <- as.character(
  Biostrings::reverseComplement(
    Biostrings::DNAStringSet(
      sec_probes_df$Sequence
    )
  )
)

sec_mapping_table <- c(
  "5`Alexa Fluor 488" = "1",
  "5`ATTO 490LS"      = "2",
  "5`Cy3"             = "3",
  "5`Alexa Fluor 594" = "4",
  "5`Cy5"             = "5"
)


sec_probes_df <- sec_probes_df %>%
  dplyr::mutate(
    modified_mapped = sec_mapping_table[
      .data$modified
    ]
  ) %>%
  dplyr::group_by(
    .data$modified_mapped
  ) %>%
  dplyr::mutate(
    round = paste0(
      "Round",
      dplyr::row_number()
    )
  ) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    matching_column = paste(
      .data$modified_mapped,
      .data$round,
      sep = "_"
    )
  )


if (any(is.na(sec_probes_df$modified_mapped))) {
  
  print(
    sec_probes_df %>%
      dplyr::filter(
        is.na(
          .data$modified_mapped
        )
      )
  )
  
  stop(
    "Some secondary probes contain unrecognized fluorophore labels."
  )
}


# ============================================================
# 11. Read codebook and target rRNA sequences
# ============================================================

codebook_df <- read.csv(
  codebook_path,
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)


if (!"code" %in% colnames(codebook_df)) {
  
  stop(
    "The codebook does not contain the required 'code' column."
  )
}


rRNA_seqs <- Biostrings::readDNAStringSet(
  filepath = input_fasta
)


# ============================================================
# 12. Load final primary probes from Step 2
# ============================================================

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


end_col <- if ("end" %in% colnames(fin_primary_probes)) {
  
  "end"
  
} else if (".end" %in% colnames(fin_primary_probes)) {
  
  ".end"
  
} else {
  
  stop(
    "'fin_primary_probes' contains neither an 'end' nor '.end' column."
  )
}


pri_probe_info <- fin_primary_probes %>%
  dplyr::transmute(
    probe_id = as.character(
      .data$probe_id
    ),
    probe = as.character(
      .data$probe[, 1]
    ),
    efficiency = as.numeric(
      .data$efficiency[, 1]
    ),
    identifier = as.character(
      .data$identifier
    ),
    Genus = if ("Genus" %in% colnames(fin_primary_probes)) {
      as.character(
        .data$Genus
      )
    } else {
      sub(
        "\\s+.*$",
        "",
        as.character(
          .data$identifier
        )
      )
    },
    start = as.integer(
      .data$start
    ),
    end = as.integer(
      .data[[end_col]]
    )
  )


# ============================================================
# 13. Add short target-derived flanking regions
# ============================================================

message("============================================================")
message("[step 3] Generating candidate combined probes")
message("============================================================")


pri_probe_info_region <- calculate_regions(
  pri_probe_info = pri_probe_info,
  seq_df         = rRNA_seqs,
  id_col         = "identifier",
  flank_length   = 2,
  interval       = 0
)

pri_probes <- add_flank_regions_to_probe(
  pri_probes        = pri_probe_info_region,
  probe_col         = "probe",
  start_region_col  = "start_region",
  end_region_col    = "end_region",
  left_tag          = "left_",
  right_tag         = "_right",
  remove_region_cols   = TRUE,
  remove_position_cols = TRUE
)


# ============================================================
# 14. Generate primary probe combinations
# ============================================================

pri_probes <- duplicate_probe_combinations(
  pri_probes     = pri_probes,
  group_ids      = c("1", "2"),
  probe_name_col = "probe_id",
  genus_col      = "Genus"
)


# ============================================================
# 15. Map codebook entries to secondary probes
# ============================================================

codebook_long <- map_codebook_to_secondary_probes(
  codebook_df    = codebook_df,
  sec_probes_df  = sec_probes_df,
  code_col       = "code",
  round_prefix   = "Round",
  sec_match_col  = "matching_column",
  keep_sec_cols  = c(
    "Name",
    "rc_sequence",
    "modified_mapped"
  )
)


if (any(is.na(codebook_long$rc_sequence))) {
  
  print(
    codebook_long %>%
      dplyr::filter(
        is.na(
          .data$rc_sequence
        )
      ) %>%
      dplyr::select(
        code,
        round,
        color,
        sec_matching_column,
        matching_column
      )
  )
  
  stop(
    "Some codebook readouts could not be matched to secondary probes."
  )
}


# ============================================================
# 16. Expand primary probes by all available codes
# ============================================================

pri_probes <- expand_probes_by_code(
  pri_probes  = pri_probes,
  codebook_df = codebook_df,
  code_col    = "code"
)


# ============================================================
# 17. Generate grouped codebook combinations
# ============================================================

grouped_codebook_long <- generate_grouped_codebook_combinations(
  codebook_long = codebook_long,
  code_col      = "code",
  group_labels  = c(
    "group1",
    "group2"
  )
)


# Assign one readout to each side of a probe combination.
#
# The left/right assignment is intentionally independent of
# fluorophore identity. This avoids imposing a fluorophore-
# specific positional preference on the general example
# pipeline.

grouped_codebook_long <- assign_left_right_positions(
  grouped_codebook_long = grouped_codebook_long
)


check_grouped_codebook <- grouped_codebook_long %>%
  dplyr::count(
    .data$code,
    .data$combination_id,
    .data$group,
    .data$position
  ) %>%
  dplyr::filter(
    .data$n != 1
  )


if (nrow(check_grouped_codebook) > 0) {
  
  print(check_grouped_codebook)
  
  stop(
    "Invalid left/right assignment was detected in grouped_codebook_long."
  )
}


# ============================================================
# 18. Combine primary probes with readout flanks
# ============================================================

combined_probes <- combine_probes(
  pri_probes             = pri_probes,
  grouped_codebook_long  = grouped_codebook_long,
  group_cols             = c(
    "rc_sequence"
  )
) %>%
  dplyr::rename(
    combined_probe = combined_rc_sequence
  ) %>%
  dplyr::mutate(
    combined_probe = paste0(
      "A",
      .data$combined_probe,
      "A"
    )
  )


if (any(is.na(combined_probes$combined_probe)) ||
    any(combined_probes$combined_probe == "")) {
  
  stop(
    "NA or empty sequences were found in combined_probes."
  )
}


# ============================================================
# 19. Write candidate combined probes to FASTA
# ============================================================

combined_probes_fasta <- Biostrings::DNAStringSet(
  combined_probes$combined_probe
)


names(combined_probes_fasta) <- combined_probes$probe_name_info


Biostrings::writeXStringSet(
  combined_probes_fasta,
  filepath = candidate_combined_probes_fasta
)


message(
  "[info] Candidate combined probes: ",
  length(combined_probes_fasta)
)


# ============================================================
# 20. Run BLASTN for candidate combined probes
# ============================================================

message("============================================================")
message("[step 4] Running BLASTN for combined probes")
message("============================================================")


combined_blast_result <- run_blastn(
  blastn          = blastn,
  query_blast     = candidate_combined_probes_fasta,
  db_out          = ref_16S_db_out,
  out_dir         = blast_dir,
  blast_tag       = "candidate_combined",
  word_size       = 5,
  evalue          = "1e-2",
  strand          = "minus",
  outfmt          = 5,
  max_hsps        = 1,
  max_target_seqs = 1000
)


combined_blast_info <- xml_extract(
  combined_blast_result$blast_out,
  keep_seqs = TRUE
)


# ============================================================
# 21. Filter combined probes by off-target GC matching
# ============================================================

message("============================================================")
message("[step 5] Filtering combined probe BLAST matches")
message("============================================================")


combined_blast_filtered_df <- blast_match_filter(
  result_df                = combined_blast_info,
  max_gc                   = 14,
  max_con_gc               = 8,
  enforce_strict_self_match = FALSE,
  require_self_hit         = TRUE
)


n_before <- length(
  unique(
    combined_blast_info$query_def
  )
)


n_after <- length(
  unique(
    combined_blast_filtered_df$query_def
  )
)


message(
  "[info] Combined probes before BLAST filter: ",
  n_before
)


message(
  "[info] Combined probes after BLAST filter:  ",
  n_after
)


# ============================================================
# 22. Prepare the available probe set
# ============================================================

available_probe_set <- prepare_available_probe_set(
  blast_filtered_df      = combined_blast_filtered_df,
  combined_probes        = combined_probes,
  query_col              = "query_def",
  probe_name_info_col    = "probe_name_info",
  combined_probe_col     = "combined_probe"
)


if (nrow(available_probe_set) == 0) {
  
  stop(
    "No combined probe combinations remained after BLAST filtering."
  )
}


# ============================================================
# 23. Assign codes by genus abundance rank
# ============================================================

message("============================================================")
message("[step 6] Assigning codes by genus abundance")
message("============================================================")


code_combination_summary <- available_probe_set %>%
  dplyr::group_by(
    .data$target_genus
  ) %>%
  dplyr::summarise(
    num_codes = dplyr::n_distinct(
      .data$code
    ),
    .groups = "drop"
  )


assigned_codes <- assign_codes_greedy(
  code_combination_summary = code_combination_summary,
  selected_probes          = available_probe_set,
  abundance_table          = genus_abundance,
  genus_col                = "Genus",
  rank_col                 = "rank",
  max_tries                = 1000,
  seed                     = 123
)


# ============================================================
# 24. Select final assigned probe sets
# ============================================================

assigned_probes_core <- select_assigned_probe_sets(
  assigned_codes       = assigned_codes,
  available_probe_set  = available_probe_set
) %>%
  dplyr::ungroup() %>%
  dplyr::select(
    probe_name_info,
    target_genus,
    probe_seq,
    code,
    combination_id,
    probe_comb_id
  )


if (any(is.na(assigned_probes_core$probe_seq))) {
  
  stop(
    "NA probe sequences were found in assigned_probes_core."
  )
}


if (any(assigned_probes_core$probe_seq == "")) {
  
  stop(
    "Empty probe sequences were found in assigned_probes_core."
  )
}


dup_assigned <- assigned_probes_core %>%
  dplyr::count(
    .data$probe_name_info
  ) %>%
  dplyr::filter(
    .data$n > 1
  )


if (nrow(dup_assigned) > 0) {
  
  print(dup_assigned)
  
  stop(
    "Duplicate probe_name_info values were found in assigned_probes_core."
  )
}


# ============================================================
# 25. Extract final codebook and readout information
# ============================================================

final_code_combination <- assigned_probes_core %>%
  dplyr::select(
    target_genus,
    code,
    combination_id
  ) %>%
  dplyr::distinct()


final_grouped_codebook_info <- grouped_codebook_long %>%
  dplyr::inner_join(
    final_code_combination,
    by = c(
      "code",
      "combination_id"
    ),
    relationship = "many-to-many"
  ) %>%
  dplyr::arrange(
    .data$target_genus,
    .data$code,
    .data$combination_id,
    .data$group,
    .data$position,
    .data$round
  )


# ============================================================
# 26. Save final outputs
# ============================================================

fin_comb_probes <- Biostrings::DNAStringSet(
  assigned_probes_core$probe_seq
)


names(fin_comb_probes) <- assigned_probes_core$probe_name_info


Biostrings::writeXStringSet(
  fin_comb_probes,
  filepath = fin_comb_probes_fasta
)


readr::write_csv(
  assigned_probes_core,
  fin_assigned_probes_core_csv
)


readr::write_csv(
  final_grouped_codebook_info,
  fin_grouped_codebook_info_csv
)


readr::write_csv(
  assigned_codes,
  assigned_codes_csv
)


readr::write_csv(
  available_probe_set,
  available_probe_set_csv
)


message("============================================================")
message("[done] Probe combination and code assignment completed.")
message("[output] Candidate combined probes: ", candidate_combined_probes_fasta)
message("[output] BLAST results: ", blast_dir)
message("[output] Final combined probes: ", fin_comb_probes_fasta)
message("[output] Assigned probe table: ", fin_assigned_probes_core_csv)
message("[output] Final codebook information: ", fin_grouped_codebook_info_csv)
message("[output] Assigned codes: ", assigned_codes_csv)
message("[output] Available probe set: ", available_probe_set_csv)
message("[info] Number of final combined probes: ", length(fin_comb_probes))
message(
  "[info] Number of assigned genera: ",
  dplyr::n_distinct(
    assigned_probes_core$target_genus
  )
)
message(
  "[info] Number of final code-combination groups: ",
  nrow(final_code_combination)
)
message("============================================================")

