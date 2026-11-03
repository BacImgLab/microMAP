# Prepare the set of combined probes that pass BLAST specificity filtering.
# Only complete probe sets are retained, requiring both probe-combination
# groups and the expected number of probes for each genus-code combination.

prepare_available_probe_set <- function(
    blast_filtered_df,
    combined_probes,
    query_col = "query_def",
    probe_name_info_col = "probe_name_info",
    combined_probe_col = "combined_probe",
    probe_comb_col = "probe_comb_id"
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "prepare_available_probe_set requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(blast_filtered_df)) {
    stop(
      "'blast_filtered_df' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(combined_probes)) {
    stop(
      "'combined_probes' must be a data.frame or tibble."
    )
  }
  
  required_blast_cols <- c(
    query_col
  )
  
  required_probe_cols <- c(
    "identifier",
    "code",
    "combination_id",
    probe_comb_col,
    probe_name_info_col,
    combined_probe_col,
    "probe_count"
  )
  
  missing_blast_cols <- setdiff(
    required_blast_cols,
    colnames(blast_filtered_df)
  )
  
  missing_probe_cols <- setdiff(
    required_probe_cols,
    colnames(combined_probes)
  )
  
  if (length(missing_blast_cols) > 0) {
    stop(
      "Missing required column(s) in 'blast_filtered_df': ",
      paste(missing_blast_cols, collapse = ", ")
    )
  }
  
  if (length(missing_probe_cols) > 0) {
    stop(
      "Missing required column(s) in 'combined_probes': ",
      paste(missing_probe_cols, collapse = ", ")
    )
  }
  
  # Get combined probes that passed BLAST filtering
  passed_probe_names <- unique(
    as.character(
      blast_filtered_df[[query_col]]
    )
  )
  
  # Retain BLAST-passed combined probes and standardize key information
  available_probe_set <- combined_probes %>%
    dplyr::filter(
      .data[[probe_name_info_col]] %in%
        passed_probe_names
    ) %>%
    dplyr::transmute(
      probe_name_info = as.character(
        .data[[probe_name_info_col]]
      ),
      
      target_genus = sub(
        "\\s+.*$",
        "",
        as.character(identifier)
      ),
      
      probe_seq = as.character(
        .data[[combined_probe_col]]
      ),
      
      probe_count = as.integer(
        probe_count
      ),
      
      probe_comb_id = as.integer(
        .data[[probe_comb_col]]
      ),
      
      code = as.character(
        code
      ),
      
      combination_id = as.character(
        combination_id
      )
    ) %>%
    dplyr::distinct()
  
  # Require both probe-combination groups for each
  # genus-code-combination candidate.
  available_probe_set <- available_probe_set %>%
    dplyr::group_by(
      target_genus,
      code,
      combination_id
    ) %>%
    dplyr::filter(
      any(
        probe_comb_id == 1,
        na.rm = TRUE
      ) &
        any(
          probe_comb_id == 2,
          na.rm = TRUE
        )
    ) %>%
    dplyr::ungroup()
  
  # Count the number of BLAST-passed probes in each complete candidate set
  available_probe_set <- available_probe_set %>%
    dplyr::group_by(
      target_genus,
      code,
      combination_id
    ) %>%
    dplyr::mutate(
      group_probe_count = dplyr::n()
    ) %>%
    dplyr::ungroup()
  
  # Retain only complete probe sets.
  # The number of BLAST-passed probes must equal the expected probe_count.
  available_probe_set <- available_probe_set %>%
    dplyr::group_by(
      target_genus,
      code,
      combination_id
    ) %>%
    dplyr::filter(
      all(
        group_probe_count == probe_count
      )
    ) %>%
    dplyr::ungroup()
  
  return(available_probe_set)
}
