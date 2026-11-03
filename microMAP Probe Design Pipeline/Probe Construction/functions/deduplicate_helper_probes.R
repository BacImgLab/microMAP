# Deduplicate helper probes by probe sequence.
# When the same helper sequence is shared by multiple probes or targets,
# the corresponding helper IDs, parent probe IDs, and identifiers are recorded.

deduplicate_helper_probes <- function(
    helper_probes,
    sequence_col = "helper_probe",
    helper_id_col = "helper_id",
    parent_probe_id_col = "parent_probe_id",
    identifier_col = "identifier"
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "deduplicate_helper_probes requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(helper_probes)) {
    stop(
      "'helper_probes' must be a data.frame or tibble."
    )
  }
  
  required_cols <- c(
    sequence_col,
    helper_id_col,
    parent_probe_id_col,
    identifier_col
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(helper_probes)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'helper_probes': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Standardize relevant columns as character vectors
  helper_probes[[sequence_col]] <- as.character(
    helper_probes[[sequence_col]]
  )
  
  helper_probes[[helper_id_col]] <- as.character(
    helper_probes[[helper_id_col]]
  )
  
  helper_probes[[parent_probe_id_col]] <- as.character(
    helper_probes[[parent_probe_id_col]]
  )
  
  helper_probes[[identifier_col]] <- as.character(
    helper_probes[[identifier_col]]
  )
  
  # Remove empty sequences and summarize duplicated helper sequences
  helper_probes_dedup <- helper_probes %>%
    dplyr::filter(
      !is.na(.data[[sequence_col]]),
      .data[[sequence_col]] != ""
    ) %>%
    dplyr::group_by(
      .data[[sequence_col]]
    ) %>%
    dplyr::mutate(
      shared_count = dplyr::n(),
      
      shared_helper_ids = dplyr::if_else(
        shared_count > 1,
        paste(
          unique(.data[[helper_id_col]]),
          collapse = ";"
        ),
        NA_character_
      ),
      
      shared_parent_probe_ids = dplyr::if_else(
        shared_count > 1,
        paste(
          unique(.data[[parent_probe_id_col]]),
          collapse = ";"
        ),
        NA_character_
      ),
      
      shared_identifiers = dplyr::if_else(
        shared_count > 1,
        paste(
          unique(.data[[identifier_col]]),
          collapse = ";"
        ),
        NA_character_
      )
    ) %>%
    dplyr::ungroup() %>%
    
    # Keep one representative row for each unique helper sequence
    dplyr::distinct(
      .data[[sequence_col]],
      .keep_all = TRUE
    )
  
  return(helper_probes_dedup)
}
