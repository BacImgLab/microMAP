# Finalize helper probes based on BLAST filtering results.
# Helpers with no BLAST hits are retained, as are helpers whose BLAST hits
# pass the specificity filter. Helpers that fail BLAST filtering are removed.

finalize_helper_probes_by_blast <- function(
    helper_probes,
    helper_blast_info,
    helper_blast_info_filter,
    helper_id_col = "helper_id",
    blast_query_col = "query_def"
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "finalize_helper_probes_by_blast requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(helper_probes)) {
    stop(
      "'helper_probes' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(helper_blast_info)) {
    stop(
      "'helper_blast_info' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(helper_blast_info_filter)) {
    stop(
      "'helper_blast_info_filter' must be a data.frame or tibble."
    )
  }
  
  if (!helper_id_col %in% colnames(helper_probes)) {
    stop(
      "Missing required column in 'helper_probes': ",
      helper_id_col
    )
  }
  
  if (!blast_query_col %in% colnames(helper_blast_info)) {
    stop(
      "Missing required column in 'helper_blast_info': ",
      blast_query_col
    )
  }
  
  if (!blast_query_col %in% colnames(helper_blast_info_filter)) {
    stop(
      "Missing required column in 'helper_blast_info_filter': ",
      blast_query_col
    )
  }
  
  # Identify helper probes with any BLAST hit
  helper_ids_with_blast_hit <- unique(
    as.character(
      helper_blast_info[[blast_query_col]]
    )
  )
  
  # Identify helper probes whose BLAST hits passed filtering
  helper_ids_pass_blast_filter <- unique(
    as.character(
      helper_blast_info_filter[[blast_query_col]]
    )
  )
  
  # Assign BLAST filtering status to each helper probe
  helper_probes_final_filter <- helper_probes %>%
    dplyr::mutate(
      blast_filter_status = dplyr::case_when(
        !(.data[[helper_id_col]] %in%
            helper_ids_with_blast_hit) ~
          "no_blast_hit",
        
        .data[[helper_id_col]] %in%
          helper_ids_pass_blast_filter ~
          "pass_blast_filter",
        
        TRUE ~
          "fail_blast_filter"
      )
    ) %>%
    
    # Retain helpers with no BLAST hits or acceptable BLAST hits
    dplyr::filter(
      blast_filter_status %in% c(
        "no_blast_hit",
        "pass_blast_filter"
      )
    )
  
  return(helper_probes_final_filter)
}
