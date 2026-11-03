# Filter helper probes based on BLAST matching statistics.
# A helper probe is retained only when all of its BLAST hits satisfy
# the specified limits for total, GC, and consecutive matches.

filter_helper_blast_hits <- function(
    result_df,
    max_match = 18,
    max_gc = 12,
    max_con_match = 16,
    max_con_gc = 5
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "filter_helper_blast_hits requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(result_df)) {
    stop(
      "'result_df' must be a data.frame or tibble."
    )
  }
  
  required_cols <- c(
    "query_def",
    "hit_title",
    "match_count",
    "matched_gc_count",
    "consecutive_match_count",
    "consecutive_gc_count"
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(result_df)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'result_df': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Retain a helper probe only when all BLAST hits
  # satisfy the specified matching thresholds.
  filtered_df <- result_df %>%
    dplyr::group_by(
      query_def
    ) %>%
    dplyr::filter(
      all(
        match_count <= max_match &
          matched_gc_count <= max_gc &
          consecutive_match_count < max_con_match &
          consecutive_gc_count < max_con_gc
      )
    ) %>%
    dplyr::ungroup()
  
  return(filtered_df)
}
