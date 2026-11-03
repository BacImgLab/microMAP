# Remove helper probes located between primary probes that are too close.
# Overlap information from neighboring primary probes is joined to each
# helper probe, and only the conflicting helper side is removed.

filter_helper_overlaps <- function(
    helper_probes,
    fin_primary_probes_marked
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "filter_helper_overlaps requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(helper_probes)) {
    stop(
      "'helper_probes' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(fin_primary_probes_marked)) {
    stop(
      "'fin_primary_probes_marked' must be a data.frame or tibble."
    )
  }
  
  required_helper_cols <- c(
    "parent_probe_id",
    "helper_id",
    "helper_side",
    "helper_probe"
  )
  
  missing_helper_cols <- setdiff(
    required_helper_cols,
    colnames(helper_probes)
  )
  
  if (length(missing_helper_cols) > 0) {
    stop(
      "Missing required column(s) in 'helper_probes': ",
      paste(missing_helper_cols, collapse = ", ")
    )
  }
  
  required_mark_cols <- c(
    "probe_id",
    "too_close_to_next",
    "too_close_to_prev",
    "helper_overlap_flag",
    "distance_to_next",
    "distance_to_prev",
    "proximity_threshold"
  )
  
  missing_mark_cols <- setdiff(
    required_mark_cols,
    colnames(fin_primary_probes_marked)
  )
  
  if (length(missing_mark_cols) > 0) {
    stop(
      "Missing required column(s) in 'fin_primary_probes_marked': ",
      paste(missing_mark_cols, collapse = ", ")
    )
  }
  
  # Add neighboring-primary-probe information to each helper probe
  helper_probes_marked <- helper_probes %>%
    dplyr::left_join(
      fin_primary_probes_marked %>%
        dplyr::select(
          probe_id,
          too_close_to_next,
          too_close_to_prev,
          helper_overlap_flag,
          distance_to_next,
          distance_to_prev,
          proximity_threshold
        ),
      by = c(
        "parent_probe_id" = "probe_id"
      )
    )
  
  # Check that all helper probes were matched to their parent probes
  if (any(is.na(helper_probes_marked$helper_overlap_flag))) {
    stop(
      "Some helper probes could not be matched to their parent primary probes."
    )
  }
  
  # Mark helper probes located between primary probes that are too close
  helper_probes_marked <- helper_probes_marked %>%
    dplyr::mutate(
      remove_by_overlap = dplyr::case_when(
        helper_side == "right" &
          too_close_to_next ~ TRUE,
        
        helper_side == "left" &
          too_close_to_prev ~ TRUE,
        
        TRUE ~ FALSE
      )
    )
  
  # Retain non-conflicting helper probes
  helper_probes_filtered <- helper_probes_marked %>%
    dplyr::filter(
      !remove_by_overlap
    )
  
  # Record helper probes removed because of neighboring primary probes
  removed_helper_probes <- helper_probes_marked %>%
    dplyr::filter(
      remove_by_overlap
    )
  
  return(
    list(
      helper_probes_filtered = helper_probes_filtered,
      removed_helper_probes = removed_helper_probes,
      helper_probes_marked = helper_probes_marked
    )
  )
}
