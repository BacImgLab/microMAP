# Mark primary probes that are too close to neighboring primary probes.
# Adjacent probes are evaluated within each reference sequence using a
# distance threshold derived from helper-probe length and interval.

mark_close_primary_probes <- function(
    fin_primary_probes,
    helper_length = 30,
    interval = 3
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "mark_close_primary_probes requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(fin_primary_probes)) {
    stop(
      "'fin_primary_probes' must be a data.frame or tibble."
    )
  }
  
  required_cols <- c(
    "probe_id",
    "identifier",
    "start",
    ".end"
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(fin_primary_probes)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'fin_primary_probes': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Check helper-probe parameters
  if (
    length(helper_length) != 1 ||
    !is.numeric(helper_length) ||
    is.na(helper_length) ||
    helper_length <= 0
  ) {
    stop(
      "'helper_length' must be a single numeric value > 0."
    )
  }
  
  if (
    length(interval) != 1 ||
    !is.numeric(interval) ||
    is.na(interval) ||
    interval < 0
  ) {
    stop(
      "'interval' must be a single numeric value >= 0."
    )
  }
  
  helper_length <- as.integer(
    helper_length
  )
  
  interval <- as.integer(
    interval
  )
  
  # Define the proximity threshold used to flag neighboring primary probes
  proximity_threshold <- helper_length + 2 * interval
  
  # Sort primary probes by position and evaluate adjacent probes
  fin_primary_probes_marked <- fin_primary_probes %>%
    dplyr::arrange(
      identifier,
      start
    ) %>%
    dplyr::group_by(
      identifier
    ) %>%
    dplyr::mutate(
      
      # Information for the next primary probe
      next_probe_id = dplyr::lead(
        probe_id
      ),
      
      next_start = dplyr::lead(
        start
      ),
      
      next_end = dplyr::lead(
        .end
      ),
      
      # Number of unoccupied bases between adjacent primary probes
      distance_to_next =
        next_start - .end - 1,
      
      # Flag probes that are too close to the next probe
      too_close_to_next =
        !is.na(distance_to_next) &
        distance_to_next < proximity_threshold,
      
      # Propagate the same relationship to the following probe
      too_close_to_prev = dplyr::lag(
        too_close_to_next,
        default = FALSE
      ),
      
      distance_to_prev = dplyr::lag(
        distance_to_next,
        default = NA_integer_
      ),
      
      # Mark probes that are too close to either neighboring probe
      helper_overlap_flag =
        too_close_to_next |
        too_close_to_prev,
      
      proximity_threshold =
        proximity_threshold
    ) %>%
    dplyr::ungroup()
  
  return(fin_primary_probes_marked)
}
