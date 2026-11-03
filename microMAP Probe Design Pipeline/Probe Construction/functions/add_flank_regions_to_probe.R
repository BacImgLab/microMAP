# Add sequence-derived flanking spacers to each primary probe.
# The downstream spacer is added to the left side of the probe, while the
# upstream spacer is added to the right side, following probe orientation.

add_flank_regions_to_probe <- function(
    pri_probes,
    probe_col = "probe",
    start_region_col = "start_region",
    end_region_col = "end_region",
    left_tag = "left_",
    right_tag = "_right",
    remove_region_cols = TRUE,
    remove_position_cols = TRUE
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "add_flank_regions_to_probe requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(pri_probes)) {
    stop(
      "'pri_probes' must be a data.frame or tibble."
    )
  }
  
  required_cols <- c(
    probe_col,
    start_region_col,
    end_region_col
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(pri_probes)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'pri_probes': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Add flanking spacers according to probe orientation:
  # downstream region + primary probe + upstream region.
  pri_probes <- pri_probes %>%
    dplyr::mutate(
      !!probe_col := paste0(
        .data[[end_region_col]],
        .data[[probe_col]],
        .data[[start_region_col]]
      ),
      !!probe_col := paste0(
        left_tag,
        .data[[probe_col]],
        right_tag
      )
    )
  
  # Remove temporary flanking-region columns
  if (isTRUE(remove_region_cols)) {
    
    pri_probes <- pri_probes %>%
      dplyr::select(
        -dplyr::any_of(
          c(
            start_region_col,
            end_region_col
          )
        )
      )
  }
  
  # Remove original probe position columns
  if (isTRUE(remove_position_cols)) {
    
    pri_probes <- pri_probes %>%
      dplyr::select(
        -dplyr::any_of(
          c(
            "start",
            "end"
          )
        )
      )
  }
  
  return(pri_probes)
}
