# Duplicate each primary probe into multiple probe-combination groups.
# A new probe name is generated for each copy, while the original probe ID
# is retained for downstream tracking and combination assignment.

duplicate_probe_combinations <- function(
    pri_probes,
    group_ids = c("1", "2"),
    probe_name_col = "probe_id",
    genus_col = "Genus"
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "duplicate_probe_combinations requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(pri_probes)) {
    stop(
      "'pri_probes' must be a data.frame or tibble."
    )
  }
  
  required_cols <- c(
    probe_name_col,
    genus_col
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
  
  # Check combination group IDs
  if (
    length(group_ids) == 0 ||
    any(is.na(group_ids)) ||
    any(!nzchar(as.character(group_ids)))
  ) {
    stop(
      "'group_ids' must contain at least one valid group identifier."
    )
  }
  
  group_ids <- as.character(
    group_ids
  )
  
  if (anyDuplicated(group_ids)) {
    stop(
      "'group_ids' must contain unique values."
    )
  }
  
  # Duplicate each primary probe for every combination group
  pri_probes_out <- lapply(
    group_ids,
    function(group_id) {
      
      pri_probes %>%
        dplyr::mutate(
          probe_comb_id = group_id
        )
    }
  ) %>%
    dplyr::bind_rows()
  
  # Generate combination-specific probe names
  pri_probes_out <- pri_probes_out %>%
    dplyr::mutate(
      origin_probe_name = as.character(
        .data[[probe_name_col]]
      ),
      probe_comb_id = as.character(
        probe_comb_id
      ),
      probe_name = paste0(
        origin_probe_name,
        "_probecomb",
        probe_comb_id
      )
    )
  
  # Count duplicated probe entries within each genus
  pri_probes_out <- pri_probes_out %>%
    dplyr::group_by(
      .data[[genus_col]]
    ) %>%
    dplyr::mutate(
      probe_count = dplyr::n()
    ) %>%
    dplyr::ungroup()
  
  # Place genus column first
  pri_probes_out <- pri_probes_out %>%
    dplyr::select(
      dplyr::all_of(genus_col),
      dplyr::everything()
    )
  
  return(pri_probes_out)
}
