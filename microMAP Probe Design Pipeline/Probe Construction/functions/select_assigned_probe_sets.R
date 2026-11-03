# Select the final probe set for each genus after code assignment.
# For each assigned genus-code pair, the available combination with the
# smallest combination number is selected as the final probe configuration.

select_assigned_probe_sets <- function(
    assigned_codes,
    available_probe_set,
    combination_col = "combination_id"
) {
  
  # Check required packages
  required_packages <- c(
    "dplyr",
    "stringr"
  )
  
  missing_packages <- required_packages[
    !vapply(
      required_packages,
      requireNamespace,
      quietly = TRUE,
      FUN.VALUE = logical(1)
    )
  ]
  
  if (length(missing_packages) > 0) {
    stop(
      "Missing required package(s): ",
      paste(missing_packages, collapse = ", ")
    )
  }
  
  # Check input data
  if (!is.data.frame(assigned_codes)) {
    stop(
      "'assigned_codes' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(available_probe_set)) {
    stop(
      "'available_probe_set' must be a data.frame or tibble."
    )
  }
  
  required_assigned_cols <- c(
    "target_genus",
    "assigned_code"
  )
  
  required_probe_cols <- c(
    "target_genus",
    "code",
    combination_col,
    "probe_name_info",
    "probe_seq"
  )
  
  missing_assigned_cols <- setdiff(
    required_assigned_cols,
    colnames(assigned_codes)
  )
  
  missing_probe_cols <- setdiff(
    required_probe_cols,
    colnames(available_probe_set)
  )
  
  if (length(missing_assigned_cols) > 0) {
    stop(
      "Missing required column(s) in 'assigned_codes': ",
      paste(missing_assigned_cols, collapse = ", ")
    )
  }
  
  if (length(missing_probe_cols) > 0) {
    stop(
      "Missing required column(s) in 'available_probe_set': ",
      paste(missing_probe_cols, collapse = ", ")
    )
  }
  
  final_probes <- vector(
    "list",
    nrow(assigned_codes)
  )
  
  # Select one probe combination for each assigned genus-code pair
  for (i in seq_len(nrow(assigned_codes))) {
    
    current_genus <- assigned_codes$target_genus[i]
    current_code <- assigned_codes$assigned_code[i]
    
    subset_probes <- available_probe_set %>%
      dplyr::filter(
        target_genus == current_genus,
        code == current_code
      ) %>%
      dplyr::mutate(
        combination_number = as.numeric(
          stringr::str_extract(
            .data[[combination_col]],
            "\\d+"
          )
        )
      ) %>%
      dplyr::arrange(
        combination_number,
        .data[[combination_col]],
        probe_name_info
      )
    
    if (nrow(subset_probes) == 0) {
      stop(
        "No available probes were found for genus = ",
        current_genus,
        ", code = ",
        current_code
      )
    }
    
    # Select the available combination with the smallest numeric ID
    best_combination_id <- subset_probes[[combination_col]][1]
    
    best_combination_probes <- subset_probes %>%
      dplyr::filter(
        .data[[combination_col]] == best_combination_id
      ) %>%
      dplyr::select(
        -dplyr::any_of(
          "combination_number"
        )
      )
    
    final_probes[[i]] <- best_combination_probes
  }
  
  final_probes <- dplyr::bind_rows(
    final_probes
  )
  
  return(final_probes)
}