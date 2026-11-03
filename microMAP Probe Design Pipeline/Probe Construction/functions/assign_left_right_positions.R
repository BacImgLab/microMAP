# Assign left and right positions to the two readouts within each group.
# Readouts are assigned according to their existing order, without imposing
# any fluorophore-specific positional preference.

assign_left_right_positions <- function(
    grouped_codebook_long
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "assign_left_right_positions requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(grouped_codebook_long)) {
    stop(
      "'grouped_codebook_long' must be a data.frame or tibble."
    )
  }
  
  required_cols <- c(
    "code",
    "combination_id",
    "group"
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(grouped_codebook_long)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'grouped_codebook_long': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Assign the first readout in each group to the left position
  # and the second readout to the right position.
  grouped_codebook_long <- grouped_codebook_long %>%
    dplyr::group_by(
      .data$code,
      .data$combination_id,
      .data$group
    ) %>%
    dplyr::mutate(
      valid_group = dplyr::n() == 2,
      position = dplyr::case_when(
        valid_group & dplyr::row_number() == 1 ~ "left",
        valid_group & dplyr::row_number() == 2 ~ "right",
        TRUE ~ NA_character_
      )
    ) %>%
    dplyr::ungroup() %>%
    
    # Keep only combinations in which every group contains exactly two readouts
    dplyr::group_by(
      .data$code,
      .data$combination_id
    ) %>%
    dplyr::filter(
      all(.data$valid_group)
    ) %>%
    dplyr::ungroup() %>%
    
    # Remove temporary validation column
    dplyr::select(
      -valid_group
    )
  
  return(grouped_codebook_long)
}