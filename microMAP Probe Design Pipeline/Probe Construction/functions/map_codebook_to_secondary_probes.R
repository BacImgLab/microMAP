# Map non-zero codebook entries to secondary/readout probes.
# Codebook rounds are converted to long format and matched to secondary
# probes using a color-round identifier.

map_codebook_to_secondary_probes <- function(
    codebook_df,
    sec_probes_df,
    code_col = "code",
    round_prefix = "Round",
    sec_match_col = "matching_column",
    keep_sec_cols = c(
      "Name",
      "rc_sequence",
      "modified_mapped"
    )
) {
  
  # Check required packages
  required_packages <- c(
    "dplyr",
    "tidyr"
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
  if (!is.data.frame(codebook_df)) {
    stop(
      "'codebook_df' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(sec_probes_df)) {
    stop(
      "'sec_probes_df' must be a data.frame or tibble."
    )
  }
  
  if (!code_col %in% colnames(codebook_df)) {
    stop(
      "Missing required column in 'codebook_df': ",
      code_col
    )
  }
  
  if (!sec_match_col %in% colnames(sec_probes_df)) {
    stop(
      "Missing required column in 'sec_probes_df': ",
      sec_match_col
    )
  }
  
  # Identify imaging-round columns in the codebook
  round_cols <- grep(
    paste0("^", round_prefix),
    colnames(codebook_df),
    value = TRUE
  )
  
  if (length(round_cols) == 0) {
    stop(
      "No round columns were found in 'codebook_df' with prefix: ",
      round_prefix
    )
  }
  
  # Retain secondary-probe columns required for mapping
  sec_keep_cols <- unique(
    c(
      sec_match_col,
      keep_sec_cols
    )
  )
  
  sec_keep_cols <- intersect(
    sec_keep_cols,
    colnames(sec_probes_df)
  )
  
  sec_probes_use <- sec_probes_df %>%
    dplyr::select(
      dplyr::all_of(sec_keep_cols)
    )
  
  # Convert the codebook to long format and remove zero entries
  codebook_long <- codebook_df %>%
    tidyr::pivot_longer(
      cols = dplyr::all_of(round_cols),
      names_to = "round",
      values_to = "color"
    ) %>%
    dplyr::mutate(
      code = as.character(
        .data[[code_col]]
      ),
      round = as.character(
        round
      ),
      color = as.character(
        color
      )
    ) %>%
    dplyr::filter(
      color != "0"
    ) %>%
    
    # Generate identifiers used for secondary-probe mapping
    dplyr::mutate(
      sec_matching_column = paste(
        color,
        round,
        sep = "_"
      ),
      matching_column = paste0(
        "r",
        gsub("\\D", "", round),
        "F",
        color
      )
    ) %>%
    
    # Match each codebook entry to its secondary/readout probe
    dplyr::left_join(
      sec_probes_use,
      by = stats::setNames(
        sec_match_col,
        "sec_matching_column"
      )
    )
  
  # Warn when secondary-probe sequences cannot be matched
  if ("rc_sequence" %in% colnames(codebook_long)) {
    
    missing_sec <- codebook_long %>%
      dplyr::filter(
        is.na(rc_sequence)
      )
    
    if (nrow(missing_sec) > 0) {
      warning(
        nrow(missing_sec),
        " codebook entries did not match 'sec_probes_df'. ",
        "Check the secondary-probe matching identifiers."
      )
    }
  }
  
  return(codebook_long)
}