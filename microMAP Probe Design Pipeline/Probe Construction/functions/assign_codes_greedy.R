# Assign one unique code to each target genus using a greedy search.
# Genera are initially prioritized by abundance rank, and repeated attempts
# with alternative genus orders are used to avoid local assignment failures.

assign_codes_greedy <- function(
    code_combination_summary,
    selected_probes,
    abundance_table,
    genus_col = "Genus",
    rank_col = "rank",
    max_tries = 1000,
    seed = 123
) {
  
  # Check required packages
  required_packages <- c(
    "dplyr",
    "tibble",
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
  if (!is.data.frame(code_combination_summary)) {
    stop(
      "'code_combination_summary' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(selected_probes)) {
    stop(
      "'selected_probes' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(abundance_table)) {
    stop(
      "'abundance_table' must be a data.frame or tibble."
    )
  }
  
  required_summary_cols <- c(
    "target_genus"
  )
  
  required_probe_cols <- c(
    "target_genus",
    "code"
  )
  
  required_abundance_cols <- c(
    genus_col,
    rank_col
  )
  
  missing_summary_cols <- setdiff(
    required_summary_cols,
    colnames(code_combination_summary)
  )
  
  missing_probe_cols <- setdiff(
    required_probe_cols,
    colnames(selected_probes)
  )
  
  missing_abundance_cols <- setdiff(
    required_abundance_cols,
    colnames(abundance_table)
  )
  
  if (length(missing_summary_cols) > 0) {
    stop(
      "Missing required column(s) in 'code_combination_summary': ",
      paste(missing_summary_cols, collapse = ", ")
    )
  }
  
  if (length(missing_probe_cols) > 0) {
    stop(
      "Missing required column(s) in 'selected_probes': ",
      paste(missing_probe_cols, collapse = ", ")
    )
  }
  
  if (length(missing_abundance_cols) > 0) {
    stop(
      "Missing required column(s) in 'abundance_table': ",
      paste(missing_abundance_cols, collapse = ", ")
    )
  }
  
  # Check search parameters
  if (
    length(max_tries) != 1 ||
    !is.numeric(max_tries) ||
    is.na(max_tries) ||
    max_tries < 1
  ) {
    stop(
      "'max_tries' must be a single numeric value >= 1."
    )
  }
  
  max_tries <- as.integer(
    max_tries
  )
  
  if (
    length(seed) != 1 ||
    !is.numeric(seed) ||
    is.na(seed)
  ) {
    stop(
      "'seed' must be a single numeric value."
    )
  }
  
  set.seed(seed)
  
  # Prepare genus-level abundance rank information
  genus_rank <- abundance_table %>%
    dplyr::transmute(
      target_genus = as.character(
        .data[[genus_col]]
      ),
      rank = as.numeric(
        .data[[rank_col]]
      )
    ) %>%
    dplyr::distinct()
  
  # Order genera by abundance rank for the first assignment attempt
  genus_order <- code_combination_summary %>%
    dplyr::select(
      target_genus
    ) %>%
    dplyr::distinct() %>%
    dplyr::left_join(
      genus_rank,
      by = "target_genus"
    ) %>%
    dplyr::arrange(
      rank,
      target_genus
    ) %>%
    dplyr::pull(
      target_genus
    )
  
  if (length(genus_order) == 0) {
    stop(
      "No target genera were found in 'code_combination_summary'."
    )
  }
  
  # Determine which codes are available for each genus
  available_code_table <- selected_probes %>%
    dplyr::select(
      target_genus,
      code
    ) %>%
    dplyr::distinct() %>%
    dplyr::mutate(
      code_number = as.numeric(
        stringr::str_extract(
          code,
          "\\d+"
        )
      )
    ) %>%
    dplyr::arrange(
      target_genus,
      code_number
    )
  
  # Check that every genus has at least one available code
  missing_available <- setdiff(
    genus_order,
    unique(
      available_code_table$target_genus
    )
  )
  
  if (length(missing_available) > 0) {
    stop(
      "Some genera have no available codes: ",
      paste(missing_available, collapse = ", ")
    )
  }
  
  # Check that the total number of unique codes is sufficient
  all_codes <- unique(
    available_code_table$code
  )
  
  if (length(all_codes) < length(genus_order)) {
    stop(
      "The number of available unique codes is smaller than ",
      "the number of target genera: ",
      length(all_codes),
      " codes for ",
      length(genus_order),
      " genera."
    )
  }
  
  best_assigned <- NULL
  best_n <- -1
  
  # Repeated greedy assignment
  for (try_id in seq_len(max_tries)) {
    
    # Use abundance-ranked order first; randomize later attempts
    # to escape local assignment failures.
    if (try_id == 1) {
      current_order <- genus_order
    } else {
      current_order <- sample(
        genus_order
      )
    }
    
    assigned_codes <- tibble::tibble(
      target_genus = character(),
      assigned_code = character()
    )
    
    success <- TRUE
    
    # Assign the first unused available code to each genus
    for (genus in current_order) {
      
      available_codes <- available_code_table %>%
        dplyr::filter(
          target_genus == genus
        ) %>%
        dplyr::arrange(
          code_number
        ) %>%
        dplyr::pull(
          code
        ) %>%
        unique()
      
      candidate_codes <- setdiff(
        available_codes,
        assigned_codes$assigned_code
      )
      
      if (length(candidate_codes) == 0) {
        success <- FALSE
        break
      }
      
      selected_code <- candidate_codes[1]
      
      assigned_codes <- assigned_codes %>%
        tibble::add_row(
          target_genus = genus,
          assigned_code = selected_code
        )
    }
    
    # Keep the best partial assignment for error reporting
    if (nrow(assigned_codes) > best_n) {
      best_assigned <- assigned_codes
      best_n <- nrow(assigned_codes)
    }
    
    # Return once all genera receive unique codes
    if (
      success &&
      nrow(assigned_codes) == length(genus_order)
    ) {
      
      assigned_codes <- assigned_codes %>%
        dplyr::left_join(
          genus_rank,
          by = "target_genus"
        ) %>%
        dplyr::arrange(
          rank,
          target_genus
        )
      
      return(assigned_codes)
    }
  }
  
  # Report genera that could not be assigned after all attempts
  failed_genus <- setdiff(
    genus_order,
    best_assigned$target_genus
  )
  
  stop(
    "Failed to assign unique codes to all genera within max_tries = ",
    max_tries,
    ".\nAssigned genera: ",
    nrow(best_assigned),
    " / ",
    length(genus_order),
    "\nUnassigned genera: ",
    paste(failed_genus, collapse = ", ")
  )
}
