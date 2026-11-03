# Assign sequential IDs to probes within each target and export probe sequences
# as FASTA. Reverse-complement target sequences can optionally be exported.

numbering_probes <- function(
    probes_df,
    probes_fasta_path,
    targets_fasta_path = NULL,
    seq_col = "probe",
    id_col = "identifier",
    probe_id_col = "probe_id"
) {
  
  # Check required packages
  required_packages <- c(
    "dplyr",
    "Biostrings"
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
  if (!is.data.frame(probes_df)) {
    stop("'probes_df' must be a data.frame or tibble.")
  }
  
  required_cols <- c(
    id_col,
    seq_col
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(probes_df)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s): ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  if (nrow(probes_df) == 0) {
    stop("No probes were provided for numbering.")
  }
  
  # Assign sequential probe IDs within each target
  probes_numbered <- probes_df %>%
    dplyr::group_by(
      .data[[id_col]]
    ) %>%
    dplyr::mutate(
      !!probe_id_col := paste(
        .data[[id_col]],
        dplyr::row_number(),
        sep = "_"
      )
    ) %>%
    dplyr::ungroup()
  
  # Check probe ID uniqueness
  if (anyDuplicated(probes_numbered[[probe_id_col]])) {
    stop("Generated probe IDs are not unique.")
  }
  
  # Extract probe sequences
  seq_vec <- probes_numbered[[seq_col]]
  
  if (is.matrix(seq_vec) || is.data.frame(seq_vec)) {
    seq_vec <- seq_vec[, 1]
  }
  
  seq_vec <- as.character(seq_vec)
  
  if (any(is.na(seq_vec)) || any(seq_vec == "")) {
    stop("Probe sequences contain NA or empty values.")
  }
  
  # Export probe FASTA
  probes_set <- Biostrings::DNAStringSet(
    seq_vec
  )
  
  names(probes_set) <- probes_numbered[[probe_id_col]]
  
  Biostrings::writeXStringSet(
    probes_set,
    filepath = probes_fasta_path
  )
  
  # Optionally export reverse-complement target FASTA
  if (!is.null(targets_fasta_path)) {
    
    targets_set <- Biostrings::reverseComplement(
      probes_set
    )
    
    names(targets_set) <- probes_numbered[[probe_id_col]]
    
    Biostrings::writeXStringSet(
      targets_set,
      filepath = targets_fasta_path
    )
  }
  
  return(
    invisible(
      list(
        probes_numbered = probes_numbered,
        n_probes = nrow(probes_numbered),
        probes_fasta = probes_fasta_path,
        targets_fasta = targets_fasta_path,
        probe_id_col = probe_id_col
      )
    )
  )
}
