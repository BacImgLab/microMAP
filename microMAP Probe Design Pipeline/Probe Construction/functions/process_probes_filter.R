# Filter candidate primary probes by probe length, GC content, and binding free energy.
# Probe binding efficiency is calculated with DECIPHER::CalculateEfficiencyFISH()
# under defined hybridization conditions.

process_probes_filter <- function(
    probes,
    minLength_dp = 28,
    maxLength_dp = 34,
    min_GC = 0.45,
    max_GC = 0.65,
    temp = 37,
    P = 100e-9,
    ions = 0.3,
    FA = 50,
    batchSize = 1000,
    max_dG = -13.0
) {
  
  # Check required packages
  required_packages <- c(
    "Biostrings",
    "DECIPHER"
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
  
  # Check input
  if (!is.data.frame(probes)) {
    stop("'probes' must be a data.frame or tibble.")
  }
  
  if (!"probe" %in% colnames(probes)) {
    stop("'probes' must contain a 'probe' column.")
  }
  
  # Extract probe sequences
  probe_seq <- probes$probe
  
  if (is.matrix(probe_seq) || is.data.frame(probe_seq)) {
    probe_seq <- probe_seq[, 1]
  }
  
  probe_seq <- as.character(probe_seq)
  
  # Filter by probe length
  probes$len <- nchar(probe_seq)
  
  probes <- probes[
    probes$len >= minLength_dp &
      probes$len <= maxLength_dp,
    ,
    drop = FALSE
  ]
  
  if (nrow(probes) == 0) {
    warning("No probes remained after probe-length filtering.")
    return(probes)
  }
  
  # Re-extract sequences after length filtering
  probe_seq <- probes$probe
  
  if (is.matrix(probe_seq) || is.data.frame(probe_seq)) {
    probe_seq <- probe_seq[, 1]
  }
  
  probe_seq <- as.character(probe_seq)
  
  # Calculate GC content
  probe_set <- Biostrings::DNAStringSet(probe_seq)
  
  probes$GC_content <- (
    Biostrings::letterFrequency(
      probe_set,
      letters = c("G", "C")
    ) |>
      rowSums()
  ) / Biostrings::width(probe_set)
  
  # Filter by GC content
  filtered_probes <- probes[
    probes$GC_content >= min_GC &
      probes$GC_content <= max_GC,
    ,
    drop = FALSE
  ]
  
  if (nrow(filtered_probes) == 0) {
    warning("No probes remained after GC-content filtering.")
    return(filtered_probes)
  }
  
  # Calculate probe binding free energy
  probe <- filtered_probes$probe
  
  if (is.matrix(probe) || is.data.frame(probe)) {
    probe <- probe[, 1]
  }
  
  probe <- as.character(probe)
  
  target <- Biostrings::reverseComplement(
    Biostrings::DNAStringSet(probe)
  )
  
  probe_G <- DECIPHER::CalculateEfficiencyFISH(
    probe,
    target,
    temp,
    P,
    ions,
    FA,
    batchSize
  )
  
  filtered_probes$dG1 <- probe_G[, 4]
  
  # Retain probes with sufficiently favorable binding free energy
  filtered_probes <- filtered_probes[
    filtered_probes$dG1 < max_dG,
    ,
    drop = FALSE
  ]
  
  return(filtered_probes)
}