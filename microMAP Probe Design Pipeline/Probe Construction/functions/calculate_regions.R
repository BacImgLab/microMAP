# Extract short flanking regions adjacent to each primary probe.
# These regions are used as sequence-derived spacers between the primary
# probe region and the fluorescent/readout probe region.

calculate_regions <- function(
    pri_probe_info,
    seq_df,
    id_col = "identifier",
    flank_length = 2,
    interval = 0,
    reverse_region = TRUE
) {
  
  # Check required package
  if (
    isTRUE(reverse_region) &&
    !requireNamespace("stringi", quietly = TRUE)
  ) {
    stop(
      "calculate_regions requires the 'stringi' package when ",
      "'reverse_region = TRUE'."
    )
  }
  
  # Check primary probe input
  if (!is.data.frame(pri_probe_info)) {
    stop(
      "'pri_probe_info' must be a data.frame or tibble."
    )
  }
  
  required_cols <- c(
    id_col,
    "start",
    "end"
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(pri_probe_info)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'pri_probe_info': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Check region parameters
  if (
    length(flank_length) != 1 ||
    !is.numeric(flank_length) ||
    is.na(flank_length) ||
    flank_length < 0
  ) {
    stop(
      "'flank_length' must be a single numeric value >= 0."
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
  
  flank_length <- as.integer(
    flank_length
  )
  
  interval <- as.integer(
    interval
  )
  
  # Standardize reference sequence input
  if (is.data.frame(seq_df)) {
    
    required_seq_cols <- c(
      id_col,
      "sequence"
    )
    
    missing_seq_cols <- setdiff(
      required_seq_cols,
      colnames(seq_df)
    )
    
    if (length(missing_seq_cols) > 0) {
      stop(
        "Missing required column(s) in 'seq_df': ",
        paste(missing_seq_cols, collapse = ", ")
      )
    }
    
    seq_values <- as.character(
      seq_df[["sequence"]]
    )
    
    names(seq_values) <- as.character(
      seq_df[[id_col]]
    )
    
  } else {
    
    if (is.null(names(seq_df))) {
      stop(
        "When 'seq_df' is not a data.frame, sequence names must be provided."
      )
    }
    
    seq_values <- as.character(
      seq_df
    )
    
    names(seq_values) <- names(
      seq_df
    )
  }
  
  # Keep the first sequence when duplicated identifiers are present
  if (anyDuplicated(names(seq_values)) > 0) {
    seq_values <- seq_values[
      !duplicated(names(seq_values))
    ]
  }
  
  # Match each probe to its reference sequence
  ids <- as.character(
    pri_probe_info[[id_col]]
  )
  
  seq_idx <- match(
    ids,
    names(seq_values)
  )
  
  if (any(is.na(seq_idx))) {
    
    missing_ids <- unique(
      ids[is.na(seq_idx)]
    )
    
    stop(
      "Some identifiers were not found in 'seq_df': ",
      paste(
        head(missing_ids, 20),
        collapse = ", "
      )
    )
  }
  
  # Initialize flanking sequence columns
  pri_probe_info$start_region <- NA_character_
  pri_probe_info$end_region <- NA_character_
  
  # Extract flanking regions for each primary probe
  for (i in seq_len(nrow(pri_probe_info))) {
    
    current_id <- ids[i]
    
    current_start <- as.integer(
      pri_probe_info$start[i]
    )
    
    current_end <- as.integer(
      pri_probe_info$end[i]
    )
    
    current_seq <- seq_values[
      seq_idx[i]
    ]
    
    current_seq_len <- nchar(
      current_seq
    )
    
    # Check probe coordinates
    if (is.na(current_start) || is.na(current_end)) {
      stop(
        "Probe start or end position is NA at row ",
        i
      )
    }
    
    if (current_start < 1) {
      stop(
        "Probe start position is < 1 at row ",
        i,
        ": ",
        current_id
      )
    }
    
    if (current_start > current_end) {
      stop(
        "Probe start position exceeds end position at row ",
        i,
        ": ",
        current_id
      )
    }
    
    if (current_end > current_seq_len) {
      stop(
        "Probe end position exceeds sequence length at row ",
        i,
        ": ",
        current_id,
        ", end = ",
        current_end,
        ", sequence length = ",
        current_seq_len
      )
    }
    
    # Define upstream flanking region
    start_region_start <- max(
      1,
      current_start - flank_length - interval
    )
    
    start_region_end <- current_start - interval - 1
    
    # Define downstream flanking region
    end_region_start <- current_end + interval + 1
    
    end_region_end <- min(
      current_seq_len,
      current_end + interval + flank_length
    )
    
    # Extract upstream spacer sequence
    if (start_region_end >= start_region_start) {
      
      start_region <- substr(
        current_seq,
        start_region_start,
        start_region_end
      )
      
    } else {
      
      start_region <- ""
    }
    
    # Extract downstream spacer sequence
    if (end_region_end >= end_region_start) {
      
      end_region <- substr(
        current_seq,
        end_region_start,
        end_region_end
      )
      
    } else {
      
      end_region <- ""
    }
    
    # Reverse sequence order when required.
    # No complement or reverse-complement operation is performed.
    if (isTRUE(reverse_region)) {
      
      start_region <- stringi::stri_reverse(
        start_region
      )
      
      end_region <- stringi::stri_reverse(
        end_region
      )
    }
    
    pri_probe_info$start_region[i] <- start_region
    pri_probe_info$end_region[i] <- end_region
  }
  
  return(pri_probe_info)
}