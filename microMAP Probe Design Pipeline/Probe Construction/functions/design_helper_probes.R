# Design left and right helper probes adjacent to each final primary probe.
# Helper probes are generated from flanking 16S rRNA regions with a defined
# interval and converted to reverse-complement probe sequences.

design_helper_probes <- function(
    fin_primary_probes,
    seq_16S,
    helper_length = 30,
    interval = 3
) {
  
  # Check required packages
  required_packages <- c(
    "Biostrings",
    "dplyr"
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
  
  # Check primary probe input
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
  
  # Check helper probe parameters
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
  
  # Standardize 16S reference sequences
  if (is.null(names(seq_16S))) {
    stop(
      "'seq_16S' must contain sequence names."
    )
  }
  
  seq_values <- as.character(
    seq_16S
  )
  
  names(seq_values) <- names(
    seq_16S
  )
  
  if (anyDuplicated(names(seq_values)) > 0) {
    message(
      "[design_helper_probes] duplicated sequence names found; ",
      "keeping the first occurrence."
    )
    
    seq_values <- seq_values[
      !duplicated(names(seq_values))
    ]
  }
  
  # Design one helper probe on the specified side
  make_one_helper <- function(
    row_id,
    side
  ) {
    
    parent_probe_id <- as.character(
      fin_primary_probes$probe_id[row_id]
    )
    
    identifier <- as.character(
      fin_primary_probes$identifier[row_id]
    )
    
    primary_start <- as.integer(
      fin_primary_probes$start[row_id]
    )
    
    primary_end <- as.integer(
      fin_primary_probes$.end[row_id]
    )
    
    if (!(identifier %in% names(seq_values))) {
      stop(
        "Identifier not found in 'seq_16S': ",
        identifier
      )
    }
    
    current_seq <- seq_values[[identifier]]
    
    seq_len <- nchar(
      current_seq
    )
    
    # Check primary probe coordinates
    if (is.na(primary_start) || is.na(primary_end)) {
      stop(
        "Primary probe start or end position is NA at row ",
        row_id
      )
    }
    
    if (primary_start < 1) {
      stop(
        "Primary probe start position is < 1 at row ",
        row_id,
        ": ",
        identifier
      )
    }
    
    if (primary_start > primary_end) {
      stop(
        "Primary probe start position exceeds end position at row ",
        row_id,
        ": ",
        identifier
      )
    }
    
    if (primary_end > seq_len) {
      stop(
        "Primary probe end position exceeds sequence length at row ",
        row_id,
        ": ",
        identifier,
        ", end = ",
        primary_end,
        ", sequence length = ",
        seq_len
      )
    }
    
    # Define helper target region
    if (side == "left") {
      
      target_end <- primary_start - interval - 1
      target_start <- target_end - helper_length + 1
      
    } else if (side == "right") {
      
      target_start <- primary_end + interval + 1
      target_end <- target_start + helper_length - 1
      
    } else {
      
      stop(
        "'side' must be 'left' or 'right'."
      )
    }
    
    # Skip helper probes that would extend beyond the reference sequence
    if (
      target_start < 1 ||
      target_end > seq_len ||
      target_start > target_end
    ) {
      return(NULL)
    }
    
    # Extract helper target sequence
    target_sequence <- substr(
      current_seq,
      target_start,
      target_end
    )
    
    # Convert target sequence to reverse-complement helper probe
    helper_probe <- as.character(
      Biostrings::reverseComplement(
        Biostrings::DNAString(
          target_sequence
        )
      )
    )
    
    data.frame(
      parent_probe_id = parent_probe_id,
      helper_id = paste0(
        parent_probe_id,
        "_helper_",
        side
      ),
      identifier = identifier,
      helper_side = side,
      primary_start = primary_start,
      primary_end = primary_end,
      target_start = target_start,
      target_end = target_end,
      target_sequence = target_sequence,
      helper_probe = helper_probe,
      helper_length = nchar(
        helper_probe
      ),
      interval = interval,
      stringsAsFactors = FALSE
    )
  }
  
  # Generate left and right helper probes for each primary probe
  helper_list <- lapply(
    seq_len(nrow(fin_primary_probes)),
    function(i) {
      
      dplyr::bind_rows(
        make_one_helper(
          i,
          "left"
        ),
        make_one_helper(
          i,
          "right"
        )
      )
    }
  )
  
  helper_probes <- dplyr::bind_rows(
    helper_list
  )
  
  if (nrow(helper_probes) == 0) {
    warning(
      "No helper probes were generated."
    )
    
    return(
      data.frame()
    )
  }
  
  return(helper_probes)
}
