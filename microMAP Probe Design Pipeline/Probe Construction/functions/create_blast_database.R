# Build a local nucleotide BLAST database from input FASTA sequences.
# Reference sequences can optionally be filtered by abundance rank or
# minimum abundance before database construction.

create_blast_database <- function(
    makeblastdb,
    query_db,
    abundance_table = NULL,
    rank_threshold = NULL,
    min_abundance = NULL,
    taxon_col = "taxon_name",
    db_out = NULL,
    db_dir = NULL,
    db_name = NULL
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
  
  # Check input files
  if (!file.exists(query_db)) {
    stop(
      "Input FASTA file does not exist: ",
      query_db
    )
  }
  
  if (!file.exists(makeblastdb)) {
    stop(
      "makeblastdb executable does not exist: ",
      makeblastdb
    )
  }
  
  # Determine BLAST database output prefix
  if (is.null(db_out)) {
    
    if (
      is.null(db_dir) ||
      is.null(db_name) ||
      !nzchar(db_dir) ||
      !nzchar(db_name)
    ) {
      stop(
        "Provide either 'db_out' or both 'db_dir' and 'db_name'."
      )
    }
    
    db_out <- file.path(
      db_dir,
      db_name
    )
  }
  
  # Create output directory
  dir.create(
    dirname(db_out),
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  # Read reference sequences
  sequences <- Biostrings::readDNAStringSet(
    query_db
  )
  
  if (length(sequences) == 0) {
    stop(
      "No sequences were found in the input FASTA file."
    )
  }
  
  seq_headers <- names(sequences)
  
  keep_headers <- seq_headers
  filter_note <- "no filter (use all sequences in query_db)"
  
  # Optionally filter reference sequences by abundance
  if (!is.null(rank_threshold) || !is.null(min_abundance)) {
    
    if (is.null(abundance_table)) {
      stop(
        "'abundance_table' is required when abundance filtering is used."
      )
    }
    
    if (!is.data.frame(abundance_table)) {
      stop(
        "'abundance_table' must be a data.frame or tibble."
      )
    }
    
    if (!(taxon_col %in% colnames(abundance_table))) {
      stop(
        "Missing required column in 'abundance_table': ",
        taxon_col
      )
    }
    
    df <- abundance_table
    
    if (!is.null(rank_threshold)) {
      
      if (!"rank" %in% colnames(df)) {
        stop(
          "Missing required column in 'abundance_table': rank"
        )
      }
      
      df <- df %>%
        dplyr::filter(
          rank <= rank_threshold
        )
    }
    
    if (!is.null(min_abundance)) {
      
      if (!"Abundance" %in% colnames(df)) {
        stop(
          "Missing required column in 'abundance_table': Abundance"
        )
      }
      
      df <- df %>%
        dplyr::filter(
          Abundance >= min_abundance
        )
    }
    
    keep_headers <- intersect(
      unique(df[[taxon_col]]),
      seq_headers
    )
    
    if (length(keep_headers) == 0) {
      stop(
        "No reference sequences remained after abundance filtering. ",
        "Check that '", taxon_col,
        "' matches the FASTA sequence identifiers."
      )
    }
    
    filter_note <- paste0(
      "filtered by ",
      ifelse(
        is.null(rank_threshold),
        "",
        paste0("rank<=", rank_threshold)
      ),
      ifelse(
        !is.null(rank_threshold) &&
          !is.null(min_abundance),
        " and ",
        ""
      ),
      ifelse(
        is.null(min_abundance),
        "",
        paste0("Abundance>=", min_abundance)
      )
    )
  }
  
  filtered_sequences <- sequences[
    seq_headers %in% keep_headers
  ]
  
  # Check whether the BLAST database already exists
  db_exists <- all(
    file.exists(
      paste0(
        db_out,
        c(".nhr", ".nin", ".nsq")
      )
    )
  )
  
  message(
    "[blastdb] policy: ",
    filter_note
  )
  
  message(
    "[blastdb] seqs used: ",
    length(filtered_sequences),
    " / ",
    length(sequences)
  )
  
  message(
    "[blastdb] db_out: ",
    db_out
  )
  
  if (db_exists) {
    message(
      "[blastdb] exists, skip makeblastdb"
    )
    
    return(db_out)
  }
  
  # Write filtered reference sequences
  fasta_out <- paste0(
    db_out,
    ".fasta"
  )
  
  Biostrings::writeXStringSet(
    filtered_sequences,
    filepath = fasta_out
  )
  
  # Build nucleotide BLAST database
  args <- c(
    "-in", fasta_out,
    "-dbtype", "nucl",
    "-out", db_out
  )
  
  message(
    "[makeblastdb] ",
    shQuote(makeblastdb),
    " ",
    paste(
      shQuote(args),
      collapse = " "
    )
  )
  
  status <- system2(
    command = makeblastdb,
    args = args
  )
  
  if (status != 0) {
    stop(
      "makeblastdb failed with exit status: ",
      status
    )
  }
  
  return(db_out)
}
