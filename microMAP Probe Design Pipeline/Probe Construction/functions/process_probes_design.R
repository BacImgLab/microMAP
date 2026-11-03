# Design candidate primary probes from input 16S rRNA sequences.
# Candidate regions are generated with DECIPHER::TileSeqs(), followed by
# probe design using DECIPHER::DesignProbes().

process_probes_design <- function(
    input_fasta,
    db_file,
    minLength_tiles = 28,
    maxLength_tiles = 34,
    maxPermutations_tiles = 1000,
    minCover_tiles = 1,
    minLength_dp = 28,
    maxLength_dp = 34,
    maxPermutations = 1,
    hybTemp = 37,
    P = 100e-9,
    Na = 0.3,
    FA = 50,
    processors = 8
) {
  
  # Check required packages
  required_packages <- c(
    "Biostrings",
    "DBI",
    "RSQLite",
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
  
  # Check input files and parameters
  if (!file.exists(input_fasta)) {
    stop(
      "Input FASTA file does not exist: ",
      input_fasta
    )
  }
  
  if (
    length(processors) != 1 ||
    !is.numeric(processors) ||
    is.na(processors) ||
    processors < 1
  ) {
    stop(
      "'processors' must be a single numeric value >= 1."
    )
  }
  
  processors <- as.integer(processors)
  
  # Read input sequences
  rRNA_seqs <- Biostrings::readDNAStringSet(
    filepath = input_fasta
  )
  
  if (length(rRNA_seqs) == 0) {
    stop(
      "No sequences were found in the input FASTA file."
    )
  }
  
  # Build DECIPHER sequence database
  con <- DBI::dbConnect(
    RSQLite::SQLite(),
    dbname = db_file
  )
  
  on.exit(
    {
      if (DBI::dbIsValid(con)) {
        DBI::dbDisconnect(con)
      }
    },
    add = TRUE
  )
  
  DECIPHER::Seqs2DB(
    seqs = rRNA_seqs,
    type = "XStringSet",
    dbFile = con,
    identifier = names(rRNA_seqs),
    tblName = "Seqs",
    verbose = TRUE,
    replaceTbl = TRUE,
    processors = processors
  )
  
  # Generate candidate sequence tiles
  tiles <- DECIPHER::TileSeqs(
    con,
    add2tbl = "Tiles",
    identifier = "",
    minLength = minLength_tiles,
    maxLength = maxLength_tiles,
    maxTilePermutations = maxPermutations_tiles,
    minCoverage = minCover_tiles,
    processors = processors
  )
  
  DBI::dbDisconnect(con)
  
  # Design candidate probes
  probes <- DECIPHER::DesignProbes(
    tiles,
    identifier = "",
    minLength = minLength_dp,
    maxLength = maxLength_dp,
    maxPermutations = maxPermutations,
    hybTemp = hybTemp,
    P = P,
    Na = Na,
    FA = FA,
    verbose = TRUE
  )
  
  return(probes)
}