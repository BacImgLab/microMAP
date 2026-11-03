# Run nucleotide BLAST for probe specificity analysis.
# BLAST parameters, strand, and output format can be specified explicitly.

run_blastn <- function(
    blastn,
    query_blast,
    db_out,
    out_dir,
    blast_tag = NULL,
    word_size = 5,
    evalue = "1e-2",
    outfmt = 5,
    max_hsps = 1,
    max_target_seqs = 1000,
    strand,
    overwrite = TRUE
) {
  
  # Check input files
  if (!file.exists(query_blast)) {
    stop(
      "Query FASTA file does not exist: ",
      query_blast
    )
  }
  
  if (!file.exists(blastn)) {
    stop(
      "blastn executable does not exist: ",
      blastn
    )
  }
  
  # Check BLAST database
  db_files <- paste0(
    db_out,
    c(".nhr", ".nin", ".nsq")
  )
  
  if (!all(file.exists(db_files))) {
    stop(
      "BLAST database files were not found for: ",
      db_out
    )
  }
  
  # Check strand
  strand <- tolower(strand)
  
  if (!strand %in% c("plus", "minus", "both")) {
    stop(
      "'strand' must be 'plus', 'minus', or 'both'."
    )
  }
  
  # Create output directory
  dir.create(
    out_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  # Generate output tag from query filename when not provided
  if (is.null(blast_tag) || !nzchar(blast_tag)) {
    
    blast_tag <- tools::file_path_sans_ext(
      basename(query_blast)
    )
  }
  
  blast_tag <- gsub(
    "[^A-Za-z0-9_\\-]+",
    "_",
    blast_tag
  )
  
  # Use XML extension for BLAST XML output
  file_ext <- if (as.integer(outfmt) == 5) {
    "xml"
  } else {
    "txt"
  }
  
  blast_out <- file.path(
    out_dir,
    paste0(
      blast_tag,
      "_blast.",
      file_ext
    )
  )
  
  # Reuse existing output if requested
  if (file.exists(blast_out) && !overwrite) {
    
    message(
      "[run_blastn] BLAST output exists, skip: ",
      blast_out
    )
    
    return(
      invisible(
        list(
          blast_out = blast_out,
          skipped = TRUE
        )
      )
    )
  }
  
  # Build BLASTN arguments
  args <- c(
    "-query", shQuote(query_blast),
    "-db", shQuote(db_out),
    "-word_size", as.integer(word_size),
    "-evalue", as.character(evalue),
    "-strand", strand,
    "-outfmt", as.integer(outfmt),
    "-out", shQuote(blast_out),
    "-max_hsps", as.integer(max_hsps),
    "-max_target_seqs", as.integer(max_target_seqs)
  )
  
  blast_cmd <- paste(
    shQuote(blastn),
    paste(args, collapse = " ")
  )
  
  message(
    "[run_blastn] query: ",
    query_blast
  )
  
  message(
    "[run_blastn] db: ",
    db_out
  )
  
  message(
    "[run_blastn] tag: ",
    blast_tag
  )
  
  message(
    "[run_blastn] strand: ",
    strand
  )
  
  message(
    "[run_blastn] CMD:\n",
    blast_cmd
  )
  
  # Run BLASTN
  status <- system2(
    command = blastn,
    args = args
  )
  
  if (status != 0) {
    stop(
      "BLASTN failed with exit status: ",
      status,
      "\nCMD: ",
      blast_cmd
    )
  }
  
  return(
    invisible(
      list(
        blast_out = blast_out,
        skipped = FALSE,
        strand = strand,
        blast_tag = blast_tag,
        cmd = blast_cmd
      )
    )
  )
}
