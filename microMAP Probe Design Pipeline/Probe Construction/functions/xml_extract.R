# Parse BLAST XML output and calculate sequence-matching statistics for each HSP.
# Match count, matched GC content, and maximum consecutive matches are calculated
# for downstream probe specificity filtering. Aligned sequences can optionally
# be retained in the output.

xml_extract <- function(
    blast_xml_file,
    keep_seqs = FALSE
) {
  
  # Check required package
  if (!requireNamespace("XML", quietly = TRUE)) {
    stop("xml_extract requires the 'XML' package.")
  }
  
  # Check input file
  if (!file.exists(blast_xml_file)) {
    stop(
      "BLAST XML file does not exist: ",
      blast_xml_file
    )
  }
  
  blast_xml <- XML::xmlParse(
    blast_xml_file
  )
  
  iter_nodes <- XML::getNodeSet(
    blast_xml,
    "//Iteration"
  )
  
  # Safely combine a list of data frames
  safe_rbind <- function(lst) {
    
    lst <- Filter(
      Negate(is.null),
      lst
    )
    
    if (length(lst) == 0) {
      return(NULL)
    }
    
    do.call(
      rbind,
      lst
    )
  }
  
  # Calculate the longest consecutive TRUE run
  longest_run <- function(x) {
    
    if (length(x) == 0 || !any(x)) {
      return(0)
    }
    
    r <- rle(x)
    
    max(
      r$lengths[r$values]
    )
  }
  
  iter_list <- lapply(
    iter_nodes,
    function(iter_node) {
      
      query_id <- XML::xmlValue(
        iter_node[["Iteration_query-ID"]]
      )
      
      query_def <- XML::xmlValue(
        iter_node[["Iteration_query-def"]]
      )
      
      hit_nodes <- XML::getNodeSet(
        iter_node,
        ".//Hit"
      )
      
      if (length(hit_nodes) == 0) {
        return(NULL)
      }
      
      hit_list <- lapply(
        hit_nodes,
        function(hit_node) {
          
          hit_id <- XML::xmlValue(
            hit_node[["Hit_id"]]
          )
          
          hit_title <- XML::xmlValue(
            hit_node[["Hit_def"]]
          )
          
          hsp_nodes <- XML::getNodeSet(
            hit_node,
            ".//Hsp"
          )
          
          if (length(hsp_nodes) == 0) {
            return(NULL)
          }
          
          hsp_list <- lapply(
            hsp_nodes,
            function(hsp_node) {
              
              hsp_gaps <- as.numeric(
                XML::xmlValue(
                  hsp_node[["Hsp_gaps"]]
                )
              )
              
              hsp_identity <- as.numeric(
                XML::xmlValue(
                  hsp_node[["Hsp_identity"]]
                )
              )
              
              # Extract aligned query and hit sequences
              hsp_query_seq <- toupper(
                XML::xmlValue(
                  hsp_node[["Hsp_qseq"]]
                )
              )
              
              hsp_hit_seq <- toupper(
                XML::xmlValue(
                  hsp_node[["Hsp_hseq"]]
                )
              )
              
              q <- strsplit(
                hsp_query_seq,
                "",
                fixed = TRUE
              )[[1]]
              
              h <- strsplit(
                hsp_hit_seq,
                "",
                fixed = TRUE
              )[[1]]
              
              # Identify exact nucleotide matches
              exact_match <- (
                q == h &
                  q != "-" &
                  h != "-"
              )
              
              # Count matched G/C bases
              matched_gc <- (
                exact_match &
                  q %in% c("G", "C")
              )
              
              matched_gc_count <- sum(
                matched_gc
              )
              
              # Maximum consecutive matched G/C bases
              consecutive_gc_count <- longest_run(
                matched_gc
              )
              
              # Maximum consecutive exact matches
              consecutive_match_count <- longest_run(
                exact_match
              )
              
              df <- data.frame(
                query_id = query_id,
                query_def = query_def,
                hit_id = hit_id,
                hit_title = hit_title,
                gap_count = hsp_gaps,
                match_count = hsp_identity,
                matched_gc_count = matched_gc_count,
                consecutive_gc_count = consecutive_gc_count,
                consecutive_match_count = consecutive_match_count,
                stringsAsFactors = FALSE
              )
              
              # Optionally retain aligned sequences
              if (isTRUE(keep_seqs)) {
                df$hsp_query_seq <- hsp_query_seq
                df$hsp_hit_seq <- hsp_hit_seq
              }
              
              df
            }
          )
          
          safe_rbind(
            hsp_list
          )
        }
      )
      
      safe_rbind(
        hit_list
      )
    }
  )
  
  out <- safe_rbind(
    iter_list
  )
  
  # Return an empty data frame with the expected structure
  # when no BLAST hits are found.
  if (is.null(out)) {
    
    out <- data.frame(
      query_id = character(),
      query_def = character(),
      hit_id = character(),
      hit_title = character(),
      gap_count = numeric(),
      match_count = numeric(),
      matched_gc_count = numeric(),
      consecutive_gc_count = numeric(),
      consecutive_match_count = numeric(),
      stringsAsFactors = FALSE
    )
    
    if (isTRUE(keep_seqs)) {
      out$hsp_query_seq <- character()
      out$hsp_hit_seq <- character()
    }
  }
  
  return(out)
}
