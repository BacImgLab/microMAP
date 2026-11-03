# Filter candidate probes based on BLAST off-target matching statistics.
# Hits from the same genus are treated as expected target hits, while
# different-genus hits are evaluated using the specified matching thresholds.

blast_match_filter <- function(
    result_df,
    max_match = NULL,
    max_gc = NULL,
    max_con_match = NULL,
    max_con_gc = NULL,
    enforce_strict_self_match = FALSE,
    strict_self_con_match = 28,
    require_self_hit = TRUE
) {
  
  # Check required packages
  required_packages <- c(
    "dplyr",
    "tibble"
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
  if (!is.data.frame(result_df)) {
    stop("'result_df' must be a data.frame or tibble.")
  }
  
  required_cols <- c(
    "query_def",
    "hit_title"
  )
  
  if (!is.null(max_match)) {
    required_cols <- c(
      required_cols,
      "match_count"
    )
  }
  
  if (!is.null(max_gc)) {
    required_cols <- c(
      required_cols,
      "matched_gc_count"
    )
  }
  
  if (
    !is.null(max_con_match) ||
    isTRUE(enforce_strict_self_match)
  ) {
    required_cols <- c(
      required_cols,
      "consecutive_match_count"
    )
  }
  
  if (!is.null(max_con_gc)) {
    required_cols <- c(
      required_cols,
      "consecutive_gc_count"
    )
  }
  
  missing_cols <- setdiff(
    required_cols,
    colnames(result_df)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'result_df': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # Extract genus name as the first word of each sequence identifier
  parse_genus <- function(x) {
    
    x <- trimws(
      as.character(x)
    )
    
    parts <- strsplit(
      x,
      " ",
      fixed = TRUE
    )
    
    vapply(
      parts,
      function(p) {
        
        if (
          length(p) >= 1 &&
          nzchar(p[[1]])
        ) {
          p[[1]]
        } else {
          NA_character_
        }
      },
      character(1)
    )
  }
  
  df <- result_df
  
  df$target_genus <- parse_genus(
    df$query_def
  )
  
  df$hit_genus <- parse_genus(
    df$hit_title
  )
  
  df$is_same_genus <- (
    !is.na(df$target_genus) &
      !is.na(df$hit_genus) &
      df$target_genus == df$hit_genus
  )
  
  # Filter probes according to all different-genus BLAST hits
  df_filtered <- tibble::as_tibble(df) %>%
    dplyr::group_by(
      query_def
    ) %>%
    dplyr::filter({
      
      g <- dplyr::pick(
        dplyr::everything()
      )
      
      off <- g[
        !g$is_same_genus,
        ,
        drop = FALSE
      ]
      
      if (nrow(off) == 0) {
        
        TRUE
        
      } else {
        
        ok <- rep(
          TRUE,
          nrow(off)
        )
        
        if (!is.null(max_match)) {
          ok <- ok &
            off$match_count <= max_match
        }
        
        if (!is.null(max_gc)) {
          ok <- ok &
            off$matched_gc_count <= max_gc
        }
        
        if (!is.null(max_con_match)) {
          ok <- ok &
            off$consecutive_match_count < max_con_match
        }
        
        if (!is.null(max_con_gc)) {
          ok <- ok &
            off$consecutive_gc_count < max_con_gc
        }
        
        all(ok)
      }
    }) %>%
    dplyr::ungroup()
  
  # Require at least one expected same-genus hit
  if (isTRUE(require_self_hit)) {
    
    df_filtered <- df_filtered %>%
      dplyr::group_by(
        query_def
      ) %>%
      dplyr::filter(
        any(
          is_same_genus,
          na.rm = TRUE
        )
      ) %>%
      dplyr::ungroup()
  }
  
  # Optionally require a strict expected target match
  if (isTRUE(enforce_strict_self_match)) {
    
    df_filtered <- df_filtered %>%
      dplyr::group_by(
        query_def
      ) %>%
      dplyr::filter({
        
        g <- dplyr::pick(
          dplyr::everything()
        )
        
        self_hits <- g[
          g$is_same_genus,
          ,
          drop = FALSE
        ]
        
        any(
          self_hits$consecutive_match_count >=
            strict_self_con_match &
            self_hits$consecutive_match_count ==
            self_hits$match_count
        )
      }) %>%
      dplyr::ungroup()
  }
  
  return(df_filtered)
}