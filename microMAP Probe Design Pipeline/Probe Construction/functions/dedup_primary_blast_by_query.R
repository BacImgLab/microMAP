# Summarize filtered BLAST results for each primary probe.
# Same-genus BLAST hits are collapsed at the probe level, and their
# corresponding abundance values can optionally be summed by taxon name.

dedup_primary_blast_by_query <- function(
    blast_filtered_df,
    abundance_table = NULL,
    abundance_key_col = "taxon_name",
    abundance_value_col = "Abundance",
    same_genus_col = "is_same_genus",
    hit_title_col = "hit_title"
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
  
  # Check BLAST input
  required_cols <- c(
    "query_def",
    "target_genus",
    hit_title_col,
    same_genus_col
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(blast_filtered_df)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'blast_filtered_df': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  df <- tibble::as_tibble(
    blast_filtered_df
  )
  
  # Prepare abundance information
  if (!is.null(abundance_table)) {
    
    if (!is.data.frame(abundance_table)) {
      stop(
        "'abundance_table' must be a data.frame or tibble."
      )
    }
    
    required_abundance_cols <- c(
      abundance_key_col,
      abundance_value_col
    )
    
    missing_abundance_cols <- setdiff(
      required_abundance_cols,
      colnames(abundance_table)
    )
    
    if (length(missing_abundance_cols) > 0) {
      stop(
        "Missing required column(s) in 'abundance_table': ",
        paste(missing_abundance_cols, collapse = ", ")
      )
    }
    
    abundance_info <- abundance_table %>%
      dplyr::transmute(
        taxon_name = as.character(
          .data[[abundance_key_col]]
        ),
        Abundance = as.numeric(
          .data[[abundance_value_col]]
        )
      ) %>%
      dplyr::distinct(
        taxon_name,
        .keep_all = TRUE
      )
    
  } else {
    
    abundance_info <- NULL
  }
  
  # Calculate abundance represented by same-genus BLAST hits
  if (!is.null(abundance_info)) {
    
    per_hit <- df %>%
      dplyr::filter(
        .data[[same_genus_col]] %in% TRUE
      ) %>%
      dplyr::select(
        query_def,
        hit_title = dplyr::all_of(
          hit_title_col
        )
      ) %>%
      dplyr::distinct(
        query_def,
        hit_title
      ) %>%
      dplyr::left_join(
        abundance_info,
        by = c(
          "hit_title" = "taxon_name"
        )
      )
    
    abundance_sum <- per_hit %>%
      dplyr::group_by(
        query_def
      ) %>%
      dplyr::summarise(
        abundance_sum = sum(
          Abundance,
          na.rm = TRUE
        ),
        n_same_genus_hit_titles = dplyr::n(),
        n_abundance_matched = sum(
          !is.na(Abundance)
        ),
        .groups = "drop"
      )
    
  } else {
    
    abundance_sum <- df %>%
      dplyr::distinct(
        query_def
      ) %>%
      dplyr::transmute(
        query_def,
        abundance_sum = NA_real_,
        n_same_genus_hit_titles = NA_integer_,
        n_abundance_matched = NA_integer_
      )
  }
  
  # Summarize BLAST results at the probe level
  probe_level <- df %>%
    dplyr::group_by(
      query_def
    ) %>%
    dplyr::summarise(
      
      probe_name = dplyr::first(
        query_def
      ),
      
      target_genus = dplyr::first(
        target_genus
      ),
      
      # Summary of all BLAST hits
      n_hits_total = dplyr::n(),
      
      n_hit_titles_total = dplyr::n_distinct(
        .data[[hit_title_col]]
      ),
      
      hit_titles_total = paste(
        unique(
          .data[[hit_title_col]]
        ),
        collapse = ";"
      ),
      
      # Summary of same-genus BLAST hits
      n_same_genus_hits = sum(
        .data[[same_genus_col]] %in% TRUE
      ),
      
      n_same_genus_hit_titles =
        dplyr::n_distinct(
          .data[[hit_title_col]][
            .data[[same_genus_col]] %in% TRUE
          ]
        ),
      
      same_genus_hit_titles = paste(
        unique(
          .data[[hit_title_col]][
            .data[[same_genus_col]] %in% TRUE
          ]
        ),
        collapse = ";"
      ),
      
      .groups = "drop"
    ) %>%
    dplyr::left_join(
      abundance_sum,
      by = "query_def"
    ) %>%
    dplyr::mutate(
      n_same_genus_hit_titles =
        dplyr::coalesce(
          n_same_genus_hit_titles.y,
          n_same_genus_hit_titles.x
        ),
      n_same_genus_hit_titles.x = NULL,
      n_same_genus_hit_titles.y = NULL
    )
  
  return(probe_level)
}