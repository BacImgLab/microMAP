# Select non-overlapping primary probes within each target.
# Probes are prioritized by abundance and probe efficiency using either
# lexicographic ranking or a weighted score, while enforcing a minimum gap.

select_nonoverlap_probes_by_priority <- function(
    probes_df,
    id_col = "identifier",
    start_col = "start",
    len_col = "len",
    eff_col = "efficiency",
    abundance_col = "abundance_sum",
    probe_id_col = "probe_id",
    min_gap = 3,
    max_n_per_id = NULL,
    method = c("lexicographic", "weighted"),
    abundance_weight = 1,
    efficiency_weight = 1
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "select_nonoverlap_probes_by_priority requires the 'dplyr' package."
    )
  }
  
  method <- match.arg(method)
  
  # Check input data
  if (!is.data.frame(probes_df)) {
    stop("'probes_df' must be a data.frame or tibble.")
  }
  
  required_cols <- c(
    id_col,
    start_col,
    len_col,
    eff_col,
    abundance_col,
    probe_id_col
  )
  
  missing_cols <- setdiff(
    required_cols,
    colnames(probes_df)
  )
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required column(s) in 'probes_df': ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  if (
    length(min_gap) != 1 ||
    !is.numeric(min_gap) ||
    is.na(min_gap) ||
    min_gap < 0
  ) {
    stop("'min_gap' must be a single numeric value >= 0.")
  }
  
  min_gap <- as.integer(min_gap)
  
  if (!is.null(max_n_per_id)) {
    
    if (
      length(max_n_per_id) != 1 ||
      !is.numeric(max_n_per_id) ||
      is.na(max_n_per_id) ||
      max_n_per_id < 1
    ) {
      stop(
        "'max_n_per_id' must be NULL or a single numeric value >= 1."
      )
    }
    
    max_n_per_id <- as.integer(max_n_per_id)
  }
  
  # Convert matrix-like columns to atomic vectors when needed
  get_atomic_col <- function(x) {
    
    if (is.matrix(x) || is.data.frame(x)) {
      x <- x[, 1]
    }
    
    return(x)
  }
  
  # Prepare internal columns for probe selection
  df <- probes_df %>%
    dplyr::mutate(
      .identifier = as.character(
        .data[[id_col]]
      ),
      .probe_id = as.character(
        .data[[probe_id_col]]
      ),
      .start = as.integer(
        get_atomic_col(.data[[start_col]])
      ),
      .len = as.integer(
        get_atomic_col(.data[[len_col]])
      ),
      .eff = as.numeric(
        get_atomic_col(.data[[eff_col]])
      ),
      .abundance = as.numeric(
        get_atomic_col(.data[[abundance_col]])
      ),
      .end = .start + .len - 1L
    )
  
  if (any(is.na(df$.start))) {
    stop("NA values were found in probe start positions.")
  }
  
  if (any(is.na(df$.len))) {
    stop("NA values were found in probe lengths.")
  }
  
  if (any(is.na(df$.eff))) {
    stop("NA values were found in probe efficiency.")
  }
  
  if (any(is.na(df$.abundance))) {
    stop("NA values were found in abundance values.")
  }
  
  # Calculate weighted priority scores when requested
  if (method == "weighted") {
    
    range01 <- function(x) {
      
      r <- range(
        x,
        na.rm = TRUE
      )
      
      if (isTRUE(all.equal(r[1], r[2]))) {
        return(
          rep(1, length(x))
        )
      }
      
      return(
        (x - r[1]) /
          (r[2] - r[1])
      )
    }
    
    df <- df %>%
      dplyr::group_by(
        .identifier
      ) %>%
      dplyr::mutate(
        .abundance_scaled = range01(
          .abundance
        ),
        .eff_scaled = range01(
          .eff
        ),
        .score =
          abundance_weight * .abundance_scaled +
          efficiency_weight * .eff_scaled
      ) %>%
      dplyr::ungroup()
  }
  
  # Check whether a candidate probe overlaps or is too close
  # to any previously selected probe.
  is_compatible <- function(
    candidate_start,
    candidate_end,
    selected_df,
    min_gap
  ) {
    
    if (nrow(selected_df) == 0) {
      return(TRUE)
    }
    
    conflict <- (
      candidate_start <=
        selected_df$.end + min_gap
    ) &
      (
        candidate_end >=
          selected_df$.start - min_gap
      )
    
    return(
      !any(conflict)
    )
  }
  
  # Select probes for one target
  select_one_group <- function(
    group_df,
    min_gap,
    max_n_per_id,
    method
  ) {
    
    if (method == "lexicographic") {
      
      group_df <- group_df %>%
        dplyr::arrange(
          dplyr::desc(.abundance),
          dplyr::desc(.eff),
          .start,
          .probe_id
        )
      
    } else {
      
      group_df <- group_df %>%
        dplyr::arrange(
          dplyr::desc(.score),
          dplyr::desc(.abundance),
          dplyr::desc(.eff),
          .start,
          .probe_id
        )
    }
    
    selected_idx <- integer(0)
    
    selected_df <- group_df[
      0,
      ,
      drop = FALSE
    ]
    
    for (i in seq_len(nrow(group_df))) {
      
      if (
        !is.null(max_n_per_id) &&
        length(selected_idx) >= max_n_per_id
      ) {
        break
      }
      
      candidate_start <- group_df$.start[i]
      candidate_end <- group_df$.end[i]
      
      if (
        is_compatible(
          candidate_start,
          candidate_end,
          selected_df,
          min_gap
        )
      ) {
        
        selected_idx <- c(
          selected_idx,
          i
        )
        
        selected_df <- group_df[
          selected_idx,
          ,
          drop = FALSE
        ]
      }
    }
    
    return(
      group_df[
        selected_idx,
        ,
        drop = FALSE
      ]
    )
  }
  
  # Apply selection independently to each target
  selected <- df %>%
    dplyr::group_by(
      .identifier
    ) %>%
    dplyr::group_split(
      .keep = TRUE
    ) %>%
    lapply(
      select_one_group,
      min_gap = min_gap,
      max_n_per_id = max_n_per_id,
      method = method
    ) %>%
    dplyr::bind_rows() %>%
    dplyr::select(
      -.identifier,
      -.probe_id,
      -.start,
      -.len,
      -.eff,
      -.abundance,
      dplyr::any_of(
        c(
          ".abundance_scaled",
          ".eff_scaled",
          ".score"
        )
      )
    ) %>%
    dplyr::arrange(
      .data[[id_col]],
      .data[[start_col]]
    ) %>%
    dplyr::relocate(
      dplyr::all_of(probe_id_col)
    )
  
  return(selected)
}