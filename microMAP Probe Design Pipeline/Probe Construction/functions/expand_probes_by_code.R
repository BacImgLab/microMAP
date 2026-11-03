# Expand each primary probe across all codebook entries.
# Each probe is duplicated once for every code, producing all possible
# probe-code combinations for downstream assignment.

expand_probes_by_code <- function(
    pri_probes,
    codebook_df,
    code_col = "code"
) {
  
  # Check input data
  if (!is.data.frame(pri_probes)) {
    stop(
      "'pri_probes' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(codebook_df)) {
    stop(
      "'codebook_df' must be a data.frame or tibble."
    )
  }
  
  if (!code_col %in% colnames(codebook_df)) {
    stop(
      "Missing required column in 'codebook_df': ",
      code_col
    )
  }
  
  if (nrow(pri_probes) == 0) {
    stop(
      "'pri_probes' contains no probes."
    )
  }
  
  if (nrow(codebook_df) == 0) {
    stop(
      "'codebook_df' contains no codes."
    )
  }
  
  # Check code uniqueness
  if (anyDuplicated(codebook_df[[code_col]]) > 0) {
    stop(
      "Duplicated codes were found in 'codebook_df'."
    )
  }
  
  code_count <- nrow(
    codebook_df
  )
  
  expected_n <- nrow(pri_probes) *
    code_count
  
  # Duplicate each probe once for every code
  expanded_probes <- pri_probes[
    rep(
      seq_len(nrow(pri_probes)),
      each = code_count
    ),
    ,
    drop = FALSE
  ]
  
  # Assign all codebook codes to each duplicated probe set
  expanded_probes[[code_col]] <- rep(
    as.character(
      codebook_df[[code_col]]
    ),
    times = nrow(pri_probes)
  )
  
  # Check expansion result
  if (nrow(expanded_probes) != expected_n) {
    stop(
      "Expanded probe number is incorrect."
    )
  }
  
  return(expanded_probes)
}