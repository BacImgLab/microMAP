# Generate all equal two-group partitions of the active readouts for each code.
# Each possible assignment is stored as a separate combination and used
# downstream to construct alternative paired probe configurations.

generate_grouped_codebook_combinations <- function(
    codebook_long,
    code_col = "code",
    group_labels = c("group1", "group2")
) {
  
  # Check required package
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop(
      "generate_grouped_codebook_combinations requires the 'dplyr' package."
    )
  }
  
  # Check input data
  if (!is.data.frame(codebook_long)) {
    stop(
      "'codebook_long' must be a data.frame or tibble."
    )
  }
  
  if (!code_col %in% colnames(codebook_long)) {
    stop(
      "Missing required column in 'codebook_long': ",
      code_col
    )
  }
  
  # Exactly two distinct group labels are required
  if (
    length(group_labels) != 2 ||
    any(is.na(group_labels)) ||
    any(!nzchar(as.character(group_labels))) ||
    anyDuplicated(group_labels)
  ) {
    stop(
      "'group_labels' must contain exactly two distinct group labels."
    )
  }
  
  group_labels <- as.character(
    group_labels
  )
  
  # Generate all equal two-group assignments independently for each code
  grouped_codebook_long <- codebook_long %>%
    dplyr::group_by(
      .data[[code_col]]
    ) %>%
    dplyr::group_modify(
      ~ {
        
        data <- .x
        n_active <- nrow(data)
        
        # Active readouts must be divisible equally between two groups
        if (n_active %% 2 != 0) {
          stop(
            "Code ",
            .y[[code_col]],
            " has an odd number of active readouts: ",
            n_active,
            ". Cannot split them equally into two groups."
          )
        }
        
        group_size <- n_active / 2
        
        # Enumerate all possible choices for group 1.
        # Remaining entries are assigned to group 2.
        group1_indices_list <- utils::combn(
          seq_len(n_active),
          group_size,
          simplify = FALSE
        )
        
        result <- lapply(
          seq_along(group1_indices_list),
          function(i) {
            
            group1_indices <- group1_indices_list[[i]]
            
            data_temp <- data
            
            data_temp$group <- group_labels[2]
            
            data_temp$group[
              group1_indices
            ] <- group_labels[1]
            
            data_temp$combination_id <- paste0(
              "combination",
              i
            )
            
            return(data_temp)
          }
        ) %>%
          dplyr::bind_rows()
        
        return(result)
      }
    ) %>%
    dplyr::ungroup()
  
  return(grouped_codebook_long)
}