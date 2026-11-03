# Combine primary probes with paired secondary/readout probes from the codebook.
# Primary probe copies are assigned to group1 or group2, joined with all valid
# codebook combinations, and flanked by the corresponding left/right readouts.

combine_probes <- function(
    pri_probes,
    grouped_codebook_long,
    group_cols,
    probe_comb_col = "probe_comb_id",
    output_prefix = "combined"
) {
  
  # Check required packages
  required_packages <- c(
    "dplyr",
    "tidyr",
    "stringr"
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
  if (!is.data.frame(pri_probes)) {
    stop(
      "'pri_probes' must be a data.frame or tibble."
    )
  }
  
  if (!is.data.frame(grouped_codebook_long)) {
    stop(
      "'grouped_codebook_long' must be a data.frame or tibble."
    )
  }
  
  if (
    length(group_cols) < 1 ||
    any(is.na(group_cols)) ||
    any(!nzchar(as.character(group_cols)))
  ) {
    stop(
      "'group_cols' must contain at least one valid column name."
    )
  }
  
  group_cols <- as.character(
    group_cols
  )
  
  # Check required columns
  required_pri_cols <- c(
    "identifier",
    "code",
    "probe_name",
    probe_comb_col,
    "probe",
    "probe_count"
  )
  
  required_code_cols <- c(
    "code",
    "group",
    "combination_id",
    "position",
    group_cols
  )
  
  missing_pri_cols <- setdiff(
    required_pri_cols,
    colnames(pri_probes)
  )
  
  missing_code_cols <- setdiff(
    required_code_cols,
    colnames(grouped_codebook_long)
  )
  
  if (length(missing_pri_cols) > 0) {
    stop(
      "Missing required column(s) in 'pri_probes': ",
      paste(missing_pri_cols, collapse = ", ")
    )
  }
  
  if (length(missing_code_cols) > 0) {
    stop(
      "Missing required column(s) in 'grouped_codebook_long': ",
      paste(missing_code_cols, collapse = ", ")
    )
  }
  
  # Assign duplicated primary probes to codebook groups
  pri_probes_group <- pri_probes %>%
    dplyr::mutate(
      group = dplyr::case_when(
        as.character(.data[[probe_comb_col]]) == "1" ~ "group1",
        as.character(.data[[probe_comb_col]]) == "2" ~ "group2",
        TRUE ~ NA_character_
      )
    )
  
  if (any(is.na(pri_probes_group$group))) {
    
    bad_ids <- unique(
      pri_probes_group[[probe_comb_col]][
        is.na(pri_probes_group$group)
      ]
    )
    
    stop(
      "Unsupported probe combination ID(s): ",
      paste(bad_ids, collapse = ", ")
    )
  }
  
  # Join primary probes with all compatible codebook combinations
  joined_df <- pri_probes_group %>%
    dplyr::left_join(
      grouped_codebook_long,
      by = c(
        "code",
        "group"
      ),
      relationship = "many-to-many"
    )
  
  if (any(is.na(joined_df$combination_id))) {
    stop(
      "Some primary probes failed to match 'grouped_codebook_long' ",
      "by code and group."
    )
  }
  
  key_cols <- c(
    "identifier",
    "code",
    "probe_name",
    probe_comb_col,
    "combination_id",
    "group"
  )
  
  # Keep only columns needed for probe construction
  # and remove exact duplicated rows.
  joined_df_unique <- joined_df %>%
    dplyr::select(
      dplyr::all_of(key_cols),
      position,
      probe,
      probe_count,
      dplyr::all_of(group_cols)
    ) %>%
    dplyr::distinct()
  
  # Check that each combination contains only one row
  # for each left/right position.
  position_check <- joined_df_unique %>%
    dplyr::count(
      dplyr::across(
        dplyr::all_of(key_cols)
      ),
      position,
      name = "n_position"
    ) %>%
    dplyr::filter(
      n_position > 1
    )
  
  if (nrow(position_check) > 0) {
    
    print(
      dplyr::slice_head(
        position_check,
        n = 20
      )
    )
    
    stop(
      "Some probe-code combinations contain more than one row ",
      "for the same left/right position."
    )
  }
  
  # Store primary-probe information once for each combination
  base_info <- joined_df_unique %>%
    dplyr::group_by(
      dplyr::across(
        dplyr::all_of(key_cols)
      )
    ) %>%
    dplyr::summarise(
      probe = dplyr::first(probe),
      probe_count = dplyr::first(probe_count),
      .groups = "drop"
    )
  
  # Construct combined sequences independently for each requested column
  combined_list <- lapply(
    group_cols,
    function(current_col) {
      
      tmp <- joined_df_unique %>%
        dplyr::select(
          dplyr::all_of(key_cols),
          position,
          value = dplyr::all_of(
            current_col
          )
        ) %>%
        dplyr::mutate(
          value = as.character(value)
        ) %>%
        tidyr::pivot_wider(
          names_from = position,
          values_from = value
        )
      
      if (!all(c("left", "right") %in% colnames(tmp))) {
        stop(
          "Missing left or right position when combining column: ",
          current_col
        )
      }
      
      tmp <- tmp %>%
        dplyr::left_join(
          base_info %>%
            dplyr::select(
              dplyr::all_of(key_cols),
              probe
            ),
          by = key_cols
        ) %>%
        
        # Build: left readout + primary probe + right readout
        dplyr::mutate(
          combined_value = paste0(
            left,
            probe,
            right
          ),
          
          # Remove temporary left/right tags introduced earlier
          combined_value = stringr::str_replace_all(
            combined_value,
            c(
              "left_" = "",
              "_right" = ""
            )
          )
        ) %>%
        dplyr::select(
          dplyr::all_of(key_cols),
          combined_value
        ) %>%
        dplyr::rename(
          !!paste0(
            output_prefix,
            "_",
            current_col
          ) := combined_value
        )
      
      return(tmp)
    }
  )
  
  # Merge combined sequence columns
  combined_info <- Reduce(
    function(x, y) {
      
      dplyr::left_join(
        x,
        y,
        by = key_cols
      )
    },
    combined_list
  )
  
  # Prepare final combined-probe table
  combined_probes <- base_info %>%
    dplyr::left_join(
      combined_info,
      by = key_cols
    ) %>%
    dplyr::mutate(
      probe_name_info = paste(
        probe_name,
        code,
        combination_id,
        sep = "_"
      )
    ) %>%
    dplyr::select(
      identifier,
      code,
      probe_name,
      dplyr::all_of(probe_comb_col),
      combination_id,
      probe_name_info,
      probe_count,
      dplyr::starts_with(
        paste0(output_prefix, "_")
      )
    )
  
  return(combined_probes)
}
