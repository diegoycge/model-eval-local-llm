# Helper Functions for Shiny App
# ============================================================================

library(ggplot2)
library(ggrepel)
library(scales)
library(dplyr)
library(forcats)
library(tidyr)
library(purrr)
library(tibble)

# Data Helper Functions ------------------------------------------------------

#' Get list of available models for selection
#'
#' @param eval_data Processed evaluation data
#' @return Tibble with model_display and model_join columns
get_available_models <- function(eval_data) {
  eval_data |>
    distinct(model_display, model_join) |>
    arrange(model_display)
}

#' Compute summary statistics for selected models
#'
#' @param eval_data Processed evaluation data
#' @param cost_data Cost data
#' @param selected_models Character vector of model_join IDs
#' @param model_info Model metadata with provider information
#' @return Tibble with summary statistics per model
compute_summary_stats <- function(
  eval_data,
  cost_data,
  selected_models,
  model_info
) {
  eval_data |>
    filter(model_join %in% selected_models) |>
    group_by(model_display, model_join) |>
    summarize(
      total_samples = n(),
      correct = sum(score == "Correct"),
      partially_correct = sum(score == "Partially Correct"),
      incorrect = sum(score == "Incorrect"),
      percent_correct = correct / total_samples,
      .groups = "drop"
    ) |>
    left_join(
      cost_data |> select(model_join, price, input, output),
      by = "model_join"
    ) |>
    left_join(
      model_info |>
        select(model_join, provider, parameters_b, active_parameters_b, machine),
      by = "model_join"
    ) |>
    arrange(desc(percent_correct))
}

# Plotting Functions ---------------------------------------------------------

#' Create performance bar chart
#'
#' @param eval_data Filtered evaluation data with model_display and score columns
#' @return ggplot object
plot_performance <- function(eval_data) {
  # Reorder models by correct count
  eval_data <- eval_data |>
    mutate(
      model_display = fct_reorder(
        model_display,
        score,
        .fun = \(x) sum(x == "Correct", na.rm = TRUE)
      )
    )

  eval_data |>
    ggplot(aes(y = model_display, fill = score)) +
    geom_bar(position = "fill") +
    scale_fill_manual(
      breaks = rev,
      values = c(
        "Correct" = "#6caea7",
        "Partially Correct" = "#f6e8c3",
        "Incorrect" = "#ef8a62"
      )
    ) +
    scale_x_continuous(labels = percent, expand = c(5e-3, 5e-3)) +
    labs(
      x = "Percent",
      y = NULL,
      fill = "Score"
    ) +
    theme_light() +
    theme(
      legend.position = "bottom",
      plot.margin = margin(10, 10, 10, 10),
      axis.title = element_text(size = 14),
      title = element_text(size = 16),
      axis.text = element_text(size = 12),
      legend.text = element_text(size = 12)
    )
}

#' Create parameters vs performance scatter plot for local LLMs
#'
#' Filters to local LLMs (provider == "LM Studio"), with X-axis switchable
#' between active and total parameter count, and points colored by
#' architecture (Dense vs MoE — derived from whether active == total).
#'
#' @param summary_data Summary statistics tibble. Must contain
#'   `parameters_b`, `active_parameters_b`, `provider`, `percent_correct`,
#'   `model_display`.
#' @param x_axis One of "active" or "total". Controls which parameter
#'   column is plotted on the X-axis.
#' @return ggplot object
plot_parameters_vs_performance <- function(summary_data, x_axis = c("active", "total")) {
  x_axis <- match.arg(x_axis)
  x_col <- if (x_axis == "active") "active_parameters_b" else "parameters_b"
  x_label <- if (x_axis == "active") {
    "Active parameters (billions, log scale)"
  } else {
    "Total parameters (billions, log scale)"
  }

  plot_data <- summary_data |>
    mutate(model_display = as.character(model_display)) |>
    filter(provider == "LM Studio") |>
    filter(!is.na(.data[[x_col]]))

  if (nrow(plot_data) == 0) {
    return(
      ggplot() +
        annotate(
          "text", x = 0.5, y = 0.5,
          label = "Select one or more LM Studio models with parameter counts to see this plot.",
          size = 5
        ) +
        theme_void()
    )
  }

  plot_data <- plot_data |>
    mutate(
      architecture = if_else(
        !is.na(parameters_b) &
          !is.na(active_parameters_b) &
          parameters_b > active_parameters_b,
        "MoE",
        "Dense"
      ),
      color = case_when(
        architecture == "Dense" ~ "#1f78b4",
        architecture == "MoE" ~ "#ff7f00"
      )
    )

  n_models <- nrow(plot_data)
  min_seg_length <- if_else(n_models < 10, 1.2, 0.5)

  ggplot(plot_data, aes(.data[[x_col]], percent_correct, color = architecture)) +
    geom_point(size = 5) +
    geom_label_repel(
      aes(
        label = model_display,
        fill = alpha(color, 0.8),
        segment.color = color
      ),
      force = 3,
      max.overlaps = 20,
      size = 7,
      color = "#333333",
      show.legend = FALSE,
      min.segment.length = min_seg_length,
      box.padding = 0.5
    ) +
    scale_color_identity(aesthetics = "segment.color") +
    scale_fill_identity() +
    scale_x_log10(
      labels = \(x) paste0(x, "B"),
      breaks = c(1, 3, 10, 30, 100, 300)
    ) +
    scale_y_continuous(
      labels = label_percent(),
      breaks = breaks_width(0.05)
    ) +
    scale_color_manual(
      values = c("Dense" = "#1f78b4", "MoE" = "#ff7f00")
    ) +
    labs(
      x = x_label,
      y = "Percent Correct",
      color = "Architecture"
    ) +
    theme_light() +
    theme(
      plot.subtitle = element_text(face = "italic", size = 12),
      plot.margin = margin(10, 10, 20, 10),
      axis.title = element_text(size = 14),
      title = element_text(size = 16),
      axis.text = element_text(size = 12),
      legend.position = "bottom",
      legend.text = element_text(size = 12)
    )
}

# Table Functions ------------------------------------------------------------

#' Create pricing and performance table
#'
#' @param summary_data Summary statistics with percent_correct, price, and token usage
#' @param model_info Model metadata with pricing information
#' @return gt table object
create_pricing_table <- function(summary_data, model_info) {
  filtered <- summary_data |>
    filter(provider == "LM Studio")

  if (nrow(filtered) == 0) {
    return(
      tibble::tibble(
        Note = "Select one or more LM Studio (local) models to populate this table."
      ) |>
        gt::gt() |>
        gt::tab_options(table.font.size = gt::px(14))
    )
  }

  filtered |>
    arrange(desc(percent_correct)) |>
    mutate(
      Architecture = if_else(
        !is.na(parameters_b) &
          !is.na(active_parameters_b) &
          parameters_b > active_parameters_b,
        "MoE",
        "Dense"
      )
    ) |>
    select(
      Model = model_display,
      Architecture,
      `Total Params (B)` = parameters_b,
      `Active Params (B)` = active_parameters_b,
      `Input Tokens Used` = input,
      `Output Tokens Used` = output,
      `% Correct` = percent_correct,
      Machine = machine
    ) |>
    gt::gt() |>
    gt::fmt_number(
      columns = c(`Total Params (B)`, `Active Params (B)`),
      decimals = 1,
      drop_trailing_zeros = TRUE
    ) |>
    gt::fmt_number(
      columns = c(`Input Tokens Used`, `Output Tokens Used`),
      decimals = 0,
      use_seps = TRUE
    ) |>
    gt::fmt_percent(
      columns = `% Correct`,
      decimals = 1
    ) |>
    gt::sub_missing(
      columns = c(`Total Params (B)`, `Active Params (B)`, Machine),
      missing_text = "—"
    ) |>
    gt::cols_align(
      align = "left",
      columns = everything()
    ) |>
    gt::tab_header(
      title = "Local model details",
      subtitle = "LM Studio runs only. Sorted by percent correct (descending)."
    ) |>
    gt::data_color(
      columns = `% Correct`,
      palette = c("#ef8a62", "#f6e8c3", "#6caea7"),
      domain = NULL
    ) |>
    gt::tab_options(
      table.font.size = gt::px(14),
      heading.title.font.size = gt::px(18),
      heading.subtitle.font.size = gt::px(14),
      column_labels.font.weight = "bold",
      ihtml.use_pagination = FALSE,
      ihtml.use_page_size_select = FALSE,
      table.width = gt::pct(100)
    )
}

#' Create runtime-configuration comparison table
#'
#' Pivots `runtime_config` (a list-column on model_info) into a wide table
#' with one row per setting and one column per selected model. Setting order
#' matches first-occurrence order across models, which preserves the logical
#' grouping used in models.yaml.
#'
#' @param model_info Model metadata tibble (must contain a `runtime_config`
#'   list-column populated by load_model_info()).
#' @param selected_models Character vector of model_join IDs to include.
#' @return A gt table object.
create_settings_table <- function(model_info, selected_models) {
  filtered <- model_info |>
    filter(model_join %in% selected_models)

  has_any_config <- "runtime_config" %in% names(filtered) &&
    any(map_int(filtered$runtime_config, length) > 0)

  if (!has_any_config) {
    return(
      tibble::tibble(
        Note = "No runtime_config recorded for the selected models. Add a `runtime_config:` block under each model in data/models.yaml to populate this table."
      ) |>
        gt::gt() |>
        gt::tab_options(table.font.size = gt::px(14))
    )
  }

  long <- map_dfr(seq_len(nrow(filtered)), \(i) {
    rc <- filtered$runtime_config[[i]]
    if (length(rc) == 0) return(NULL)
    tibble::tibble(
      Setting = names(rc),
      Value = map_chr(rc, \(v) {
        if (is.null(v)) NA_character_
        else if (length(v) == 0) NA_character_
        else if (length(v) > 1) paste(v, collapse = ", ")
        else as.character(v)
      }),
      Model = as.character(filtered$Name[i])
    )
  })

  setting_order <- unique(long$Setting)
  long$Setting <- factor(long$Setting, levels = setting_order)

  wide <- long |>
    pivot_wider(names_from = Model, values_from = Value)

  wide |>
    gt::gt(rowname_col = "Setting") |>
    gt::tab_header(
      title = "Model runtime configuration",
      subtitle = "From `runtime_config` blocks in data/models.yaml"
    ) |>
    gt::sub_missing(missing_text = "—") |>
    gt::cols_align(align = "left", columns = everything()) |>
    gt::tab_options(
      table.font.size = gt::px(13),
      heading.title.font.size = gt::px(18),
      heading.subtitle.font.size = gt::px(14),
      column_labels.font.weight = "bold",
      stub.font.weight = "bold",
      table.width = gt::pct(100)
    )
}
