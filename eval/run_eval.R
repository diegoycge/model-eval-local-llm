# Runs the `are` eval for models listed in data/models.yaml
# Will skip models that have already been run (by looking in results_rds)
# Combines all rds results into data/data_combined.rds

library(ellmer)
library(vitals)
library(purrr)
library(glue)

# Source helper functions
source(here::here("R/task_definition.R"))
source(here::here("R/data_loading.R"))
source(here::here("R/eval_functions.R"))

# Configuration
YAML_PATH <- here::here("data/models.yaml")
RESULTS_DIR <- here::here("results_rds")
LOG_DIR <- here::here("logs")
# Scorer is configured in R/task_definition.R (uses local `claude -p` CLI).

# Restrict the run to specific model_ids from data/models.yaml. Set to NULL to
# evaluate all unevaluated models in the YAML.
ONLY_RUN_MODEL_IDS <- c("gemma_4_26b_a4b_lmstudio")

# Set up logging
vitals::vitals_log_dir_set(LOG_DIR)

# ============================================================================
# Run Evaluation
# ============================================================================

# Parse YAML configuration
model_configs <- parse_model_configs(YAML_PATH)

# Find unevaluated models
unevaluated <- find_unevaluated_models(model_configs, RESULTS_DIR)

# Filter to the allow-list, if set
if (!is.null(ONLY_RUN_MODEL_IDS)) {
  unknown <- setdiff(ONLY_RUN_MODEL_IDS, names(model_configs))
  if (length(unknown) > 0) {
    stop(glue(
      "ONLY_RUN_MODEL_IDS contains model_id(s) not in YAML: {paste(unknown, collapse=', ')}"
    ))
  }
  unevaluated <- intersect(unevaluated, ONLY_RUN_MODEL_IDS)
}

# Run evaluations if needed
if (length(unevaluated) > 0) {
  message(glue("Running {length(unevaluated)} unevaluated model(s)..."))

  # scorer_chat is passed through but unused — create_are_task() ignores it
  # and uses the local claude CLI instead. See R/claude_code_scorer.R.
  scorer_chat <- NULL

  eval_results <- run_all_evals(
    model_configs = model_configs,
    unevaluated_ids = unevaluated,
    model_eval_fn = model_eval,
    results_dir = RESULTS_DIR,
    scorer_chat = scorer_chat
  )

  # Report failures only
  n_failed <- sum(!eval_results)
  if (n_failed > 0) {
    message(glue("\nWarning: {n_failed} model(s) failed"))
    failed_ids <- names(eval_results)[!eval_results]
    walk(failed_ids, ~ message(glue("  - {model_configs[[.x]]$name}")))
  }
}

# Combine results
combine_results(
  yaml_path = YAML_PATH,
  results_dir = RESULTS_DIR,
  load_model_info_fn = load_model_info,
  load_eval_results_fn = load_eval_results,
  process_eval_data_fn = process_eval_data,
  compute_cost_data_fn = compute_cost_data
)
