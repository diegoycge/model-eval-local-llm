# This file defines:
# - create_are_task(): Function to create the evaluation task
# - model_eval(): Core function to evaluate a single model

library(ellmer)
library(vitals)

source(here::here("R/claude_code_scorer.R"))

# Set results directory
results_dir <- here::here("results_rds")

# Scoring runs through the local `claude -p` CLI (Claude Opus 4.7) using the
# user's existing Claude Code session auth. See R/claude_code_scorer.R.
SCORER_CC_MODEL <- "claude-opus-4-7"
SCORER_CC_WORKERS <- 1L  # bump to 4 for ~4x throughput on the 264 grading calls

# Solver concurrency. LM Studio serves one request at a time, so >1 just
# queues client-side and risks request timeouts. Bump for cloud-API models.
SOLVER_MAX_ACTIVE <- 1L

#' Create the ARE evaluation task
#'
#' @param scorer_chat Unused — retained for backward-compatible call signature.
#'   Scoring is performed by `model_graded_qa_claude_code()` via the local
#'   `claude` CLI rather than an ellmer Chat object.
#' @return A Task object configured for ARE evaluation
create_are_task <- function(scorer_chat = NULL) {
  Task$new(
    dataset = are,
    solver = generate(),
    scorer = model_graded_qa_claude_code(
      partial_credit = TRUE,
      model = SCORER_CC_MODEL,
      workers = SCORER_CC_WORKERS
    ),
    epochs = 3, # Run 3 evaluation rounds
    name = "An R Eval"
  )
}

#' Evaluate a model on the ARE dataset
#'
#' @param model API model identifier (e.g., "anthropic/claude-sonnet-4-20250514")
#' @param filename Output filename (without .rds extension). Defaults to model name.
#' @param scorer_chat Chat object used for model-graded scoring
#' @param overwrite Whether to overwrite existing results. Defaults to TRUE.
#' @param ... Additional arguments passed to chat():
#'   - base_url: Custom API endpoint
#'   - api_key: Custom API key
#'   - api_args: List of additional API arguments (e.g., thinking config)
#'
#' @return Invisible NULL. Results saved to results_rds/{filename}.rds
model_eval <- function(
  model,
  filename = model,
  scorer_chat,
  overwrite = TRUE,
  ...
) {
  model_path <- fs::path(results_dir, filename, ext = "rds")

  if (!overwrite & fs::file_exists(model_path)) {
    message(glue::glue("Skipping {model}: file already exists at {model_path}"))
    return(invisible(NULL))
  }

  chat <- chat(name = model, ...)

  are_task <- create_are_task(scorer_chat)
  are_task$eval(
    solver_chat = chat,
    max_active = SOLVER_MAX_ACTIVE,
    on_error = "continue"
  )

  readr::write_rds(are_task, file = model_path)
}
