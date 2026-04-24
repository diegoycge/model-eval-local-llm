# This file defines:
# - create_are_task(): Function to create the evaluation task
# - model_eval(): Core function to evaluate a single model

library(ellmer)
library(vitals)
library(jsonlite)

source(here::here("R/claude_code_scorer.R"))

# Set results directory
results_dir <- here::here("results_rds")

# Scoring runs through the local `claude -p` CLI (Claude Opus 4.7) using the
# user's existing Claude Code session auth. See R/claude_code_scorer.R.
SCORER_CC_MODEL <- "claude-opus-4-7"
SCORER_CC_WORKERS <- 2L  # bump to 4 for ~4x throughput on the 264 grading calls

# Solver concurrency. LM Studio serves one request at a time, so >1 just
# queues client-side and risks request timeouts. Bump for cloud-API models.
SOLVER_MAX_ACTIVE <- 2L

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
  sidecar_path <- fs::path(results_dir, filename, ext = "json")

  if (!overwrite & fs::file_exists(model_path)) {
    message(glue::glue("Skipping {model}: file already exists at {model_path}"))
    return(invisible(NULL))
  }

  chat_args <- list(...)

  # Snapshot LM Studio loader state BEFORE running. Skip for non-local URLs.
  lm_state <- NULL
  if (!is.null(chat_args$base_url) &&
      grepl("localhost|127\\.0\\.0\\.1", chat_args$base_url)) {
    state_url <- paste0(sub("/v1/?$", "", chat_args$base_url), "/api/v0/models")
    lm_state <- tryCatch(
      jsonlite::fromJSON(state_url, simplifyVector = FALSE),
      error = function(e) list(error = conditionMessage(e))
    )
  }

  run_started <- Sys.time()

  chat <- chat(name = model, ...)

  are_task <- create_are_task(scorer_chat)
  are_task$eval(
    solver_chat = chat,
    max_active = SOLVER_MAX_ACTIVE,
    on_error = "continue"
  )

  run_completed <- Sys.time()

  readr::write_rds(are_task, file = model_path)

  # Strip any function-typed entries (e.g. credentials) before serializing.
  chat_args_safe <- chat_args[!vapply(chat_args, is.function, logical(1))]

  sidecar <- list(
    model_id = filename,
    api_model_id = model,
    base_url = chat_args_safe$base_url,
    api_args_sent = chat_args_safe$api_args,
    run_started = format(run_started, "%Y-%m-%dT%H:%M:%S%z"),
    run_completed = format(run_completed, "%Y-%m-%dT%H:%M:%S%z"),
    run_duration_seconds = as.numeric(
      difftime(run_completed, run_started, units = "secs")
    ),
    epochs = 3L,
    solver_max_active = SOLVER_MAX_ACTIVE,
    scorer_model = SCORER_CC_MODEL,
    scorer_workers = SCORER_CC_WORKERS,
    lm_studio_state = lm_state,
    sessionInfo = paste(
      utils::capture.output(utils::sessionInfo()),
      collapse = "\n"
    )
  )

  jsonlite::write_json(
    sidecar,
    sidecar_path,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null"
  )
  message(glue::glue("Wrote sidecar metadata: {sidecar_path}"))
}
