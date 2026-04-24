# Custom vitals-compatible scorer that uses the local `claude` CLI as the
# grading model. Bypasses ellmer entirely so that scoring runs through the
# user's existing Claude Code session auth (subscription/keychain) instead of
# requiring an ANTHROPIC_API_KEY.
#
# Drop-in replacement for `vitals::model_graded_qa()`. The scorer it returns
# implements the same interface that `vitals::Task$new(scorer = ...)` expects:
# `function(samples, ..., scorer_chat = NULL) -> list(score, scorer_metadata)`.

library(glue)
library(purrr)

# Reproduces vitals::qa_default_template (R/scorer-model.R)
claude_code_qa_template <- function() {
  "You are assessing a submitted answer on a given task based on a criterion.
  [BEGIN DATA]
  ***
  [Task]: {input}
  ***
  [Submission]: {answer}
  ***
  [Criterion]: {criterion}
  ***
  [END DATA]
  Does the submission meet the criterion?
  {instructions}"
}

# Reproduces vitals::qa_default_instructions
claude_code_qa_instructions <- function(partial_credit = FALSE) {
  partial_letter <- if (partial_credit) ", P, or " else " or "
  partial_prompt <- if (partial_credit) {
    '"P" for partially correct answers,'
  } else {
    ""
  }
  glue::glue(
    "After assessing the submitted answer, reply with 'GRADE: $LETTER' where
    LETTER is one of C{partial_letter}I.
    Please choose ONE option: either 'C' for correct answers, {partial_prompt}
    or 'I' for incorrect answers.
    First **briefly** explain your reasoning, then end with GRADE: $LETTER.
    Do not format the grading string and do not include any punctuation or
    exposition after it."
  )
}

#' Heuristic: does the captured output look like a transient/retryable error?
#'
#' Returns TRUE for rate-limit (429), service-overloaded (529), and common
#' transient network/server messages. We don't have structured error codes
#' from `claude -p`, just text mixed from stdout/stderr — so this is a
#' string match against known phrases.
is_retryable_failure <- function(out_text) {
  if (length(out_text) == 0 || all(is.na(out_text))) return(FALSE)
  txt <- paste(out_text, collapse = " ")
  patterns <- c(
    "rate limit", "rate-limit", "rate_limit",
    "too many requests",
    "\\b429\\b", "\\b529\\b",
    "overloaded", "overload",
    "service unavailable", "\\b503\\b",
    "temporarily unavailable",
    "try again",
    "internal server error", "\\b500\\b"
  )
  any(vapply(patterns, function(p) grepl(p, txt, ignore.case = TRUE),
             logical(1)))
}

#' Invoke the local `claude` CLI in non-interactive mode and return its text.
#'
#' Uses the user's existing Claude Code authentication (keychain/OAuth or
#' ANTHROPIC_API_KEY). The prompt is piped via stdin to avoid shell-escaping
#' issues with code samples that contain quotes/newlines.
#'
#' Retries on transient failures (rate limits, overloads, 5xx) with
#' exponential backoff + jitter so concurrent workers don't all retry in
#' lockstep. Non-retryable failures (e.g. malformed prompt, auth error)
#' fail fast with a single attempt.
#'
#' @param prompt User text to send.
#' @param model Claude model identifier.
#' @param system_prompt System prompt prepended to the call.
#' @param timeout_sec Timeout per attempt (passed to `system2`).
#' @param max_attempts Total attempts including the first try. Default 4
#'   (initial + 3 retries → ~2s, 8s, 30s waits).
#' @return The combined stdout/stderr text on success, or NA_character_ if
#'   all attempts failed.
call_claude_code <- function(
  prompt,
  model = "claude-opus-4-7",
  system_prompt = "You are a precise grading assistant. Respond with brief reasoning and the exact requested format. Do not use any tools.",
  timeout_sec = 180,
  max_attempts = 4L
) {
  args <- c(
    "--print",
    "--model", shQuote(model),
    "--system-prompt", shQuote(system_prompt),
    "--output-format", "text",
    "--no-session-persistence"
  )

  # Backoff schedule for retries 1..3 (in seconds before the *next* attempt).
  # Add 0..1s jitter so parallel workers desync.
  backoff_base <- c(2, 8, 30)

  for (attempt in seq_len(max_attempts)) {
    out <- tryCatch(
      system2(
        "claude",
        args = args,
        input = prompt,
        stdout = TRUE,
        stderr = TRUE,
        timeout = timeout_sec
      ),
      error = function(e) {
        structure(
          paste("R-side system2 error:", e$message),
          status = -1L
        )
      }
    )

    status <- attr(out, "status")
    failed <- (!is.null(status) && !identical(status, 0L)) ||
      length(out) == 0 || all(is.na(out))

    if (!failed) {
      return(paste(out, collapse = "\n"))
    }

    retryable <- is_retryable_failure(out)
    last_attempt <- attempt == max_attempts

    if (!retryable || last_attempt) {
      warning(glue::glue(
        "claude -p failed (attempt {attempt}/{max_attempts}, status={status %||% 'NA'}, retryable={retryable}): ",
        "{paste(utils::head(out, 5), collapse=' | ')}"
      ))
      return(NA_character_)
    }

    wait <- backoff_base[min(attempt, length(backoff_base))] + stats::runif(1, 0, 1)
    message(glue::glue(
      "claude -p transient failure (attempt {attempt}/{max_attempts}); retrying in {round(wait, 1)}s..."
    ))
    Sys.sleep(wait)
  }

  NA_character_
}

# Reproduces vitals::qa_extract_grade
extract_grade <- function(
  response,
  pattern = "(?i)GRADE\\s*:\\s*([CPI])(.*)$"
) {
  if (is.na(response)) return(NA_character_)
  m <- regmatches(response, regexec(pattern, response, perl = TRUE))[[1]]
  if (length(m) < 2 || is.na(m[2]) || !nzchar(m[2])) return(NA_character_)
  toupper(m[2])
}

# Reproduces vitals::process_grades
process_grades_local <- function(grades, partial_credit) {
  unique_grades <- unique(grades[!is.na(grades)])
  is_ipc <- any(c("I", "C") %in% unique_grades) &&
    all(unique_grades %in% c("I", "P", "C"))
  if (!is_ipc) return(grades)
  levels <- if (partial_credit) c("I", "P", "C") else c("I", "C")
  factor(grades, levels = levels, ordered = TRUE)
}

#' Build a vitals-compatible scorer that grades via `claude -p`.
#'
#' @param partial_credit Whether to allow "P" (partially correct) grades.
#' @param model Claude model identifier (default "claude-opus-4-7").
#' @param grade_pattern Regex used to pull the grade letter out of the response.
#' @param workers Number of parallel `claude -p` processes. 1 = sequential.
#'   Higher values use `furrr::future_map_chr` if `furrr`+`future` are installed.
#' @return A function with signature `function(samples, ..., scorer_chat = NULL)`
#'   returning `list(score, scorer_metadata)` — drop-in for `model_graded_qa()`.
model_graded_qa_claude_code <- function(
  partial_credit = TRUE,
  model = "claude-opus-4-7",
  grade_pattern = "(?i)GRADE\\s*:\\s*([CPI])(.*)$",
  workers = 1L
) {
  force(partial_credit)
  force(model)
  force(grade_pattern)
  force(workers)

  function(samples, ..., scorer_chat = NULL) {
    template <- claude_code_qa_template()
    instructions <- claude_code_qa_instructions(partial_credit)

    prompts <- purrr::map_chr(seq_len(nrow(samples)), function(i) {
      glue::glue(
        template,
        input = samples$input[[i]],
        answer = samples$result[[i]],
        criterion = samples$target[[i]],
        instructions = instructions
      )
    })

    message(glue::glue(
      "Scoring {length(prompts)} sample(s) via `claude -p --model {model}` (workers={workers})..."
    ))

    use_parallel <- workers > 1L &&
      requireNamespace("furrr", quietly = TRUE) &&
      requireNamespace("future", quietly = TRUE)

    if (use_parallel) {
      old_plan <- future::plan(future::multisession, workers = workers)
      on.exit(future::plan(old_plan), add = TRUE)
      responses <- furrr::future_map_chr(
        prompts,
        function(p) call_claude_code(p, model = model),
        .options = furrr::furrr_options(seed = TRUE)
      )
    } else {
      responses <- purrr::map_chr(
        seq_along(prompts),
        function(i) {
          message(glue::glue("  [{i}/{length(prompts)}] scoring..."))
          call_claude_code(prompts[[i]], model = model)
        }
      )
    }

    grades <- purrr::map_chr(responses, extract_grade, pattern = grade_pattern)
    n_failed <- sum(is.na(grades))
    if (n_failed > 0) {
      warning(glue::glue(
        "{n_failed}/{length(grades)} sample(s) returned no extractable grade."
      ))
    }
    scores <- process_grades_local(grades, partial_credit)

    metadata <- purrr::map(seq_along(prompts), function(i) {
      list(
        prompt = prompts[[i]],
        response = responses[[i]],
        grade_pattern = grade_pattern
      )
    })

    # scorer_chat field deliberately omitted — vitals' Task$cbind_scores() and
    # Task$get_cost() both treat it as optional, and we have no ellmer Chat
    # object to return. This means scorer token counts are NOT tracked, which
    # is fine since claude-code subscription usage isn't billed per token.
    list(score = scores, scorer_metadata = metadata)
  }
}
