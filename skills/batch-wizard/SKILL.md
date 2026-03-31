---
name: batch-wizard
description: >
  Guided multi-phase workflow to design, red-team, optimize, and generate
  a batch of claude-queue jobs. Walks the user through scoping, target
  discovery, prompt design, variable extraction, CSV generation, priority
  config, job generation, adversarial review, token efficiency, and launch.
  Triggers on: "create a batch", "design queue jobs", "batch wizard",
  "plan a batch run", "generate jobs for", "queue a bunch of",
  "batch workflow".
allowed-tools: [Bash, Read, Glob, Grep, Write, Edit, Agent]
argument-hint: "[project path or short description of the work]"
disable-model-invocation: false
---

# Batch Job Wizard

Guide the user through designing and generating a batch of claude-queue
jobs. Follow each phase in order. **Do NOT skip phases** unless the user
explicitly asks to. Ask for confirmation before advancing to the next
phase.

**Flexible entry:** If the user already has a template, CSV, or partial
work, acknowledge what exists and pick up from the appropriate phase
rather than forcing them to restart.

## Progress Tracker

At the start of **every phase** (including Phase 1), print the progress
tracker below. Mark completed phases with `[x]`, the current phase with
`-->`, and future phases with `[ ]`. This gives the user a visual map of
where they are and what decisions are coming.

```
Batch Wizard Progress:
  [x]  1. Scope
  [x]  2. Target Discovery
  -->  3. Prompt Design
  [ ]  4. Template Variables
  [ ]  5. CSV Generation
  [ ]  6. Priority & Configuration
  [ ]  7. Generate (Dry Run)
  [ ]  8. Red Team Review
  [ ]  9. Token Efficiency Review
  [ ] 10. Review & Launch
```

When re-entering a phase (e.g., looping back from Phase 8 to revise the
template), mark the revisited phase with `-->` and keep earlier phases
as `[x]`. Phases after the current one revert to `[ ]` only if their
output is invalidated by the revision.

---

## Phase 1: Scope

Goal: Understand what the user wants to accomplish and where.

- Ask: What project/directory are these jobs for?
- Ask: What is the goal? (refactor, review, documentation, tests, migration, etc.)
- Ask: Roughly how many targets do you expect?
- Explore the project briefly (read CLAUDE.md, scan directory structure) to
  build context for later phases.

Confirm scope before proceeding.

---

## Phase 2: Target Discovery

Goal: Build the concrete list of targets (files, functions, modules, etc.)
that each job will operate on.

- Based on Phase 1, use Glob/Grep/Agent to enumerate candidates.
- Present the list to the user. Include the count.
- Ask: Should any targets be excluded? Are there any missing?
- Finalize the target list.

---

## Phase 3: Prompt Design

Goal: Craft a high-quality prompt for one representative target.

- Pick a representative target (ideally one of medium complexity).
- Draft a complete prompt — title, context, step-by-step instructions,
  expected output — as it would appear in a queue `.md` file body.
- Show it to the user for feedback.
- Iterate until they approve the prompt.

**Tip:** Write the prompt as if addressing a capable colleague who has
never seen this codebase and has no conversation history. Be specific
about what to read, what to change, and what to verify.

---

## Phase 4: Template Variables

Goal: Parameterize the approved prompt so it works across all targets.

- Identify which parts of the prompt vary per target (filenames, paths,
  function names, module names, etc.).
- Replace them with `{{variable}}` placeholders.
- Show the user the parameterized template side-by-side with the original
  to confirm nothing was lost.
- List the variables and their meanings.

---

## Phase 5: CSV Generation

Goal: Produce the data file that drives batch generation.

- Generate a CSV (or TSV) with one row per target and columns matching
  each `{{variable}}`.
- Show a preview of the first 5 and last 2 rows for confirmation.
- Report the total row count and verify it matches the Phase 2 target list.

---

## Phase 6: Priority & Configuration

Goal: Set the YAML frontmatter values for the batch.

- Ask about or recommend:
  - `priority` / `--base-priority` / `--priority-step` — explain that
    without `--base-priority`, all jobs get the same priority and
    execution order becomes non-deterministic.
  - `model` — whether a specific model is needed or the default suffices.
  - `max_retries` — recommend `-1` (unlimited) for idempotent tasks,
    `3` for tasks with side effects.
  - `working_directory` — confirm the absolute path.
  - `context_files` — any files every job should have loaded.
- Show the complete frontmatter block for approval.

---

## Phase 7: Generate (Dry Run)

Goal: Write the template and CSV, validate, and preview before committing.

- Write the template to `~/.claude-queue/bank/<name>.md`
- Write the CSV alongside it or to a temp location.
- Run: `claude-queue batch validate <name> --data <csv>`
- Run: `claude-queue batch generate <name> --data <csv> --base-priority <N> [--priority-step <S>] --dry-run`
- Show the dry-run output for review.
- Ask: Does everything look right?

---

## Phase 8: Red Team Review

Goal: Stress-test the prompt and batch design before committing real
compute time. Each job will run in a **clean context window** with no
memory of this conversation, so the prompt must stand completely on its
own.

Walk through each of the following questions with the user. For each one,
explain *why* it matters — not just the question but the failure mode it
prevents.

### 8a. Scope & Guardrails

> Could Claude, starting from a blank context with only this prompt,
> wander off into a rabbit hole and never return useful output?

Common failure modes:
- Vague verbs ("improve", "refactor", "clean up") without success criteria
- No explicit boundary on what files/directories to touch
- No instruction to stop and report rather than guess when uncertain

If the answer is yes, suggest adding scoping guardrails to the prompt.
Acknowledge the tradeoff: tighter scope limits creativity, but
well-guided execution has a higher probability of producing useful output
across dozens of jobs.

### 8b. Hidden Assumptions

> Are there things you know about this project — conventions, gotchas,
> recent decisions, tribal knowledge — that the prompt doesn't mention?

Think of it this way: if you handed this prompt to a competent intern on
their first day, what context would you need to give them verbally?
That context should be in the prompt.

Examples: "we use tabs not spaces", "don't modify the generated files in
`build/`", "the `_legacy` suffix means do-not-touch", "PR titles must
follow conventional commits".

### 8c. Parallelism & Dependencies

> Can these jobs run in parallel, or does one job's output affect another?

`claude-queue` currently executes one job at a time, but this may change
in the future. Even with serial execution, consider:
- Does job N modify a file that job N+1 also reads? (merge conflicts)
- Does job order matter? (e.g., creating an interface before implementing it)
- Should certain jobs be grouped at a higher priority to run first?

If dependencies exist, discuss whether to split into separate batches
with different base priorities or add explicit ordering.

### 8d. Idempotency

> If a job runs twice (due to crash recovery), will the second run
> produce a broken result?

`claude-queue` has at-least-once semantics. If the daemon crashes
mid-execution, the job will re-run. Flag any jobs that create resources,
send messages, open PRs, or make API calls — these need idempotency
guards in the prompt (e.g., "check if the PR already exists before
creating one").

After this review, offer to revise the template. If changes are made,
re-run the dry run from Phase 7 to confirm.

---

## Phase 9: Token Efficiency Review

Goal: Minimize wasted tokens across the batch without sacrificing quality.
Every inefficiency is multiplied by the number of jobs.

Walk through these considerations with the user:

### 9a. Context Files

- Are the `context_files` in frontmatter actually needed by every job?
  Files loaded via `context_files` consume input tokens on every run.
- Could some context be inlined in the prompt instead (a 3-line snippet
  vs. loading a 500-line file)?
- Conversely, does the prompt ask Claude to "read file X" when it could
  be a `context_file` instead (saving a tool-call round trip)?

### 9b. Prompt Verbosity

- Is the prompt longer than it needs to be? Look for:
  - Repeated instructions (said two different ways)
  - Excessive examples (one clear example beats three)
  - Boilerplate that adds no information
- Every extra token in the prompt is multiplied by the job count. For a
  batch of 100 jobs, trimming 500 tokens from the prompt saves 50,000
  input tokens.

### 9c. Model Selection

- Does every job need the most capable (and most expensive) model?
- Could a smaller/faster model handle straightforward tasks (e.g.,
  simple renames, formatting fixes) while reserving the larger model for
  complex reasoning?
- Remind the user that `model:` can be set per-template in frontmatter.

### 9d. Output Scope

- Does the prompt constrain what Claude outputs? Without guidance, Claude
  may produce verbose explanations, summaries, or commentary that consume
  output tokens without adding value.
- Consider adding: "Do not explain your changes. Just make them." or
  "Keep your response under 200 words" where appropriate.

Summarize estimated token impact if changes are made (rough order of
magnitude is fine). Offer to revise the template.

---

## Phase 10: Review & Launch

Goal: Final review and optional queue start.

- Run: `claude-queue batch generate <name> --data <csv> --base-priority <N> [--priority-step <S>]`
- Run: `claude-queue status --detailed` — show what will execute.
- Report: total job count, estimated run time (based on ~1-2 min/job for
  typical prompts), priority ordering.
- Ask: Ready to start? Or do you want to review individual job files first?
- If the user says go: `claude-queue start`
- Remind the user they can monitor progress with `claude-queue status`
  and cancel individual jobs with `claude-queue cancel <id>`.
