# User Global Rules (Project Level)

## Command Execution Rules
- **NEVER execute .bat files automatically.**
- Always create or update .bat files, then ask the user to execute them manually.
- This rule applies to all future interactions in this workspace.

## Version Control Rules (CRITICAL)
- **Git Command Sharing**: After ANY code modification, ALWAYS provide the `git add` and `git commit` commands.
  - Do NOT auto-execute git commands.
  - Commit messages MUST be in Japanese.
  - Always include the `--author` flag based on the model (Gemini/Claude).
  - This must be done proactively without the user asking.

## Prompt-Driven Development Flow (CRITICAL)
Follow this workflow for every new request or feature:
1. **Record Prompt (APPEND ONLY)**: ALWAYS **APPEND** the user's prompt verbatim to `docs/prompt.md`. Do not add headers/dates/separators, but ensure there is a single newline separator between prompts.
2. **Update Specs & Docs (MANDATORY)**: Update relevant documentation in `docs/*.md` first.
    - **ALWAYS keep docs up-to-date**: Even for small code changes, reflect them in `pipeline_specs.md` or `03_training_workflow.md`.
    - The content of `docs/prompt.md` is the source of truth and overrides existing specs.
3. **Implement**: Implement code changes ONLY based on the updated specification files.
