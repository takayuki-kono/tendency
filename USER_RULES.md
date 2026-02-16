# User Global Rules (Project Level)

## Command Execution Rules
- **NEVER execute .bat files automatically.**
- Always create or update .bat files, then ask the user to execute them manually.
- This rule applies to all future interactions in this workspace.

## Prompt-Driven Development Flow (CRITICAL)
Follow this workflow for every new request or feature:
1. **Record Prompt (APPEND ONLY)**: ALWAYS **APPEND** the user's prompt verbatim to `docs/prompt.md`. Do not add headers/dates/separators, but ensure there is a single newline separator between prompts.
2. **Update Specs**: Update relevant documentation in `docs/*.md` first. The content of `docs/prompt.md` is the source of truth and overrides existing specs.
3. **Implement**: Implement code changes ONLY based on the updated specification files.
