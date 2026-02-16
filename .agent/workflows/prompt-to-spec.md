---
description: Update documentation from prompt and implement changes based on specs
---

Follow these steps for any major feature request or rule change:

1.  **Append to Prompt History**:
    Append the current user request/prompt verbatim to `d:\tendency\docs\prompt.md`.
    (If the file does not exist, create it).

2.  **Update Specification**:
    Update the relevant markdown specification files in `d:\tendency\docs\` based on the content of `d:\tendency\docs\prompt.md`.
    **Important**: The content of `prompt.md` takes precedence over existing specs. If there is a conflict, `prompt.md` wins.

3.  **Implement Changes**:
    Implement the changes in the codebase based on the updated specification files.
    Do not implement directly from the prompt unless the spec has been updated first.
