---
applyTo: "**/*.py"
---

# Python Code Quality

- **Style and linting:** Follow `ruff` defaults. Keep lines readable; don't sacrifice clarity for brevity. Avoid wildcard imports and unused variables.
- **Type annotations:** Add type hints to new functions and methods. Prefer built-in generics (`list[str]`, `dict[str, Any]`) over `typing` equivalents where Python 3.10+ syntax is available. Don't retroactively annotate large existing functions unless the task touches them.
- **Type checking:** Code should pass `mypy` without ignoring errors unless the external library has no stubs — in that case use a targeted `# type: ignore[<code>]` with the specific error code, not a blanket ignore.
- **Naming:** Use `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module-level constants. Names should describe intent, not implementation.
- **Error handling:** Catch specific exception types. Avoid bare `except:`. Use `logger.exception(...)` (not `print`) so errors surface in the existing log pipeline.
- **Imports:** Group stdlib → third-party → local, separated by blank lines. Don't import inside functions unless needed to break a circular dependency.
- **Don't over-engineer:** These practices apply to new code and code that is being actively changed. Don't refactor unrelated passing code just to improve style.
