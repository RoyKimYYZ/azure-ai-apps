---
name: code-quality-modular-design
description: "Best practices for creating modular, well-designed, robust, maintainable, and easy-to-read code. Covers decomposition, naming, boundaries, error handling, testing, observability, and practical implementation workflow. WHEN: refactor code, improve design, modularize code, simplify logic, reduce coupling, improve readability, add tests, review maintainability, clean up architecture, improve error handling, improve code quality, make code easier to understand, design new module, split large class, split large function, review robustness, make agent code maintainable."
---

# Code Quality: Modular Design And Maintainability

Use this skill when the goal is not just to make code work, but to make it easier to understand, safer to change, and more resilient over time.

## Primary Goals

- Keep code easy to read without requiring extra explanation.
- Break systems into focused units with clear responsibilities.
- Prefer explicit boundaries, data flow, and failure behavior.
- Make common changes local and predictable.
- Add only as much abstraction as the current use case justifies.

## What Good Code Looks Like

Good code in this repo should usually have these properties:

- A reader can identify the main responsibility of each file, class, and function quickly.
- Names describe intent, not implementation trivia.
- Dependencies flow in a clear direction.
- Inputs, outputs, and side effects are visible at the call site.
- Error handling preserves context instead of hiding failures.
- Tests exercise behavior at the right level rather than overfitting internals.
- Logging and diagnostics help explain production behavior without cluttering core logic.

## Design Principles

### 1. Prefer Focused Units

- A module should have one reason to change.
- A class should represent one concept, not a pile of related tasks.
- A function should do one meaningful thing and return a clear result.
- Split code when one unit mixes orchestration, transformation, I/O, and presentation.

Good signs:

- The name of the unit matches what it actually does.
- The function can be summarized in one sentence.
- The caller does not need to know internal implementation details.

### 2. Optimize For Reading Order

- Put the most important path first.
- Keep control flow shallow where practical.
- Use guard clauses to handle invalid or exceptional cases early.
- Group related logic so readers do not need to jump across files or branches to understand one behavior.

Prefer:

- clear branch conditions
- early returns
- small helper functions with meaningful names

Avoid:

- deeply nested conditional logic
- long functions with mixed levels of abstraction
- helpers with vague names such as `process_data`, `handle`, `run_logic`, or `do_work`

### 3. Make Boundaries Explicit

- Separate domain logic from transport, persistence, UI, and infrastructure concerns.
- Pass dependencies in clearly rather than reaching for global state.
- Keep parsing, validation, computation, and side effects distinct when possible.
- Make data shape changes obvious at module boundaries.

Examples of healthy boundaries:

- a pure transformation function for mapping model output to UI state
- a repository object for persistence details
- a client abstraction for external service calls
- a dedicated adapter for framework-specific glue code

### 4. Prefer Stable Interfaces Over Clever Internals

- Design public functions and classes to be boring and predictable.
- Keep optional behavior explicit through parameters or focused wrapper functions.
- Add abstractions only after identifying repeated change patterns, not to speculate about future reuse.
- Favor composition over inheritance unless inheritance clearly models the domain.

### 5. Keep Naming Concrete

- Choose names that reflect business meaning or architectural role.
- Encode the distinction between domain entities, adapters, clients, repositories, handlers, and views.
- Use verbs for actions and nouns for stateful concepts.
- Make boolean names read like predicates.

Prefer names like:

- `load_agent_config`
- `FitnessProfileRepository`
- `build_chat_response`
- `is_profile_complete`

Avoid names like:

- `helper`
- `manager`
- `data`
- `misc`
- `thing`
- `do_stuff`

## Robustness Rules

### Error Handling

- Catch exceptions where you can add context or recover meaningfully.
- Do not swallow exceptions silently.
- Avoid broad `except Exception` unless you re-raise or convert it with useful context.
- Make failure modes visible in logs, return types, or user-facing error paths.

Prefer:

- validation close to input boundaries
- domain-specific error messages
- structured logging with key identifiers

### State And Side Effects

- Keep hidden mutation to a minimum.
- Prefer returning values over mutating distant state.
- Make I/O operations explicit in function signatures or surrounding orchestration.
- Be careful with caches, singletons, and cross-request shared state; document why they exist.

### Performance

- Choose simple code first, then optimize measured bottlenecks.
- Avoid unnecessary work inside loops, repeated parsing, or duplicate network/database calls.
- Do not introduce performance complexity without evidence that it matters.
- If a tradeoff hurts readability, justify it with a real constraint.

## Testing Guidance

- Test behavior, invariants, and error paths, not private implementation details.
- Add focused unit tests for pure logic and narrow integration tests for boundaries.
- Mock external systems at the edge, not everywhere.
- Keep test setup small enough that failures are easy to diagnose.
- If code is hard to test, treat that as a design smell and simplify the design first.

Good tests usually:

- state intent clearly
- use realistic but minimal fixtures
- cover success and failure paths
- avoid fragile assertions on incidental formatting or call ordering unless that behavior matters

## Observability Guidance

- Add logs where they help explain decisions, failures, latency, or state transitions.
- Log identifiers and outcomes, not just generic messages.
- Do not let logging overwhelm the main logic.
- Prefer reusable diagnostics patterns over ad hoc print-style debugging.

## Refactoring Heuristics

Refactor when you see:

- one function mixing parsing, orchestration, I/O, and formatting
- duplicated branching logic in multiple places
- repeated stringly-typed lookups or shape conversions
- a class growing unrelated responsibilities
- tests that require excessive mocking or setup
- names that no longer match the code after feature growth

Useful refactor moves:

- extract a pure function for business rules
- isolate framework glue from domain behavior
- introduce a small data model for repeated dictionaries
- split one class into repository, service, and presentation responsibilities when those concerns are clearly different
- replace a boolean mode flag with separate functions when the behaviors diverge

## Anti-Patterns To Avoid

- God classes and god functions that own too many responsibilities
- Hidden side effects such as writing state, hitting I/O, or mutating shared objects unexpectedly
- Tight coupling between UI, persistence, model calls, and business rules
- Premature abstraction introduced without repeated pressure from real changes
- Weak naming that forces the reader to inspect implementation to understand purpose
- Catch-all exception handling that hides failure causes

## Copilot Workflow

When using this skill, follow this workflow:

1. Identify the unit of change.
   Decide whether the problem is in domain logic, UI flow, persistence, integration, or configuration.

2. Find the real responsibility boundary.
   Before editing, determine what code should own the behavior and what code should only call it.

3. Prefer the smallest design improvement that fixes the root problem.
   Do not refactor broadly just because the code could be cleaner.

4. Make the data flow more explicit.
   Clarify inputs, outputs, errors, and side effects.

5. Validate proportionally.
   Run the narrowest useful lint, type, test, or runtime check for the area changed.

6. Explain changes in terms of maintainability.
   Summaries should describe what responsibility was clarified, what coupling was reduced, or what failure path was made safer.

## Review Checklist

Before finishing, ask:

- Is each changed unit easier to explain than before?
- Did the change reduce or increase coupling?
- Are names more specific and useful?
- Are side effects and dependencies clearer?
- Is failure behavior explicit?
- Is the level of abstraction consistent within each function?
- Are tests and validation proportional to the risk of the change?

## Apply This Skill Especially To

- Python services and backend modules
- SDK or library-style reusable code
- Streamlit or web application logic
- AI agent orchestration code
- RAG and pipeline code
- test design and refactoring work

Use this skill to make code simpler, not more ornate. The target is durable clarity.