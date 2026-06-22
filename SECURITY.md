# Security Policy

Astro Suite is a research and engineering codebase for deterministic mission-analysis workflows. It
is not flight software and is not intended for operational spacecraft command authority.

## Reporting Vulnerabilities

Please do not open a public issue for sensitive security reports. Use GitHub private vulnerability
reporting if it is enabled for this repository, or contact the maintainer through their GitHub
profile with a minimal description and a safe way to coordinate details.

Useful reports include:

- Unsafe command execution paths.
- Path traversal or artifact overwrite issues.
- Secret exposure in examples, logs, tests, or documentation.
- Optional backend adapter behavior that silently downgrades fidelity or bypasses validation.
- Assistant workflow behavior that executes outside the allow-listed registry or skips approval.

## Scope Boundaries

The assistant layer is intentionally constrained. Natural-language prompts compile to typed plans,
and deterministic validators decide what can execute. Arbitrary shell execution, hidden network
calls, and unapproved artifact writes are outside the accepted assistant safety model.
