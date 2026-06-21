# Assistant MCP Contract

The MCP server is a later interface over the same assistant registry used by `astro ask`.

## Tools

- `astro_plan_workflow`: accepts natural-language intent and returns `AstroWorkflowPlan`.
- `astro_dry_run_workflow`: accepts an `AstroWorkflowPlan` and returns command specs plus warnings.
- `astro_execute_workflow`: accepts an approved plan and returns `WorkflowTrace`.

## Resources

- `astro://examples/assistant/od_workflow_prompt`
- `astro://schemas/assistant/workflow-plan`
- `astro://schemas/assistant/workflow-trace`

## Policy

The MCP layer must not expose arbitrary shell execution. It must call the same registry, policy, and
executor modules used by the CLI. Write-producing tools must require explicit approval from the MCP
client before execution.
