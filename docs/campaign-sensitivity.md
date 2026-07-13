# Campaign Sensitivity And Margin Attribution

Astro derives rank-based sensitivity evidence from a completed uncertainty campaign without
rerunning physics or modifying the source artifacts. The public command is:

```bash
astro analyze-campaign-sensitivity /tmp/astro-lifecycle-sensitivity/campaign \
  --metric deorbit_propellant_used \
  --metric propellant_reserve_margin \
  --metric entry_interface_margin \
  --metric reentry_peak_heat_rate \
  --metric reentry_peak_dynamic_pressure \
  --requirement-margin propellant_reserve \
  --requirement-margin twin_worst_observed_link_margin \
  --requirement-margin twin_propellant_fraction \
  --output /tmp/astro-lifecycle-sensitivity/sensitivity.json
```

The output must be outside the source campaign directory. It binds the campaign definition,
sample, and case digests so a report cannot be detached from its source evidence.

## Methods

Each target reports two complementary coefficients for every declared continuous parameter:

- **Spearman rho** is the marginal monotonic association between one input and one target.
- **Partial rank correlation (PRCC)** removes the linear rank effects of the other inputs before
  correlating the residuals. It is a conditional monotonic association, not a causal effect.

Requirement-margin targets preserve Astro's signed margin convention and record
`orientation=higher_is_safer`. A positive coefficient means larger input values tend to increase
the configured safety margin; a negative coefficient means they tend to reduce it. Boolean
requirement sentinels are rejected because `+1/-1` is not a physical distance to a boundary.

This product does not emit p-values or confidence intervals. LHS and low-discrepancy campaign rows
are not treated as ordinary IID observations, and the v1 product is deterministic descriptive
screening rather than an inferential test.

## Validity Gates

Sensitivity analysis fails closed unless:

- the campaign is complete and every analyzed case succeeded;
- every case joins exactly one integrity-checked planned sample;
- weights are equal and all selected inputs and targets are finite and numeric;
- `n >= max(30, 5 * (parameter_count + 1))` and residual degrees of freedom are at least 20;
- every input and target has at least five unique values;
- the standardized ranked design has full column rank and condition number no greater than `1e8`.

Average ranks handle ties. Tie fractions above `0.20` and condition numbers above `1e4` are recorded
as warnings. Discrete model variants, weighted rank methods, failed-case attribution, binary failure
models, and interaction decomposition are outside v1.

Existing `kind: sobol` campaigns use a Sobol low-discrepancy sequence. They do not contain the
independent A/B and hybrid matrix roles required for a Saltelli pick-and-freeze estimator, so this
command never labels its coefficients as Sobol indices or variance fractions.

## Checked Lifecycle Evidence

`examples/campaigns/leo_lifecycle_sensitivity.yaml` uses 64 Latin-hypercube cases over the seven
reviewed cross-phase epistemic inputs. The checked local run completed 64/64 cases with effective
sample size 64, 56 PRCC residual degrees of freedom, full design rank, condition number 1.6532, and
no input or target ties. Generated bulk artifacts remain under `/tmp` and are not committed.

| Target | Largest observed absolute PRCC values | Interpretation |
| --- | --- | --- |
| Deorbit propellant used | delta-v `+0.9706`; specific impulse `-0.9645` | Larger burns increase use; higher specific impulse reduces it. |
| Propellant reserve margin | delta-v `-0.9638`; specific impulse `+0.9569`; wet mass `+0.9411` | Burn design and available wet-mass allocation dominate reserve margin in this configured domain. |
| Reentry peak heat rate | drag coefficient `-0.9844`; delta-v `+0.8422` | Vehicle drag and the upstream deorbit design dominate this local screening response. |
| Reentry peak dynamic pressure | drag coefficient `-0.9895`; delta-v `+0.8415` | The same bounded design inputs dominate peak dynamic-pressure association. |
| Entry-interface margin | largest observed absolute PRCC `0.2015` | The observed associations are smaller than the dominant coefficients above; no practical-effect threshold is asserted. |
| Twin worst observed contact link margin | launch thrust `-0.9965`; wet mass `+0.8913` | The conditional association reflects the coupled launch-state and observed link-geometry response in this configured lifecycle model. The full 64-case target range is only `0.0035 dB`; no practical-effect threshold was evaluated. |
| Twin propellant-fraction margin | wet mass `+1.0000` | The wet-mass override changes available propellant while dry and payload mass stay fixed. This is not conventional mass-growth allowance. |

All ten checked subsystem requirements passed in all 64 cases. Battery-SOC, minimum thermal,
pointing, torque, slew-rate, actuator-utilization, and itemized mass-budget margins were constant
under the existing seven inputs, as were contact count and duration. They are therefore valid
campaign requirement evidence but invalid PRCC targets. Astro fails closed on those constant
targets; it does not manufacture a ranking. Relevant typed thermal, ADCS, power-load, or link
inputs must be introduced through a separate decision-specific scope before those margins can be
attributed.

These coefficients apply only to the declared ranges, models, sampler, and local deterministic
lifecycle workflow. They are not causal contributions, portable variance fractions, operational
probabilities, mission assurance, or certification evidence.
