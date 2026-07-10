# Reentry Modeling And Simulation

Astro Suite provides a suite-owned reentry workflow for ballistic capsules, lifting vehicles with
prescribed bank commands, and target-tracking guided lifting entry. The workflow produces a typed
`ReentryResult` with trajectory, loads, aerothermal history, peak events, target miss, and design
margins.

## Run The Reference Cases

Ballistic capsule:

```bash
astro simulate-reentry examples/reentry/ballistic_capsule.yaml \
  --output /tmp/astro-ballistic-reentry.json \
  --summary-output /tmp/astro-ballistic-reentry.txt
```

Prescribed-bank lifting entry:

```bash
astro simulate-reentry examples/reentry/lifting_bank_schedule.yaml \
  --output /tmp/astro-lifting-reentry.json \
  --summary-output /tmp/astro-lifting-reentry.txt
```

Target-tracking guided entry:

```bash
astro simulate-reentry examples/reentry/guided_lifting_body.yaml \
  --output /tmp/astro-guided-reentry.json \
  --summary-output /tmp/astro-guided-reentry.txt
```

Tune the velocity-indexed bank magnitudes for the declared landing target:

```bash
astro optimize-reentry examples/reentry/guided_lifting_body.yaml \
  --output /tmp/astro-reentry-optimization.json \
  --tuned-scenario-output /tmp/astro-reentry-tuned.yaml \
  --summary-output /tmp/astro-reentry-optimization.txt
```

## Model

The local simulator integrates a spherical-Earth, three-degree-of-freedom point-mass entry state:

- radius, geocentric latitude, and longitude;
- atmosphere-relative speed;
- flight-path angle and heading;
- accumulated downrange and convective heat load.

Gravity varies with radius. Aerodynamic drag is computed from density, speed, reference area, and
drag coefficient. Lift is the configured constant lift-to-drag ratio times drag. Bank angle splits
lift into vertical and lateral components. Fixed-step RK4 integration uses a separately configured
internal step so public output cadence does not set numerical resolution.

The current atmosphere options are:

- `exponential`: configurable reference density, reference altitude, scale height, and density
  scale factor;
- `none`: vacuum control case.

The `sutton_graves` aerothermal option computes stagnation-point convective heat-rate screening from
density, nose radius, and velocity. Heat rate is integrated into heat load, and a radiative
equilibrium wall temperature is reported from configured emissivity. `none` disables heating.

## Guidance Modes

- `ballistic`: zero lift-to-drag ratio and zero bank command.
- `constant_bank`: one signed bank command for a lifting vehicle.
- `bank_schedule`: signed bank commands interpolated against decreasing velocity knots.
- `target_tracking`: non-negative bank magnitudes interpolated against velocity; bank sign follows
  great-circle heading error to the target, with heading deadband, minimum reversal interval, and
  minimum control speed to prevent command chatter.

`astro optimize-reentry` supports `target_tracking` scenarios. It uses deterministic bounded SciPy
optimization to tune bank-schedule magnitudes against final great-circle target miss plus normalized
penalties for exceeded dynamic-pressure, deceleration, heat-rate, and heat-load limits. The
optimization result contains both the tuned scenario and its full `ReentryResult`. An optimized
candidate is accepted only when its objective does not regress from the input schedule; otherwise
the product retains the initial scenario and records `accepted_solution = initial_no_regression`.

## Result Product

Each output sample records:

- epoch, elapsed time, altitude, latitude, longitude, downrange, speed, flight-path angle, heading,
  and commanded bank;
- atmospheric density, dynamic pressure, drag and lift acceleration, and aerodynamic deceleration
  in g;
- convective heat rate, integrated heat load, radiative equilibrium temperature, and optional range
  to target.

The result also includes entry-interface, bank-reversal, peak-heating, peak-dynamic-pressure,
peak-deceleration, and terminal events; peak summaries; final target miss; requirement margins with
pass/warn/fail status; backend and model provenance; and warning text describing model authority.

## Orbit Handoff

An existing suite `Trajectory` can provide a reentry initial state through a reentry scenario
template:

```bash
astro handoff-reentry /tmp/astro-orbit-trajectory.json \
  examples/reentry/ballistic_capsule.yaml \
  --sample-index -1 \
  --output /tmp/astro-reentry-handoff.yaml
```

The handoff rotates the selected EME2000 state into an Earth-fixed frame using Greenwich sidereal
angle, subtracts constant Earth rotation from velocity, resolves local radial/north/east velocity,
and derives altitude, geocentric latitude/longitude, atmosphere-relative speed, flight-path angle,
and heading. The selected trajectory sample must already represent the intended entry-interface or
deorbit state; the handoff does not synthesize a deorbit burn.

## Authority And Limits

This workflow is deterministic engineering-screening evidence for entry corridors, load envelopes,
thermal-protection sizing signals, guidance trades, and reproducible software workflows. It does not
provide:

- six-degree-of-freedom attitude, control-surface, actuator, or vehicle-stability dynamics;
- winds, weather, rotating-Earth inertial terms in the entry equations, atmospheric composition, or
  density uncertainty ensembles;
- rarefied-flow, real-gas chemistry, radiation, shock interaction, CFD, material response,
  ablation, or certified TPS analysis;
- terrain, parachute, powered descent, impact dispersion, or operational landing prediction;
- flight-qualified guidance, navigation, and control.

Higher-authority work should preserve `ReentryScenario` and `ReentryResult` as the public product
boundary while mapping external Dymos, TudatPy, Orekit, CFD, atmosphere, or material-response
evidence through explicit adapters and campaign-specific provenance.
