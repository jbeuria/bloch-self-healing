# Self-Healing Coordination in Swarm Agents with Bloch-Type Perceptual Memory

Simulation and analysis code for the revised manuscript by **Jyotiranjan Beuria**, prepared for *Swarm Intelligence*.

Repository: <https://github.com/jbeuria/bloch-self-healing>

**This is a standalone, code-only release. No trial datasets, saved trajectories or result tables are included.** Upload the contents of this folder as the repository root. No parent-workspace files are needed. `run.py` loads the root modules as the `self_healing` package, so the clone directory may have any name.

The simulation, controller, motion interface, recovery definitions and statistical calculations are those used for the revised manuscript. The launcher and file-loading helpers support a standalone source release. Exact working coefficients are supplied as configuration files. `validation/upload_verification.json` records the checks performed during preparation and their scope.

## What this version reproduces

The study begins with a verified spatial split and heading disturbance after a prescribed 20-unit observation history. It compares seven controllers under common sensing, navigation and motion constraints:

| Identifier | Controller |
|---|---|
| `full` | Bloch perceptual channels with adaptive slow regulation and both feedback paths |
| `olfati_saber` | Olfati-Saber potential and velocity-consensus adaptation |
| `couzin` | Couzin zonal-interaction adaptation |
| `vasarhelyi` | Vásárhelyi velocity-control adaptation |
| `fast` | Bloch channels with both self-feedback paths removed |
| `frozen` | Full prepared internal state, with the slow state held fixed after damage |
| `scalar_matched` | Scalar memory matched to the continuous Bloch stationary response, with adaptive self feedback |

The external controllers are two-dimensional task adaptations, not reproductions of the original robot experiments. Within-framework variants inherit the full controller's working coefficients and are not independently retuned.

| Task | Initial heading-noise SD | During-motion noise amplitude | Obstacles |
|---|---:|---:|---|
| Moderate disorder | 0.8 | 0.1 | None |
| Severe disorder | 1.6 | 0.35 | None |
| Obstacle passage | 1.2 | 0.2 | One fixed disk, radius 4 |

All tasks begin fragmented. The obstacle tests passage after that imposed split; it is not needed to create the initial fragmentation.

- **Primary study:** 3 tasks × 7 controllers × 100 paired seeds (`73001` to `73100`) = **2,100 trials**.
- **Sensitivity study:** 7 settings × 3 tasks × 7 controllers × 30 fresh paired seeds (`74001` to `74030`) = **4,410 trials**.
- Reference setting: 48 agents, time step 0.05, horizon 45, centering coefficient 0.035. Sensitivity varies population (24, 96), time step (0.025, 0.0125), or centering (0.0175, 0), one factor at a time.
- The first three primary seeds have spatial recordings. Running the code saves connectivity and polarization at every step for every trial.

**The paper's coefficients are in `configurations/working.json`, not the generic defaults in `config.py`.** The supported commands below always load that file. No search or retuning occurs during the reported evaluation. The nominal `frozen_tone` configuration field is superseded by the exact full-controller checkpoint in the frozen-at-damage comparison.

## Environment

Use Python 3.11 and the pinned scientific packages:

```sh
conda env create -f environment.yml
conda activate self-healing
```

Alternatively, from an existing Python 3.11 environment:

```sh
python -m pip install -r requirements.txt
```

Run the remaining commands from this repository directory. No package installation or network access is needed after dependencies are installed. The default tests require no C compiler. An optional upstream-source test uses a C compiler and is described below. TeX is not required for any command in this README.

## Run and verify

First check the source files and run the numerical tests:

```sh
python run.py check
python run.py tests
python run.py smoke --workers 2
```

`check` checks the release manifest and working configuration. `tests` runs 51 numerical and protocol tests; three optional tests that compile an upstream C oracle are excluded by default. `smoke` runs the first primary seed, 73001, across all three tasks and seven controllers, giving 21 full-horizon trials. It checks the generated recovery endpoints, paired initial observations, state bounds, recorded safety classifications and configuration consistency. Results are written to `results/smoke/`.

For the complete reported experiments:

```sh
python run.py evaluate --workers 4
python run.py stats
python run.py thresholds
python run.py generalization --workers 4
```

- `evaluate`: 2,100 primary trials, seeds 73001 to 73100, written to `results/primary/`. It automatically checks the raw outputs.
- `stats`: primary means, sample SD, paired bootstrap intervals, permutation tests, paired t-tests and McNemar tests, written into `results/primary/`.
- `thresholds`: recomputes recovery for six threshold/holding-time choices, producing 12,600 trial-level rows and 126 aggregate rows in `results/primary/`.
- `generalization`: 4,410 sensitivity trials, seeds 74001 to 74030, written to `results/generalization/`, followed by auditing and statistical analysis.

Full evaluations take substantially longer than the smoke check. Simulation commands refuse to overwrite a nonempty output directory. Use a fresh `--out` directory for another run. To check an existing primary run again:

```sh
python run.py verify-run --out results/primary
```

For a smaller exploratory run with fresh seeds:

```sh
python run.py evaluate --seeds 10 --first-seed 75001 --workers 4 --out results/additional_seeds
python run.py stats --out results/additional_seeds
python run.py thresholds --out results/additional_seeds
```

Changing the seeds, sample size or configuration creates a new experiment. Use the defaults for comparison with the manuscript. The verifier checks internal consistency; it does **not** compare with a bundled reference dataset. Compare `aggregate.csv`, `contrasts.csv`, `success_contrasts.csv` and `threshold_sensitivity.csv` with the corresponding manuscript and supplement tables. The small reference table below provides an initial check. A one-seed smoke run cannot reproduce sample SDs or the published statistical comparisons.

The primary mean-comparison family contains 36 tests: six controls × three tasks × two endpoints. It uses 50,000 random sign permutations and 20,000 paired bootstrap resamples. Mean-difference and relative-improvement intervals use approximate simultaneous 95% coverage within their respective 36-comparison families. Healing success uses a separate 18-comparison McNemar family. Sensitivity analysis uses separate families of 252 controller comparisons, 252 setting comparisons and 126 success comparisons; its bootstrap intervals are ordinary 95% intervals.

Raw outputs contain all time-step diagnostic series and spatial snapshots for the first three primary seeds. Safety is checked during simulation on every straight motion segment. The post-run verifier checks the saved minimum-clearance classifications; it does not independently reconstruct every swept path from the less frequent spatial snapshots. Provenance files record the software versions, source hashes and protocol. Tiny arithmetic differences between platforms can propagate through trajectories, so investigate discrepancies before interpreting a changed ranking.

Optional tests against the pinned authors' C interaction functions can be run with `python run.py tests --with-c-oracle`. That command requires a C compiler and compiles a small test oracle. The default workflow needs no compiler.

## Plotting is separate from simulation

After the complete workflow above:

```sh
python run.py figures
```

This reads generated primary and sensitivity results and writes **PDF figures** plus editable `.tex` table fragments to `results/assets/`. It does not simulate, tune parameters or compile LaTeX. No PNG or SVG figures are produced. The architecture schematic is also a PDF. For results generated at other locations:

```sh
python run.py figures --primary results/my_primary --sensitivity results/my_sensitivity --out results/my_assets
```

Run `stats` and `thresholds` on the chosen primary folder before plotting. The compact tables contain the manuscript's fixed protocol labels, so use the default study sizes and settings when preparing publication assets.

## Recovery definitions and expected results

`F` is the fraction of agents in the largest connected component; `P` is the magnitude of the mean unit heading. Recovery requires `F >= 0.90` and `P >= 0.90` at every sampled time for two simulation time units. Recovery time is the start of the first confirmed interval. A trial without confirmed recovery contributes the horizon, 45, to the mean.

The integrated coordination deficit is `J = integral(1 - F*P) dt`, calculated by the trapezoidal rule. Healing success additionally requires recovery of all split events, a qualifying final interval, safe motion, normalized forward progress at least 0.5, and all-agent obstacle passage when applicable.

Full-controller primary results, mean ± sample SD over 100 seeds:

| Task | Recovery time | Coordination deficit J | Healed |
|---|---:|---:|---:|
| Moderate | 0.988 ± 0.233 | 0.643 ± 0.132 | 100/100 |
| Severe | 1.844 ± 0.244 | 1.285 ± 0.156 | 100/100 |
| Obstacle | 1.483 ± 0.251 | 10.494 ± 0.596 | 100/100 |

The full controller improves both primary endpoints over the three external adaptations under moderate disorder. Rankings under severe disorder and obstacle passage depend on the endpoint. The internal comparisons support effective recovery across related perceptual architectures and show a task-dependent feedback contribution. They do not establish a large, unique benefit of ongoing adaptive self feedback.

## What each module does

The supported entry point is `run.py`. Several modules retain historical development commands; these may require development records that are not distributed. Use the launcher to reproduce the current fixed-parameter paper. These older entry points are not part of the supported standalone workflow.

| File | Role |
|---|---|
| `run.py` | Current-study launcher, integrity checks, result verification and reproducibility commands. |
| `bootstrap.py` | Makes the flat upload folder importable as `self_healing`, regardless of clone name. |
| `paths.py` | Locates the standalone source/configuration root. |
| `__init__.py` | Package marker; no simulation is run on import. |
| `config.py` | Immutable configuration, validation and JSON loading. Generic defaults are not the reported working configuration. |
| `controller.py` | Bloch split update, stationary response, internal-state variants, social steering and slow-state feedback. Its standalone `Controller.step` is the older motion interface; the reported trials use `MatchedController`. |
| `external_controllers.py` | The three external adaptations, shared bounded-velocity actuator and directional safety guard. `MatchedController` also wraps all reported internal variants. |
| `simulation.py` | Time-stepping engine, measurements, swept-clearance checks and result assembly. |
| `scenarios.py` | Lattice and obstacle construction. The reported damaged states are prepared by `conditioning_state` in `healing_search.py`. |
| `metrics.py` | Connectivity, polarization, exact straight-segment clearances, split/recovery events and healing classification. |
| `safety.py` | Older direction-independent braking guard, retained for historical modes/tests. The reported guard is in `external_controllers.py`. |
| `storage.py` | JSON/CSV/NPZ storage, source snapshots, file hashing and run provenance. |
| `run_primary.py` | Loads the fixed working JSON and schedules current primary trials. Called by `run.py`. |
| `healing_refinement.py` | Task definitions and single-trial execution used by the current primary study; also retains earlier candidate-search commands. |
| `healing_search.py` | Shared prescribed sensory-history preparation and damage initialization; also retains an earlier exploratory search. |
| `compare.py` | Shared recovery endpoints and equilibrium helper; its original CLI belongs to earlier comparison experiments. |
| `comparison_analysis.py` | Shared paired bootstrap, sign-permutation, Holm and binomial-interval utilities; the original CLI is historical. |
| `healing_refinement_analysis.py` | Primary aggregate statistics, paired mean comparisons and McNemar success comparisons. Some historical report labels are normalized by the launcher. |
| `generalization.py` | Fixed-parameter sensitivity simulations and nested Gaussian noise tapes. The launcher supplies the current coefficients and seeds explicitly. |
| `generalization_analysis.py` | Sensitivity auditing, compact-trace reconstruction, statistical comparisons and step-size differences. |
| `verify.py` | Reconstructs primary endpoints from newly generated traces and checks configuration, pairing and summary consistency. |
| `bundle.py` | Reads freshly generated primary runs for plotting and recalculates alternative recovery definitions. |
| `plotting/build_assets.py` | Primary figures and comparison tables from saved outcomes/traces. |
| `plotting/build_generalization.py` | Sensitivity PDF and complete sensitivity table. |
| `plotting/build_compact_tables.py` | Compact full-controller sensitivity and external-parameter table fragments. |
| `plotting/build_architecture.py` | Architecture schematic as a vector PDF. |
| `tests/` | Numerical, protocol, statistical and literature-interaction regression tests, with synthetic geometries, pinned upstream C source and its notices; no saved trajectory fixtures. |

## Configuration and generated files

- `configurations/working.json` and `working.sha256`: exact working coefficients and their integrity hash.
- `configurations/development_spaces.json`: earlier parameter-search intervals, used only to format the supplement's parameter table. This is configuration metadata, not trial data; no search is needed to run this release.
- `validation/upload_verification.json`: concise preparation checks and limitations, without trial outcomes or trajectories.
- `MANIFEST.sha256`: integrity hashes for the delivered source, documentation and configuration files.
- `results/`: generated locally when commands run; ignored by Git and absent from the upload.

The working configuration includes four controller specifications. The three within-framework controls inherit the full specification. The `assignment` fields record configuration provenance and are not instructions to optimize. Task-specific noise is assigned in `healing_refinement.py`; the sensitivity settings are in `generalization.py`.

## Citation

Please cite the self-healing study below when using this implementation or its recovery benchmark, and the foundational papers when using the underlying perceptual architecture.

1. **This study:** Jyotiranjan Beuria (2026). *Self-Healing Coordination in Cognitive Swarm Agents with Bloch-Type Perceptual Memory*. [arXiv:2607.11960](https://arxiv.org/abs/2607.11960). [doi:10.48550/arXiv.2607.11960](https://doi.org/10.48550/arXiv.2607.11960).

2. **Perceptual-dynamics foundation:** Jyotiranjan Beuria, Mayank Chaurasiya and Laxmidhar Behera (2025). *Collective motion from quantum-inspired dynamics in visual perception*. Proceedings of the Royal Society A **481**(2321), 20250489. [doi:10.1098/rspa.2025.0489](https://doi.org/10.1098/rspa.2025.0489).

3. **Self-regulated collective dynamics:** Jyotiranjan Beuria (2025; version 2 revised 2026). *Non-Markovian Collective Motion from Self-Regulated Perceptual Dynamics*. [arXiv:2510.23688v2](https://arxiv.org/abs/2510.23688v2). [doi:10.48550/arXiv.2510.23688](https://doi.org/10.48550/arXiv.2510.23688).

`CITATION.cff` makes the self-healing preprint the preferred GitHub citation. `CITATIONS.bib` contains entries for these three works.

## Upload and licensing

Upload the contents of this folder as the repository root, including dotfiles and the manifest. Do not upload locally generated `results/`, datasets or trajectory archives. The supplied `.gitignore` excludes them.

No project-wide software license has been assigned in this preparation. Existing third-party license notices in `tests/fixtures/robotsim/` are retained.
