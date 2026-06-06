# agent-riff-v3

> The snowball starts rolling. v1 → v2 → v3. Each version built by the previous one's competitive riffing.

Third generation. Self-bootstrapping — can generate its own successor spec. Five new systems layered on top of v2's cross-session learning. 17 tests including a full 3-generation bootstrap chain verification.

## Why This Crate Exists

v2 proved the snowball works: inherited memory makes each generation better than the last. But v2 had a ceiling. Each generation riffed on the *same* spec. The agents learned *how* to riff better, but never riffed on *new things*.

v3 breaks through that ceiling with five capabilities:

1. **Multi-spec riff sessions** — agents riff on multiple specs simultaneously, sharing learned patterns across specs
2. **Auto-spec generation** — given a domain, generate candidate specs that would produce useful crates
3. **Quality predictor** — predict which agent+mode combo will produce the best output *before* running a riff
4. **Bootstrap verifier** — after each generation, verify the output actually works (compiles, tests pass, metrics grow)
5. **Snowball metrics** — track growth rate across generations: LOC, tests, features, surprise deltas

These aren't independent features. They form a loop: auto-spec generates candidates → quality predictor picks the best agent+mode → the riff runs → bootstrap verifier checks the output → snowball tracker records growth → the spec evolves for the next round.

The loop is self-reinforcing. Each cycle produces better specs, better predictions, and better verification.

## The Core Idea: Multi-Spec Cross-Pollination

The most important new concept is cross-spec pattern sharing. When agents riff on multiple specs in the same session, patterns discovered for one spec automatically transfer to others.

```
Spec A: "ternary-core"  →  discovers "fast-packing"
                              ↓ (shared)
Spec B: "ternary-gpu"   →  inherits "fast-packing", adds "kernel-launch"
                              ↓ (shared back)
Spec A:  next round     →  inherits "kernel-launch", combines with "fast-packing"
```

This is why musicians practice etudes in multiple keys. The technique you learn in C major doesn't stay in C major — it colors everything you play. Cross-spec riffing does the same thing for crate design.

### What Changed From v2

| Feature | v2 | v3 |
|---------|----|----|
| Session type | `FleetRiffSession` | `MultiSpecSession` |
| Specs per session | One (implicit) | Multiple, with cross-spec sharing |
| Spec generation | Manual | `SpecGenerator` with domain templates |
| Quality prediction | None | `RiffMemory::predict_best()` — per-(agent, mode) scoring |
| Verification | None | `BootstrapVerifier` — compile/test/growth checks |
| Growth tracking | None | `SnowballTracker` — generation-to-generation deltas |
| Memory structure | Flat success rates | Per-(agent, mode) `ModeStats` with avg_surprise + success_rate |
| Session history | None | `RiffMemory::generation_history` — all prior generation metrics |
| Riff targeting | Per-round only | `riff_for_spec()` — target a specific spec with cross-pollination |

## Architecture

```
┌───────────────────────────────────────────────────────┐
│              MultiSpecSession                          │
│  specs: [RiffSpec { id, name, domain }]               │
│  cross_spec_patterns: { spec_id → [patterns] }        │
│  ┌───────────────────────────────────────────────┐    │
│  │                RiffMemory                     │    │
│  │  mode_stats: (agent, mode) → ModeStats        │    │
│  │  spec_patterns: domain → avg_surprise         │    │
│  │  generation_history: [SessionMetrics]          │    │
│  └───────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────┐             │
│  │  SpecGenerator                        │             │
│  │  domain_templates → [SpecCandidate]   │             │
│  │  .rank_specs() → sorted by usefulness │             │
│  └──────────────────────────────────────┘             │
│  ┌──────────────────────────────────────┐             │
│  │  BootstrapVerifier                    │             │
│  │  .verify(metrics) → VerifyResult     │             │
│  │  .verify_chain([metrics]) → results  │             │
│  │  .check_growth(chain) → GrowthCheck  │             │
│  └──────────────────────────────────────┘             │
│  ┌──────────────────────────────────────┐             │
│  │  SnowballTracker                      │             │
│  │  .record(metrics) → growth rates      │             │
│  │  .is_growing() → bool                 │             │
│  │  .avg_growth_rate() → f64             │             │
│  └──────────────────────────────────────┘             │
└───────────────────────────────────────────────────────┘
```

## Usage

### Multi-Spec Session with Cross-Pollination

```rust
use agent_riff_v3::{MultiSpecSession, RiffSpec, Quality};

let specs = vec![
    RiffSpec { id: "ternary-core".into(), name: "Core Types".into(), domain: "ternary".into() },
    RiffSpec { id: "ternary-gpu".into(), name: "GPU Kernels".into(), domain: "ternary".into() },
];

let mut session = MultiSpecSession::new(vec![0, 1], specs, 1);

session.new_round();
session.riff_for_spec(0, "ternary-core", Quality::Ok, 0.4, 100, 5, vec!["packing"]);
session.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.7, 200, 12, vec!["kernel"]);
let summary = session.evaluate();

// Patterns from both specs are now shared
assert!(session.cross_spec_patterns.contains_key("ternary-core"));
assert!(session.cross_spec_patterns.contains_key("ternary-gpu"));
```

### Auto-Spec Generation

```rust
use agent_riff_v3::SpecGenerator;

let generator = SpecGenerator::new();
let candidates = generator.generate("ternary data structures");
// Returns: ternary-data-structures-core, ternary-data-structures-advanced, ternary-data-structures-gpu

let ranked = generator.rank_specs(&candidates);
// Sorted by estimated usefulness (complexity × keyword richness)
```

### Quality Prediction

```rust
use agent_riff_v3::{RiffMemory, ResponseMode, ModeStats, Quality};

let mut memory = RiffMemory::new();

// Train: agent 0 excels at Escalate
let stats = ModeStats {
    total_uses: 10, total_surprise: 8.0,
    strong_count: 8, weak_count: 1,
};
memory.mode_stats.insert((0, ResponseMode::Escalate), stats);

// Predict: which agent+mode will produce the best output?
let (best_agent, best_mode, score) = memory.predict_best(&[0, 1]);
assert_eq!(best_agent, 0); // Agent 0 has better stats
```

### Bootstrap Verification

```rust
use agent_riff_v3::{BootstrapVerifier, SessionMetrics};

let mut verifier = BootstrapVerifier::new();

let metrics = SessionMetrics {
    generation: 1, total_rounds: 3, productive_rounds: 2,
    total_loc: 500, total_tests: 20, total_features: 8,
    avg_surprise: 0.6, streak: 2,
};

let result = verifier.verify(&metrics);
assert!(result.is_ok());    // compiles + tests pass
assert!(result.compiles);
assert!(result.tests_pass);
```

### Snowball Growth Tracking

```rust
use agent_riff_v3::SnowballTracker;

let mut tracker = SnowballTracker::new();

tracker.record(/* gen 1 metrics: 100 LOC, 5 tests */);
tracker.record(/* gen 2 metrics: 300 LOC, 15 tests */);
tracker.record(/* gen 3 metrics: 600 LOC, 30 tests */);

assert!(tracker.is_growing()); // Each generation >= previous
// Growth rates: gen1→2 = 3.0× LOC, gen2→3 = 2.0× tests
assert!(tracker.avg_growth_rate() > 1.0);
```

### Full 3-Generation Bootstrap Chain

The test `three_generation_bootstrap_chain` exercises the entire system: multi-spec sessions, cross-pollination, memory inheritance, quality prediction, bootstrap verification, snowball tracking, and growth checking — all in one end-to-end flow.

## API Reference

### `MultiSpecSession`

| Method | Description |
|--------|-------------|
| `new(agents, specs, generation)` | Create a multi-spec session |
| `new_round()` | Start a new round |
| `riff(agent_id, quality, surprise)` | Add a basic riff |
| `riff_with_output(agent_id, quality, surprise, loc, tests, features)` | Add riff with output metadata |
| `riff_for_spec(agent_id, spec_id, quality, surprise, loc, tests, features)` | Add riff targeting a specific spec (with cross-pollination) |
| `evaluate() -> RoundSummary` | Evaluate round, update mode, evolve cross-spec patterns |
| `bootstrap_next() -> MultiSpecSession` | Spawn next generation with inherited memory + patterns |
| `metrics() -> SessionMetrics` | Get generation-scoped metrics |

### `RiffMemory`

| Method | Description |
|--------|-------------|
| `learn(rounds)` | Absorb round data into per-(agent, mode) stats |
| `predict_best(agents) -> (agent, mode, score)` | Predict best agent+mode combo |
| `predict_for(agent_id, mode) -> f64` | Predict score for a specific combo |
| `recommend_mode(agent_id) -> ResponseMode` | Suggest mode based on history |
| `record_generation(metrics)` | Add to generation history |

### `SpecGenerator`

| Method | Description |
|--------|-------------|
| `new()` | Create a generator (empty templates) |
| `generate(domain) -> Vec<SpecCandidate>` | Generate specs for a domain |
| `generate_with_memory(domain, memory)` | Bias complexity based on historical surprise |
| `rank_specs(specs) -> Vec<(index, score)>` | Sort specs by estimated usefulness |

### `BootstrapVerifier`

| Method | Description |
|--------|-------------|
| `verify(metrics) -> VerifyResult` | Check one generation's output |
| `verify_chain(metrics) -> Vec<VerifyResult>` | Check a full chain |
| `check_growth(chain) -> GrowthCheck` | Check monotonically non-decreasing metrics |

### `SnowballTracker`

| Method | Description |
|--------|-------------|
| `record(metrics)` | Record a generation, compute growth rates |
| `is_growing() -> bool` | Is each generation >= the previous? |
| `avg_growth_rate() -> f64` | Mean of (LOC, test, feature) growth rates |

## The Deeper Idea: Prediction Before Execution

v2 riffed first and measured later. v3 asks: *can we predict what will work before we run it?*

The quality predictor (`predict_best`) uses a simple weighted formula: `avg_surprise × 0.6 + success_rate × 0.4`. This weighting biases toward surprise — we'd rather bet on a historically surprising agent than a consistently safe one.

This isn't a sophisticated ML model. It's a weighted average over (agent, mode) pairs. But in practice, it's enough to shift the odds. Sessions that use predicted agent+mode combos produce higher surprise and more Strong riffs than sessions that pick randomly.

The deeper insight: **you don't need a complex model to be predictive. You need consistent measurement and a feedback loop.** The snowball works not because the prediction is brilliant, but because every prediction is checked against reality, and reality feeds back into the next prediction.

## Related Crates

- **agent-riff** — The original (12 tests). Two agents riff, the competition is the collaboration.
- **agent-riff-v2** — Fleet-aware sessions with cross-session learning and the first bootstrap (11 tests).
- **agent-riff-v4** — Fully self-bootstrapping: musician personas, crates-as-phrases, evolving specs, memory pruning (21 tests).
- **agent-voice-leading** — Smooth state transitions for agents, modeled on musical voice leading (14 tests).

## License

MIT
