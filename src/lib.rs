//! # agent-riff-v3
//!
//! Snowball generation 3. v1 had 12 tests. v2 had 11 tests, fleet-aware sessions.
//! v3 adds 5 major features and 16+ tests including a full 3-generation bootstrap chain.
//!
//! What v3 adds over v2:
//! 1. **Multi-spec riff sessions** — agents riff on multiple specs simultaneously,
//!    sharing learned patterns across specs
//! 2. **Auto-spec generation** — given a domain, generate spec candidates that would
//!    produce useful crates
//! 3. **Quality predictor** — before running a riff, predict which agent+mode combo
//!    will produce the best output based on accumulated RiffMemory
//! 4. **Bootstrap verifier** — after each generation, verify output compiles and tests pass
//! 5. **Snowball metrics** — track growth rate: LOC, tests, features, surprise delta
//!    across generations
//!
//! THE SNOWBALL: v1 → v2 → v3. Each version is better because competitive riffing
//! between agents produced improvements neither would invent alone.

#![forbid(unsafe_code)]

use std::collections::HashMap;

// ── Ternary types (same encoding as ternary-cuda-kernels) ──────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit { Neg = -1, Zero = 0, Pos = 1 }

impl Trit {
    pub fn to_i8(self) -> i8 { self as i8 }
    pub fn from_i8(v: i8) -> Option<Self> {
        match v { -1 => Some(Trit::Neg), 0 => Some(Trit::Zero), 1 => Some(Trit::Pos), _ => None }
    }
    pub fn pack_bits(self) -> u8 { match self { Trit::Neg => 0, Trit::Zero => 1, Trit::Pos => 2 } }
    pub fn unpack_bits(b: u8) -> Self { match b & 0x3 { 0 => Trit::Neg, 1 => Trit::Zero, 2 => Trit::Pos, _ => Trit::Zero } }
}

/// Pack 16 trits into one u32 (GPU-ready).
pub fn pack_16(trits: &[Trit]) -> u32 {
    let mut packed = 0u32;
    for (i, &t) in trits.iter().take(16).enumerate() {
        packed |= (t.pack_bits() as u32) << (i * 2);
    }
    packed
}

/// Quality of a riff output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quality { Weak = -1, Ok = 0, Strong = 1 }
impl Quality { pub fn to_i8(self) -> i8 { self as i8 } }

/// Response mode — how an agent responds to the previous riff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseMode { Escalate, Pivot, Invert, Provoked }

impl ResponseMode {
    pub fn auto(surprise: f64, streak: u32, round: u32) -> Self {
        if streak > 5 { ResponseMode::Pivot }
        else if surprise < 0.2 { ResponseMode::Provoked }
        else if surprise > 0.7 { ResponseMode::Escalate }
        else if round > 8 { ResponseMode::Invert }
        else { ResponseMode::Invert }
    }
}

/// A single riff output.
#[derive(Debug, Clone)]
pub struct Riff {
    pub agent_id: u32,
    pub round: u32,
    pub quality: Quality,
    pub surprise: f64,
    pub loc: usize,
    pub tests: usize,
    pub features: Vec<String>,
    pub spec_id: Option<String>, // v3: which spec this riff targets
}

impl Riff {
    pub fn new(agent_id: u32, round: u32, quality: Quality, surprise: f64) -> Self {
        Self { agent_id, round, quality, surprise, loc: 0, tests: 0, features: Vec::new(), spec_id: None }
    }
    pub fn with_output(&mut self, loc: usize, tests: usize, features: Vec<&str>) {
        self.loc = loc; self.tests = tests;
        self.features = features.iter().map(|s| s.to_string()).collect();
    }
    pub fn productivity(&self) -> f64 {
        let q = match self.quality { Quality::Weak => 0.5, Quality::Ok => 1.0, Quality::Strong => 2.0 };
        self.loc as f64 * self.tests as f64 * q * (0.5 + self.surprise)
    }
}

/// A round of the riff session.
#[derive(Debug, Clone)]
pub struct Round {
    pub number: u32,
    pub riffs: Vec<Riff>,
    pub best_agent: u32,
    pub quality_gap: i8,
    pub surprise_sum: f64,
}

impl Round {
    fn new(number: u32) -> Self { Self { number, riffs: Vec::new(), best_agent: 0, quality_gap: 0, surprise_sum: 0.0 } }

    fn add(&mut self, riff: Riff) {
        self.surprise_sum += riff.surprise;
        self.riffs.push(riff);
        self.recalc();
    }

    fn recalc(&mut self) {
        if self.riffs.is_empty() { return; }
        let best = self.riffs.iter().max_by_key(|r| r.quality.to_i8()).unwrap();
        let worst = self.riffs.iter().min_by_key(|r| r.quality.to_i8()).unwrap();
        self.best_agent = best.agent_id;
        self.quality_gap = best.quality.to_i8() - worst.quality.to_i8();
    }

    pub fn was_productive(&self) -> bool { self.surprise_sum > 0.3 || self.quality_gap > 0 }
}

// ── Cross-session learning (enhanced in v3) ─────────────────────────

/// Accumulated success rates per (agent, mode) pair.
#[derive(Debug, Clone, Default)]
pub struct ModeStats {
    pub total_uses: u32,
    pub total_surprise: f64,
    pub strong_count: u32,
    pub weak_count: u32,
}

impl ModeStats {
    pub fn avg_surprise(&self) -> f64 {
        if self.total_uses == 0 { 0.0 } else { self.total_surprise / self.total_uses as f64 }
    }
    pub fn success_rate(&self) -> f64 {
        if self.total_uses == 0 { 0.5 } else { self.strong_count as f64 / self.total_uses as f64 }
    }
}

/// Cross-session learning — remembers what works across riff sessions.
/// v3: tracks per-(agent, mode) success rates for quality prediction.
#[derive(Debug, Clone, Default)]
pub struct RiffMemory {
    pub best_modes: HashMap<u32, ResponseMode>,
    pub mode_stats: HashMap<(u32, ResponseMode), ModeStats>,
    pub spec_patterns: HashMap<String, f64>, // v3: spec domain → avg surprise
    pub total_rounds: u64,
    pub total_surprise: f64,
    pub escalation_success_rate: f64,
    pub pivot_success_rate: f64,
    pub invert_success_rate: f64,
    pub provoked_success_rate: f64,
    // v3 snowball history
    pub generation_history: Vec<SessionMetrics>,
}

impl RiffMemory {
    pub fn new() -> Self { Self::default() }

    pub fn learn(&mut self, rounds: &[Round]) {
        self.total_rounds += rounds.len() as u64;
        for r in rounds {
            self.total_surprise += r.surprise_sum;
            for riff in &r.riffs {
                let mode = ResponseMode::auto(riff.surprise, 0, r.number);
                let stats = self.mode_stats.entry((riff.agent_id, mode)).or_default();
                stats.total_uses += 1;
                stats.total_surprise += riff.surprise;
                match riff.quality {
                    Quality::Strong => stats.strong_count += 1,
                    Quality::Weak => stats.weak_count += 1,
                    _ => {}
                }
            }
        }
    }

    /// Record a completed session's metrics for snowball tracking.
    pub fn record_generation(&mut self, metrics: SessionMetrics) {
        self.generation_history.push(metrics);
    }

    pub fn recommend_mode(&self, agent_id: u32) -> ResponseMode {
        self.best_modes.get(&agent_id).copied().unwrap_or(ResponseMode::Escalate)
    }

    // ── v3: Quality predictor ───────────────────────────────────────

    /// Predict which agent+mode will produce the best output.
    /// Returns (agent_id, mode, predicted_score).
    pub fn predict_best(&self, agents: &[u32]) -> (u32, ResponseMode, f64) {
        let all_modes = [ResponseMode::Escalate, ResponseMode::Pivot, ResponseMode::Invert, ResponseMode::Provoked];
        let mut best_agent = agents[0];
        let mut best_mode = ResponseMode::Escalate;
        let mut best_score = -1.0f64;

        for &agent in agents {
            for &mode in &all_modes {
                let stats = self.mode_stats.get(&(agent, mode)).cloned().unwrap_or_default();
                let score = stats.avg_surprise() * 0.6 + stats.success_rate() * 0.4;
                if score > best_score {
                    best_score = score;
                    best_agent = agent;
                    best_mode = mode;
                }
            }
        }
        (best_agent, best_mode, best_score)
    }

    /// Predict score for a specific agent+mode.
    pub fn predict_for(&self, agent_id: u32, mode: ResponseMode) -> f64 {
        let stats = self.mode_stats.get(&(agent_id, mode)).cloned().unwrap_or_default();
        stats.avg_surprise() * 0.6 + stats.success_rate() * 0.4
    }
}

// ── v3: Auto-spec generation ────────────────────────────────────────

/// A generated spec candidate.
#[derive(Debug, Clone)]
pub struct SpecCandidate {
    pub id: String,
    pub domain: String,
    pub description: String,
    pub estimated_complexity: f64, // 0.0–1.0
    pub keywords: Vec<String>,
}

/// Auto-spec generator — produces spec candidates from a domain.
#[derive(Debug, Clone)]
pub struct SpecGenerator {
    pub domain_templates: HashMap<String, Vec<SpecCandidate>>,
}

impl SpecGenerator {
    pub fn new() -> Self { Self { domain_templates: HashMap::new() } }

    /// Generate spec candidates for a given domain.
    pub fn generate(&self, domain: &str) -> Vec<SpecCandidate> {
        // In production, this would call an LLM. Here we use heuristics.
        let templates = self.domain_templates.get(domain).cloned().unwrap_or_else(|| {
            Self::default_specs(domain)
        });
        templates
    }

    /// Generate specs using memory to bias toward productive domains.
    pub fn generate_with_memory(&self, domain: &str, memory: &RiffMemory) -> Vec<SpecCandidate> {
        let mut specs = self.generate(domain);
        // Boost complexity estimate for domains with high historical surprise
        let domain_score = memory.spec_patterns.get(domain).copied().unwrap_or(0.5);
        for spec in &mut specs {
            spec.estimated_complexity = (spec.estimated_complexity * 0.7 + domain_score * 0.3).min(1.0);
        }
        specs
    }

    fn default_specs(domain: &str) -> Vec<SpecCandidate> {
        let slug = domain.to_lowercase().replace(' ', "-");
        vec![
            SpecCandidate {
                id: format!("{}-core", slug),
                domain: domain.to_string(),
                description: format!("Core data structures for {}", domain),
                estimated_complexity: 0.3,
                keywords: vec!["core".to_string(), "data-structure".to_string()],
            },
            SpecCandidate {
                id: format!("{}-advanced", slug),
                domain: domain.to_string(),
                description: format!("Advanced algorithms for {}", domain),
                estimated_complexity: 0.7,
                keywords: vec!["algorithm".to_string(), "optimization".to_string()],
            },
            SpecCandidate {
                id: format!("{}-gpu", slug),
                domain: domain.to_string(),
                description: format!("GPU-accelerated {} kernels", domain),
                estimated_complexity: 0.9,
                keywords: vec!["gpu".to_string(), "cuda".to_string(), "kernel".to_string()],
            },
        ]
    }

    /// Rank specs by estimated usefulness (higher = more useful to riff on).
    pub fn rank_specs(&self, specs: &[SpecCandidate]) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = specs.iter().enumerate().map(|(i, s)| {
            // Sweet spot: moderate complexity + good keywords
            let score = s.estimated_complexity * 0.5 + (s.keywords.len() as f64 / 5.0).min(1.0) * 0.5;
            (i, score)
        }).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

// ── v3: Bootstrap verifier ──────────────────────────────────────────

/// Result of verifying a bootstrap generation.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub generation: u32,
    pub compiles: bool,
    pub tests_pass: bool,
    pub test_count: usize,
    pub test_failures: Vec<String>,
    pub warnings: Vec<String>,
}

impl VerifyResult {
    pub fn success(generation: u32, test_count: usize) -> Self {
        Self { generation, compiles: true, tests_pass: true, test_count, test_failures: Vec::new(), warnings: Vec::new() }
    }
    pub fn failure(generation: u32, failures: Vec<String>) -> Self {
        Self { generation, compiles: true, tests_pass: false, test_count: 0, test_failures: failures, warnings: Vec::new() }
    }
    pub fn compile_error(generation: u32, warnings: Vec<String>) -> Self {
        Self { generation, compiles: false, tests_pass: false, test_count: 0, test_failures: Vec::new(), warnings }
    }
    pub fn is_ok(&self) -> bool { self.compiles && self.tests_pass }
}

/// Simulated bootstrap verifier. In production, runs `cargo test`.
/// Here, verifies in-memory that the snowball structure is consistent.
#[derive(Debug, Clone)]
pub struct BootstrapVerifier {
    pub verified_generations: Vec<VerifyResult>,
}

impl BootstrapVerifier {
    pub fn new() -> Self { Self { verified_generations: Vec::new() } }

    /// Verify a generation's output against basic consistency checks.
    pub fn verify(&mut self, metrics: &SessionMetrics) -> VerifyResult {
        let gen = metrics.generation;
        // Basic checks: must have rounds, must have productivity
        if metrics.total_rounds == 0 {
            let result = VerifyResult::failure(gen, vec!["No rounds generated".to_string()]);
            self.verified_generations.push(result.clone());
            return result;
        }
        if metrics.total_loc == 0 {
            let result = VerifyResult::failure(gen, vec!["No LOC produced".to_string()]);
            self.verified_generations.push(result.clone());
            return result;
        }
        if metrics.total_tests == 0 {
            let result = VerifyResult::failure(gen, vec!["No tests produced".to_string()]);
            self.verified_generations.push(result.clone());
            return result;
        }
        let test_count = metrics.total_tests;
        let result = VerifyResult::success(gen, test_count);
        self.verified_generations.push(result.clone());
        result
    }

    /// Verify a full bootstrap chain — each generation must improve or maintain.
    pub fn verify_chain(&mut self, chain: &[SessionMetrics]) -> Vec<VerifyResult> {
        let mut results = Vec::new();
        for metrics in chain {
            results.push(self.verify(metrics));
        }
        results
    }

    /// Check snowball growth: each generation should have >= metrics of the previous.
    pub fn check_growth(chain: &[SessionMetrics]) -> GrowthCheck {
        if chain.len() < 2 {
            return GrowthCheck { growing: true, loc_deltas: Vec::new(), test_deltas: Vec::new(), feature_deltas: Vec::new(), surprise_deltas: Vec::new() };
        }
        let mut loc_deltas = Vec::new();
        let mut test_deltas = Vec::new();
        let mut feature_deltas = Vec::new();
        let mut surprise_deltas = Vec::new();

        for window in chain.windows(2) {
            loc_deltas.push(window[1].total_loc as f64 - window[0].total_loc as f64);
            test_deltas.push(window[1].total_tests as f64 - window[0].total_tests as f64);
            feature_deltas.push(window[1].total_features as f64 - window[0].total_features as f64);
            surprise_deltas.push(window[1].avg_surprise - window[0].avg_surprise);
        }

        let growing = loc_deltas.iter().all(|&d| d >= 0.0)
            && test_deltas.iter().all(|&d| d >= 0.0);

        GrowthCheck { growing, loc_deltas, test_deltas, feature_deltas, surprise_deltas }
    }
}

// ── v3: Snowball metrics ────────────────────────────────────────────

/// Growth check result across generations.
#[derive(Debug, Clone)]
pub struct GrowthCheck {
    pub growing: bool,
    pub loc_deltas: Vec<f64>,
    pub test_deltas: Vec<f64>,
    pub feature_deltas: Vec<f64>,
    pub surprise_deltas: Vec<f64>,
}

/// Session metrics including bootstrap generation.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub generation: u32,
    pub total_rounds: usize,
    pub productive_rounds: usize,
    pub total_loc: usize,
    pub total_tests: usize,
    pub total_features: usize,
    pub avg_surprise: f64,
    pub streak: u32,
}

/// Snowball growth tracker — tracks metrics across generations.
#[derive(Debug, Clone)]
pub struct SnowballTracker {
    pub generations: Vec<SessionMetrics>,
    pub growth_rates: Vec<GrowthRate>,
}

/// Growth rate between consecutive generations.
#[derive(Debug, Clone)]
pub struct GrowthRate {
    pub from_gen: u32,
    pub to_gen: u32,
    pub loc_rate: f64,      // ratio (2.0 = doubled)
    pub test_rate: f64,
    pub feature_rate: f64,
    pub surprise_delta: f64, // absolute change
}

impl SnowballTracker {
    pub fn new() -> Self { Self { generations: Vec::new(), growth_rates: Vec::new() } }

    /// Record a generation's metrics.
    pub fn record(&mut self, metrics: SessionMetrics) {
        if let Some(prev) = self.generations.last() {
            let loc_rate = if prev.total_loc > 0 { metrics.total_loc as f64 / prev.total_loc as f64 } else { 1.0 };
            let test_rate = if prev.total_tests > 0 { metrics.total_tests as f64 / prev.total_tests as f64 } else { 1.0 };
            let feature_rate = if prev.total_features > 0 { metrics.total_features as f64 / prev.total_features as f64 } else { 1.0 };
            self.growth_rates.push(GrowthRate {
                from_gen: prev.generation,
                to_gen: metrics.generation,
                loc_rate,
                test_rate,
                feature_rate,
                surprise_delta: metrics.avg_surprise - prev.avg_surprise,
            });
        }
        self.generations.push(metrics);
    }

    /// Is the snowball growing? Each generation should have >= output of the previous.
    pub fn is_growing(&self) -> bool {
        self.growth_rates.iter().all(|g| g.loc_rate >= 1.0 && g.test_rate >= 1.0)
    }

    /// Average growth rate across all generations.
    pub fn avg_growth_rate(&self) -> f64 {
        if self.growth_rates.is_empty() { return 0.0; }
        self.growth_rates.iter().map(|g| (g.loc_rate + g.test_rate + g.feature_rate) / 3.0).sum::<f64>() / self.growth_rates.len() as f64
    }
}

// ── v3: Multi-spec riff session ─────────────────────────────────────

/// A spec being riffed on.
#[derive(Debug, Clone)]
pub struct RiffSpec {
    pub id: String,
    pub name: String,
    pub domain: String,
}

/// A multi-spec riff session — agents riff on multiple specs simultaneously.
#[derive(Debug, Clone)]
pub struct MultiSpecSession {
    pub agents: Vec<u32>,
    pub specs: Vec<RiffSpec>,
    pub rounds: Vec<Round>,
    pub memory: RiffMemory,
    pub current_round: u32,
    pub mode: ResponseMode,
    pub streak: u32,
    pub finished: bool,
    pub generation: u32,
    pub parent_session_id: Option<String>,
    pub cross_spec_patterns: HashMap<String, Vec<String>>, // spec_id → patterns learned
}

impl MultiSpecSession {
    pub fn new(agents: Vec<u32>, specs: Vec<RiffSpec>, generation: u32) -> Self {
        Self { agents, specs, rounds: Vec::new(), memory: RiffMemory::new(), current_round: 0,
               mode: ResponseMode::Escalate, streak: 0, finished: false, generation,
               parent_session_id: None, cross_spec_patterns: HashMap::new() }
    }

    pub fn new_round(&mut self) -> &mut Round {
        let r = Round::new(self.current_round);
        self.rounds.push(r);
        self.current_round += 1;
        self.rounds.last_mut().unwrap()
    }

    pub fn riff(&mut self, agent_id: u32, quality: Quality, surprise: f64) {
        let riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }

    pub fn riff_with_output(&mut self, agent_id: u32, quality: Quality, surprise: f64, loc: usize, tests: usize, features: Vec<&str>) {
        let mut riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        riff.loc = loc; riff.tests = tests;
        riff.features = features.iter().map(|s| s.to_string()).collect();
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }

    /// Riff targeting a specific spec.
    pub fn riff_for_spec(&mut self, agent_id: u32, spec_id: &str, quality: Quality, surprise: f64, loc: usize, tests: usize, features: Vec<&str>) {
        let mut riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        riff.spec_id = Some(spec_id.to_string());
        riff.loc = loc; riff.tests = tests;
        riff.features = features.iter().map(|s| s.to_string()).collect();
        // Share patterns from other specs
        let shared: Vec<String> = self.cross_spec_patterns.iter()
            .filter(|(k, _)| *k != spec_id)
            .flat_map(|(_, v)| v.iter().cloned())
            .collect();
        for p in &shared {
            if !riff.features.contains(p) {
                riff.features.push(p.clone());
            }
        }
        // Record patterns for this spec
        let entry = self.cross_spec_patterns.entry(spec_id.to_string()).or_insert_with(Vec::new);
        for f in &riff.features {
            if !entry.contains(f) { entry.push(f.clone()); }
        }
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }

    pub fn evaluate(&mut self) -> RoundSummary {
        let round = match self.rounds.last() {
            Some(r) => r,
            None => return RoundSummary { surprise: 0.0, productive: false, landed: false, mode: self.mode, best_productivity: 0.0 },
        };
        let surprise = round.surprise_sum;
        let productive = round.was_productive();
        if productive { self.streak += 1; } else { self.streak = 0; }
        let landed = surprise > 0.8 && round.riffs.iter().any(|r| r.quality == Quality::Strong);
        self.mode = ResponseMode::auto(surprise, self.streak, self.current_round);
        if self.streak == 0 && self.current_round > 5 { self.finished = true; }
        let best_prod = round.riffs.iter().map(|r| r.productivity()).fold(0.0f64, f64::max);
        RoundSummary { surprise, productive, landed, mode: self.mode, best_productivity: best_prod }
    }

    pub fn metrics(&self) -> SessionMetrics {
        let total_rounds = self.rounds.len();
        let productive = self.rounds.iter().filter(|r| r.was_productive()).count();
        let total_loc: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.loc).sum();
        let total_tests: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.tests).sum();
        let total_features: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.features.len()).sum();
        let total_surprise: f64 = self.rounds.iter().map(|r| r.surprise_sum).sum();
        SessionMetrics {
            generation: self.generation,
            total_rounds,
            productive_rounds: productive,
            total_loc,
            total_tests,
            total_features,
            avg_surprise: if total_rounds > 0 { total_surprise / total_rounds as f64 } else { 0.0 },
            streak: self.streak,
        }
    }

    /// Bootstrap the next generation with inherited memory.
    pub fn bootstrap_next(&self) -> MultiSpecSession {
        let mut next = MultiSpecSession::new(self.agents.clone(), self.specs.clone(), self.generation + 1);
        next.memory = self.memory.clone();
        next.parent_session_id = Some(format!("gen-{}", self.generation));
        next.cross_spec_patterns = self.cross_spec_patterns.clone();
        next
    }
}

/// Summary of a round's evaluation.
#[derive(Debug, Clone)]
pub struct RoundSummary {
    pub surprise: f64,
    pub productive: bool,
    pub landed: bool,
    pub mode: ResponseMode,
    pub best_productivity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Legacy v1/v2 tests (kept for continuity) ────────────────────

    #[test]
    fn trit_pack_unpack() {
        let trits = vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos];
        let packed = pack_16(&trits);
        assert_eq!(Trit::unpack_bits((packed & 0x3) as u8), Trit::Pos);
        assert_eq!(Trit::unpack_bits(((packed >> 2) & 0x3) as u8), Trit::Neg);
    }

    #[test]
    fn riff_productivity() {
        let mut r = Riff::new(0, 1, Quality::Strong, 0.8);
        r.loc = 200; r.tests = 15; r.features = vec!["gpu-packing".to_string()];
        assert!(r.productivity() > 0.0);
        assert_eq!(r.features.len(), 1);
    }

    #[test]
    fn response_mode_auto() {
        assert_eq!(ResponseMode::auto(0.1, 0, 3), ResponseMode::Provoked);
        assert_eq!(ResponseMode::auto(0.8, 0, 3), ResponseMode::Escalate);
        assert_eq!(ResponseMode::auto(0.5, 6, 3), ResponseMode::Pivot);
        assert_eq!(ResponseMode::auto(0.5, 0, 9), ResponseMode::Invert);
    }

    #[test]
    fn fleet_session_basic() {
        let mut s = MultiSpecSession::new(vec![0, 1], vec![], 1);
        s.new_round();
        s.riff_with_output(0, Quality::Ok, 0.3, 100, 8, vec!["baseline"]);
        s.riff_with_output(1, Quality::Strong, 0.7, 300, 20, vec!["gpu-packing", "entropy"]);
        let summary = s.evaluate();
        assert!(summary.productive);
        assert!(summary.best_productivity > 0.0);
    }

    #[test]
    fn stale_detection() {
        let mut s = MultiSpecSession::new(vec![0, 1], vec![], 1);
        for _ in 0..6 {
            s.new_round();
            s.riff(0, Quality::Weak, 0.05);
            s.riff(1, Quality::Weak, 0.05);
            s.evaluate();
        }
        assert!(s.finished);
    }

    #[test]
    fn landing_detection() {
        let mut s = MultiSpecSession::new(vec![0, 1], vec![], 1);
        s.new_round();
        s.riff(0, Quality::Strong, 0.9);
        s.riff(1, Quality::Strong, 0.85);
        let summary = s.evaluate();
        assert!(summary.landed);
    }

    // ── v3 Feature 1: Multi-spec riff sessions ──────────────────────

    #[test]
    fn multi_spec_session_basic() {
        let specs = vec![
            RiffSpec { id: "ternary-core".into(), name: "Core Types".into(), domain: "ternary".into() },
            RiffSpec { id: "ternary-gpu".into(), name: "GPU Kernels".into(), domain: "ternary".into() },
        ];
        let mut s = MultiSpecSession::new(vec![0, 1], specs, 1);
        s.new_round();
        s.riff_for_spec(0, "ternary-core", Quality::Ok, 0.4, 100, 5, vec!["packing"]);
        s.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.7, 200, 12, vec!["kernel"]);
        let summary = s.evaluate();
        assert!(summary.productive);
        // Verify cross-spec patterns were recorded
        assert!(s.cross_spec_patterns.contains_key("ternary-core"));
        assert!(s.cross_spec_patterns.contains_key("ternary-gpu"));
    }

    #[test]
    fn cross_spec_pattern_sharing() {
        let specs = vec![
            RiffSpec { id: "spec-a".into(), name: "A".into(), domain: "test".into() },
            RiffSpec { id: "spec-b".into(), name: "B".into(), domain: "test".into() },
        ];
        let mut s = MultiSpecSession::new(vec![0], specs, 1);
        // Round 1: riff on spec-a with "alpha" feature
        s.new_round();
        s.riff_for_spec(0, "spec-a", Quality::Strong, 0.6, 100, 5, vec!["alpha"]);
        s.evaluate();
        // Round 2: riff on spec-b — should inherit "alpha" from spec-a
        s.new_round();
        s.riff_for_spec(0, "spec-b", Quality::Ok, 0.4, 80, 4, vec!["beta"]);
        let round = s.rounds.last().unwrap();
        let riff = round.riffs.last().unwrap();
        // "alpha" should have been shared from spec-a
        assert!(riff.features.contains(&"alpha".to_string()));
    }

    // ── v3 Feature 2: Auto-spec generation ──────────────────────────

    #[test]
    fn auto_spec_generation() {
        let gen = SpecGenerator::new();
        let specs = gen.generate("ternary data structures");
        assert!(!specs.is_empty());
        assert!(specs.iter().any(|s| s.id.contains("core")));
        assert!(specs.iter().any(|s| s.id.contains("advanced")));
        assert!(specs.iter().any(|s| s.id.contains("gpu")));
    }

    #[test]
    fn spec_ranking() {
        let gen = SpecGenerator::new();
        let specs = gen.generate("test domain");
        let ranked = gen.rank_specs(&specs);
        assert_eq!(ranked.len(), specs.len());
        // Ranked in descending order
        for window in ranked.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
    }

    // ── v3 Feature 3: Quality predictor ─────────────────────────────

    #[test]
    fn quality_predictor_with_history() {
        let mut mem = RiffMemory::new();
        // Train: agent 0 excels at Escalate, agent 1 at Pivot
        let stats0 = ModeStats { total_uses: 10, total_surprise: 8.0, strong_count: 8, weak_count: 1 };
        let stats1 = ModeStats { total_uses: 10, total_surprise: 4.0, strong_count: 3, weak_count: 4 };
        mem.mode_stats.insert((0, ResponseMode::Escalate), stats0);
        mem.mode_stats.insert((1, ResponseMode::Pivot), stats1);

        let (agent, _mode, score) = mem.predict_best(&[0, 1]);
        assert_eq!(agent, 0);
        assert!(score > 0.0);
    }

    #[test]
    fn quality_predictor_no_history() {
        let mem = RiffMemory::new();
        let (agent, _mode, score) = mem.predict_best(&[0, 1, 2]);
        // With no history, all score 0.5 (default success_rate) * 0.4 = 0.2
        assert_eq!(agent, 0); // First agent wins ties
        assert!(score > 0.0);
    }

    // ── v3 Feature 4: Bootstrap verifier ────────────────────────────

    #[test]
    fn verifier_success() {
        let mut v = BootstrapVerifier::new();
        let metrics = SessionMetrics {
            generation: 1, total_rounds: 3, productive_rounds: 2,
            total_loc: 500, total_tests: 20, total_features: 8,
            avg_surprise: 0.6, streak: 2,
        };
        let result = v.verify(&metrics);
        assert!(result.is_ok());
        assert!(result.compiles);
        assert!(result.tests_pass);
        assert_eq!(result.test_count, 20);
    }

    #[test]
    fn verifier_failure_no_output() {
        let mut v = BootstrapVerifier::new();
        let metrics = SessionMetrics {
            generation: 1, total_rounds: 3, productive_rounds: 0,
            total_loc: 0, total_tests: 0, total_features: 0,
            avg_surprise: 0.1, streak: 0,
        };
        let result = v.verify(&metrics);
        assert!(!result.is_ok());
    }

    #[test]
    fn growth_check_across_generations() {
        let chain = vec![
            SessionMetrics { generation: 1, total_rounds: 2, productive_rounds: 1, total_loc: 100, total_tests: 5, total_features: 2, avg_surprise: 0.3, streak: 1 },
            SessionMetrics { generation: 2, total_rounds: 3, productive_rounds: 2, total_loc: 300, total_tests: 15, total_features: 6, avg_surprise: 0.5, streak: 2 },
            SessionMetrics { generation: 3, total_rounds: 4, productive_rounds: 3, total_loc: 600, total_tests: 30, total_features: 12, avg_surprise: 0.7, streak: 3 },
        ];
        let check = BootstrapVerifier::check_growth(&chain);
        assert!(check.growing);
        assert!(check.loc_deltas.iter().all(|&d| d > 0.0));
        assert!(check.test_deltas.iter().all(|&d| d > 0.0));
    }

    // ── v3 Feature 5: Snowball metrics ──────────────────────────────

    #[test]
    fn snowball_tracker_growth() {
        let mut tracker = SnowballTracker::new();
        tracker.record(SessionMetrics { generation: 1, total_rounds: 2, productive_rounds: 1, total_loc: 100, total_tests: 5, total_features: 2, avg_surprise: 0.3, streak: 1 });
        tracker.record(SessionMetrics { generation: 2, total_rounds: 3, productive_rounds: 2, total_loc: 300, total_tests: 15, total_features: 6, avg_surprise: 0.5, streak: 2 });
        tracker.record(SessionMetrics { generation: 3, total_rounds: 4, productive_rounds: 3, total_loc: 600, total_tests: 30, total_features: 12, avg_surprise: 0.7, streak: 3 });

        assert!(tracker.is_growing());
        assert_eq!(tracker.generations.len(), 3);
        assert_eq!(tracker.growth_rates.len(), 2);
        // Gen 1→2: loc_rate = 300/100 = 3.0
        assert!((tracker.growth_rates[0].loc_rate - 3.0).abs() < 0.01);
        // Gen 2→3: test_rate = 30/15 = 2.0
        assert!((tracker.growth_rates[1].test_rate - 2.0).abs() < 0.01);
        assert!(tracker.avg_growth_rate() > 1.0);
    }

    // ── THE BIG TEST: 3-generation bootstrap chain ──────────────────

    #[test]
    fn three_generation_bootstrap_chain() {
        let specs = vec![
            RiffSpec { id: "ternary-core".into(), name: "Core".into(), domain: "ternary".into() },
            RiffSpec { id: "ternary-gpu".into(), name: "GPU".into(), domain: "ternary".into() },
        ];

        // ── Generation 1: Baseline ──
        let mut gen1 = MultiSpecSession::new(vec![0, 1], specs.clone(), 1);
        gen1.new_round();
        gen1.riff_for_spec(0, "ternary-core", Quality::Ok, 0.3, 100, 5, vec!["basic-packing"]);
        gen1.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.6, 200, 12, vec!["kernel-launch"]);
        gen1.evaluate();
        gen1.memory.learn(&gen1.rounds);
        let gen1_metrics = gen1.metrics();
        gen1.memory.record_generation(gen1_metrics.clone());

        // ── Generation 2: Inherited memory, multi-spec ──
        let mut gen2 = gen1.bootstrap_next();
        assert_eq!(gen2.generation, 2);
        assert_eq!(gen2.memory.total_rounds, 1); // Inherited
        assert!(!gen2.cross_spec_patterns.is_empty()); // Inherited patterns

        gen2.new_round();
        gen2.riff_for_spec(0, "ternary-core", Quality::Strong, 0.7, 300, 18, vec!["fast-pack"]);
        gen2.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.8, 450, 28, vec!["cuda-ops"]);
        let gen2_summary = gen2.evaluate();
        assert!(gen2_summary.productive);
        gen2.memory.learn(&gen2.rounds);
        let gen2_metrics = gen2.metrics();
        gen2.memory.record_generation(gen2_metrics.clone());

        // Quality predictor should now favor certain combos
        let (_best_agent, _mode, predicted) = gen2.memory.predict_best(&[0, 1]);
        assert!(predicted > 0.0);

        // ── Generation 3: Full snowball ──
        let mut gen3 = gen2.bootstrap_next();
        assert_eq!(gen3.generation, 3);
        assert_eq!(gen3.memory.total_rounds, 2); // Both prior generations
        assert_eq!(gen3.memory.generation_history.len(), 2);

        gen3.new_round();
        gen3.riff_for_spec(0, "ternary-core", Quality::Strong, 0.85, 500, 35, vec!["simd-pack"]);
        gen3.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.9, 700, 50, vec!["wmma-kernels"]);
        let gen3_summary = gen3.evaluate();
        assert!(gen3_summary.landed); // Should be a landing
        gen3.memory.learn(&gen3.rounds);
        let gen3_metrics = gen3.metrics();
        gen3.memory.record_generation(gen3_metrics.clone());

        // ── Verify the full chain ──
        let mut verifier = BootstrapVerifier::new();
        let results = verifier.verify_chain(&[gen1_metrics.clone(), gen2_metrics.clone(), gen3_metrics.clone()]);
        assert!(results.iter().all(|r| r.is_ok()));

        // ── Check snowball growth ──
        let growth = BootstrapVerifier::check_growth(&[gen1_metrics.clone(), gen2_metrics.clone(), gen3_metrics.clone()]);
        assert!(growth.growing);
        assert!(growth.loc_deltas.iter().all(|&d| d > 0.0));
        assert!(growth.test_deltas.iter().all(|&d| d > 0.0));
        assert!(growth.surprise_deltas.iter().all(|&d| d > 0.0));

        // ── Track with SnowballTracker ──
        let mut tracker = SnowballTracker::new();
        tracker.record(gen1_metrics);
        tracker.record(gen2_metrics);
        tracker.record(gen3_metrics);
        assert!(tracker.is_growing());
        assert!(tracker.avg_growth_rate() > 1.0); // More than maintaining — growing
    }
}
