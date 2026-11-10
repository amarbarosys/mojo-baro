//! Payload vs recipe: to continue a checkpoint on another host, either ship the
//! state file (transfer + fork) or send the tokens and recompute (prefill +
//! decode). `choose` predicts both from measured costs and takes the cheaper.
//! Plan: AMDHQ docs/design/latent-os/11-state-layer.md S2. The default fit is
//! dense qwen35 on gfx1100, medians from AMDHQ runs/latent-os/C2-long and
//! B4-long (2026-09-24). Not wired to a route yet (S2).
#![cfg_attr(not(test), allow(dead_code))]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Payload,
    Recipe,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub payload_s: f64,
    pub recipe_s: f64,
    pub choice: Choice,
}

#[derive(Debug, Clone)]
pub struct Fit {
    /// fork_s = fork_base_s + state_bytes * fork_s_per_byte (restore + suffix + decode).
    pub fork_base_s: f64,
    pub fork_s_per_byte: f64,
    /// (prompt tokens, cold prefill + decode seconds), ascending; linear between
    /// points, extrapolated from the last two outside them.
    pub recipe_points: Vec<(u64, f64)>,
}

impl Fit {
    /// B4-long fork medians: 184.9 MB 0.513 s, 581.2 MB 1.212 s, 577.2 MB 1.255 s,
    /// 2,150.0 MB 4.500 s; least squares: 0.080 s + 2.049 ms/MB, largest residual 0.059 s.
    /// Recipe medians: 8,000 tokens 3.29 s, 32,000 tokens 16.37 s.
    pub fn qwen35_gfx1100() -> Fit {
        Fit { fork_base_s: 0.080, fork_s_per_byte: 2.049e-9, recipe_points: vec![(8_000, 3.29), (32_000, 16.37)] }
    }

    pub fn fork_s(&self, state_bytes: u64) -> f64 {
        self.fork_base_s + state_bytes as f64 * self.fork_s_per_byte
    }

    pub fn recipe_s(&self, tokens: u64) -> f64 {
        let p = &self.recipe_points;
        let i = p.iter().position(|&(n, _)| n >= tokens).unwrap_or(p.len() - 1).clamp(1, p.len() - 1);
        let ((n0, s0), (n1, s1)) = (p[i - 1], p[i]);
        s0 + (s1 - s0) * (tokens as f64 - n0 as f64) / (n1 as f64 - n0 as f64)
    }
}

/// `link_bytes_per_s` is measured per peer, never configured.
pub fn choose(fit: &Fit, state_bytes: u64, tokens: u64, link_bytes_per_s: f64) -> Prediction {
    let payload_s = state_bytes as f64 / link_bytes_per_s + fit.fork_s(state_bytes);
    let recipe_s = fit.recipe_s(tokens);
    let choice = if payload_s < recipe_s { Choice::Payload } else { Choice::Recipe };
    Prediction { payload_s, recipe_s, choice }
}

#[cfg(test)]
mod tests {
    use super::*;

    // B4-long measured winners (AMDHQ runs/latent-os/B4-long-2026-09-24.md), link
    // rates from B4-mini's rig: 100mbit 11.9 MB/s, 1gbit 119 MB/s, 10gbit 768 MB/s.
    #[test]
    fn picks_the_measured_winner_in_every_b4_long_cell() {
        use Choice::*;
        let fit = Fit::qwen35_gfx1100();
        let rates = [11.9e6, 119e6, 768e6];
        let arms: [(&str, u64, u64, [Choice; 3]); 4] = [
            ("f32-8k", 581_205_316, 8_000, [Recipe, Recipe, Payload]),
            ("int8-8k", 184_859_716, 8_000, [Recipe, Payload, Payload]),
            ("f32-32k", 2_149_971_012, 32_000, [Recipe, Recipe, Payload]),
            ("int8-32k", 577_171_012, 32_000, [Recipe, Payload, Payload]),
        ];
        for (name, bytes, tokens, won) in arms {
            for (rate, want) in rates.iter().zip(won) {
                let p = choose(&fit, bytes, tokens, *rate);
                assert_eq!(p.choice, want, "{name} at {rate} B/s: {p:?}");
            }
        }
    }

    #[test]
    fn fit_reproduces_its_own_points() {
        let fit = Fit::qwen35_gfx1100();
        assert!((fit.recipe_s(8_000) - 3.29).abs() < 1e-9);
        assert!((fit.recipe_s(32_000) - 16.37).abs() < 1e-9);
        assert!((fit.recipe_s(20_000) - 9.83).abs() < 1e-9);
        for (bytes, measured) in [(184_859_716u64, 0.513), (581_205_316, 1.212), (577_171_012, 1.255), (2_149_971_012, 4.500)] {
            assert!((fit.fork_s(bytes) - measured).abs() < 0.06, "fork fit at {bytes} B: {} vs {measured}", fit.fork_s(bytes));
        }
    }
}
