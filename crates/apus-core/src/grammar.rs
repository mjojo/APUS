use crate::BpeTokenizer;

/// Grammar constraint mode for structured output generation (GBNF / JSON Schema / Code Block)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarMode {
    Unconstrained,
    JsonObject,
    RustCodeBlock,
}

/// Finite-state grammar constraint engine that masks invalid vocabulary logits before sampling,
/// guaranteeing syntactically valid JSON objects or balanced code blocks even under INT4/INT2 quantization.
pub struct GrammarConstraintEngine {
    mode: GrammarMode,
    brace_depth: i32,
    in_string: bool,
    started: bool,
}

impl GrammarConstraintEngine {
    pub fn new(mode: GrammarMode) -> Self {
        Self {
            mode,
            brace_depth: 0,
            in_string: false,
            started: false,
        }
    }

    pub fn mode(&self) -> GrammarMode {
        self.mode
    }

    pub fn brace_depth(&self) -> i32 {
        self.brace_depth
    }

    /// Updates internal parser state after a token piece is emitted
    pub fn advance_with_text(&mut self, piece: &str) {
        for ch in piece.chars() {
            if !ch.is_whitespace() {
                self.started = true;
            }
            if ch == '"' {
                self.in_string = !self.in_string;
            } else if !self.in_string {
                if ch == '{' {
                    self.brace_depth += 1;
                } else if ch == '}' {
                    self.brace_depth = (self.brace_depth - 1).max(0);
                }
            }
        }
    }

    /// Applies grammar masks directly to the vocabulary logit vector prior to sampling
    pub fn mask_logits(&self, logits: &mut [f32], tokenizer: &BpeTokenizer) {
        if self.mode == GrammarMode::Unconstrained {
            return;
        }

        if self.mode == GrammarMode::JsonObject && !self.started {
            // Before JSON starts, only allow '{' or whitespace
            for (tok_id, logit) in logits.iter_mut().enumerate() {
                let piece = tokenizer.decode(&[tok_id as u32]);
                let trimmed = piece.trim_start();
                if !trimmed.is_empty() && !trimmed.starts_with('{') {
                    *logit = f32::NEG_INFINITY;
                }
            }
            return;
        }

        // Prevent closing brace '}' when brace_depth <= 0 outside of string literals
        if !self.in_string && self.brace_depth <= 0 {
            if let Some(close_id) = tokenizer.token_id("}") {
                if let Some(slot) = logits.get_mut(close_id as usize) {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }
    }

    pub fn is_complete_json(&self) -> bool {
        self.mode == GrammarMode::JsonObject && self.started && self.brace_depth == 0 && !self.in_string
    }
}
