use crate::header::{
    BG_DARK, BORDER_SLATE, CYAN_NEON, EMERALD_OK, GOLD_BADGE, INDIGO_GLOW, TEXT_MUTED,
    TelemetryHeaderState, render_header,
};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
};

/// State for the fullscreen APUS Telemetry HUD (btop++ / Aerospace Cyberpunk style)
pub struct DashboardState {
    pub header: TelemetryHeaderState,
    pub tick: u64,
    pub session_name: String,
    pub context_tokens_k: f32,
    pub l2_used_gb: f32,
    pub l2_total_gb: f32,
    pub l3_stream_gb: f32,
    pub miss_penalty_ms: f32,
    pub nvme_read_gbs: f32,
    pub queue_depth: usize,
    pub max_queue_depth: usize,
    pub active_layer: u16,
    pub total_layers: u16,
    pub nvme_cap_pct: u8,
    pub snapshot_saved_banner: Option<String>,
    pub chat_messages: Vec<(String, String)>,
    pub prompt_input: String,
    pub pending_stream_tokens: Vec<String>,
    /// Live expert activation counts for the current layer (indexed 0..288)
    pub expert_heatmap: Vec<u16>,
    /// Live I/O-Aware Continuous Batching tickets
    pub live_tickets: Vec<(String, String, bool)>,
    /// Manual vertical scroll offset for the chat panel
    pub chat_scroll_offset: u16,
    /// Active focused panel index (0 = Session Chat, 1 = ARC Memory, 2 = NVMe I/O, 3 = Braille Heatmap, 4 = Ticket Queue)
    pub active_panel_index: u8,
}

impl Default for DashboardState {
    fn default() -> Self {
        let mut initial_heatmap = vec![0u16; 288];
        for &hot_idx in &[3usize, 14, 29, 64, 98, 142, 201, 255] {
            initial_heatmap[hot_idx] = 5;
        }
        Self {
            header: TelemetryHeaderState::default(),
            tick: 0,
            session_name: "coding_rust.roost".into(),
            context_tokens_k: 48.2,
            l2_used_gb: 18.4,
            l2_total_gb: 24.0,
            l3_stream_gb: 346.4,
            miss_penalty_ms: 1.4,
            nvme_read_gbs: 5.84,
            queue_depth: 18,
            max_queue_depth: 64,
            active_layer: 42,
            total_layers: 75,
            nvme_cap_pct: 80,
            snapshot_saved_banner: None,
            chat_messages: vec![
                (
                    "You".into(),
                    "Write a lock-free SQ/CQ ring buffer for io_uring in Rust.".into(),
                ),
                (
                    "APUS :: GLM-5.2-744B".into(),
                    "To eliminate lock contention on the SQ/CQ ring buffer, use cache-padded atomic indices with `Acquire`/`Release` ordering:\n\npub struct IoRingQueue<T, const N: usize> {\n    buffer: [UnsafeCell<MaybeUninit<T>>; N],\n    head: CachePadded<AtomicUsize>,\n    tail: CachePadded<AtomicUsize>,\n}".into(),
                ),
            ],
            prompt_input: String::new(),
            pending_stream_tokens: Vec::new(),
            expert_heatmap: initial_heatmap,
            live_tickets: vec![
                (
                    "[ACTIVE]".into(),
                    "#01 Generating IoRingQueue (Layer 42/75) ── ETA ~12s".into(),
                    true,
                ),
                (
                    "[QUEUED]".into(),
                    "#02 Auditing unsafe memory blocks in crate apus-core".into(),
                    false,
                ),
            ],
            chat_scroll_offset: 0,
            active_panel_index: 0,
        }
    }
}

impl DashboardState {
    pub fn on_tick(&mut self) {
        self.tick += 1;
        let wave = (self.tick as f32 * 0.15).sin();
        self.header.tok_per_sec = 6.42 + wave * 0.35;
        self.header.hit_rate = (91.8 + (self.tick as f32 * 0.07).cos() * 1.1).clamp(85.0, 98.5);
        self.nvme_read_gbs = (5.84 + wave * 0.65).clamp(1.2, 6.9);
        self.queue_depth = (18isize + (wave * 6.0) as isize).clamp(4, 32) as usize;
        self.active_layer = ((self.tick / 3) % (self.total_layers as u64)) as u16 + 1;

        // Stream pending tokens into the active assistant response in real time
        if !self.pending_stream_tokens.is_empty() {
            let next_tok = self.pending_stream_tokens.remove(0);
            if let Some((sender, body)) = self.chat_messages.last_mut()
                && sender.starts_with("APUS")
            {
                body.push_str(&next_tok);
            }
            self.context_tokens_k += 0.002;
        }
    }

    /// Submits the current user prompt and schedules an asynchronous token stream
    pub fn submit_prompt(&mut self) {
        let trimmed = self.prompt_input.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        self.chat_messages.push(("You".into(), trimmed.clone()));
        self.chat_messages
            .push(("APUS :: GLM-5.2-744B".into(), String::new()));
        self.prompt_input.clear();

        let response = format!(
            "[Streamed via 4K Direct I/O + AVX2 FMA3 | MTP 2x] Routed 8/288 experts per layer for prompt \"{trimmed}\". ARC Hit Ratio held at {:.1}% with 0 B OS swap.",
            self.header.hit_rate
        );
        self.pending_stream_tokens = response
            .split_inclusive(' ')
            .map(|s| s.to_string())
            .collect();
    }

    pub fn apply_runtime_response(
        &mut self,
        reply_text: String,
        tok_per_sec: f32,
        hit_rate_pct: f32,
        context_tokens_total: usize,
        nvme_temp_c: u8,
    ) {
        self.header.tok_per_sec = tok_per_sec;
        self.header.hit_rate = hit_rate_pct;
        self.header.nvme_temp_c = nvme_temp_c;
        self.context_tokens_k = 48.2 + (context_tokens_total as f32 / 1000.0);
        self.pending_stream_tokens = reply_text
            .split_inclusive(' ')
            .map(|s| s.to_string())
            .collect();
    }

    /// Records a list of routed expert indices (0..288) into the live Braille heatmap
    pub fn record_routed_experts(&mut self, active_experts: &[u16]) {
        for slot in &mut self.expert_heatmap {
            *slot = slot.saturating_sub(1);
        }
        for &exp in active_experts {
            let idx = (exp as usize) % self.expert_heatmap.len().max(1);
            self.expert_heatmap[idx] = self.expert_heatmap[idx].saturating_add(3).min(15);
        }
    }

    pub fn scroll_chat_up(&mut self) {
        self.chat_scroll_offset = self.chat_scroll_offset.saturating_add(2);
    }

    pub fn scroll_chat_down(&mut self) {
        self.chat_scroll_offset = self.chat_scroll_offset.saturating_sub(2);
    }

    pub fn next_panel(&mut self) {
        self.active_panel_index = (self.active_panel_index + 1) % 5;
    }

    pub fn toggle_nvme_cap(&mut self) {
        self.nvme_cap_pct = match self.nvme_cap_pct {
            80 => 100,
            100 => 60,
            _ => 80,
        };
    }
}

/// Converts a `ratatui::buffer::Buffer` into a printable multi-line UTF-8 string
/// for headless terminal inspection and automated E2E TUI testing.
pub fn buffer_to_plain_text(buf: &ratatui::buffer::Buffer) -> String {
    let area = buf.area;
    let mut out = String::with_capacity((area.width as usize + 1) * area.height as usize);
    for y in 0..area.height {
        let mut line = String::with_capacity(area.width as usize);
        for x in 0..area.width {
            line.push_str(buf[(x, y)].symbol());
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

/// Renders the fullscreen APUS Telemetry HUD
pub fn render_dashboard(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(BG_DARK)), area);

    let root_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // Top telemetry header bar
            Constraint::Min(12),   // Main workspace (Session Chat Left | Telemetry Right)
            Constraint::Length(4), // I/O-Aware Continuous Batching Ticket Queue
            Constraint::Length(1), // Bottom hotkey bar & mjojo author signature
        ])
        .split(area);

    // 1. Telemetry Header
    render_header(frame, root_chunks[0], &state.header);

    // 2. Center area: Interactive Session left (54%) and 3 Telemetry panels right (46%)
    let main_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(54), Constraint::Percentage(46)])
        .split(root_chunks[1]);

    render_chat_panel(frame, main_cols[0], state);

    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6), // Memory & Cache Hierarchy (ARC ENGINE)
            Constraint::Length(5), // NVMe I/O & S.M.A.R.T.
            Constraint::Min(5),    // Routed Expert Heatmap (Braille Matrix)
        ])
        .split(main_cols[1]);

    render_arc_hierarchy(frame, right_rows[0], state);
    render_nvme_smart(frame, right_rows[1], state);
    render_braille_heatmap(frame, right_rows[2], state);

    // 3. I/O-Aware Continuous Batching Ticket Queue
    render_ticket_queue(frame, root_chunks[2], state);

    // 4. Bottom status & hotkey bar
    let status_right = state
        .snapshot_saved_banner
        .as_deref()
        .unwrap_or("APUS Engine // authored by mjojo");
    let footer = Paragraph::new(Line::from(vec![
        Span::styled("┗━ ", Style::default().fg(BORDER_SLATE)),
        Span::styled(
            "[Tab]",
            Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" Panel#{} ━ ", state.active_panel_index + 1),
            Style::default().fg(TEXT_MUTED),
        ),
        Span::styled(
            "[Ctrl+S]",
            Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Save .roost ━ ", Style::default().fg(TEXT_MUTED)),
        Span::styled(
            "[Ctrl+T]",
            Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" NVMe Cap ({}%) ━ ", state.nvme_cap_pct),
            Style::default().fg(TEXT_MUTED),
        ),
        Span::styled(
            "[Q/Esc]",
            Style::default().fg(GOLD_BADGE).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Quit ━━ ", Style::default().fg(TEXT_MUTED)),
        Span::styled(
            status_right,
            Style::default().fg(GOLD_BADGE).add_modifier(Modifier::BOLD),
        ),
    ]));
    frame.render_widget(footer, root_chunks[3]);
}

fn render_chat_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let border_col = if state.active_panel_index == 0 {
        CYAN_NEON
    } else {
        BORDER_SLATE
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_col).bg(BG_DARK))
        .title_top(Line::from(vec![
            Span::styled(
                format!("─ SESSION #04 [{}] ", state.session_name),
                Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("│ Context: {:.1}K MLA ", state.context_tokens_k),
                Style::default().fg(INDIGO_GLOW),
            ),
        ]));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = Vec::new();
    for (sender, text) in &state.chat_messages {
        let sender_color = if sender == "You" {
            GOLD_BADGE
        } else {
            CYAN_NEON
        };
        lines.push(Line::from(Span::styled(
            format!(" [{}]", sender),
            Style::default()
                .fg(sender_color)
                .add_modifier(Modifier::BOLD),
        )));
        for row in text.lines() {
            lines.push(Line::from(Span::styled(
                format!(" {}", row),
                Style::default().fg(Color::Rgb(226, 232, 240)),
            )));
        }
        lines.push(Line::from(""));
    }

    lines.push(Line::from(vec![
        Span::styled(
            " ▸ Prompt > ",
            Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{}█", state.prompt_input),
            Style::default().fg(Color::White),
        ),
    ]));

    let inner_width = inner.width.max(20) as usize;
    let mut wrapped_line_count = 0usize;
    for line in &lines {
        let char_len: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        wrapped_line_count += char_len.max(1).div_ceil(inner_width);
    }
    let visible_height = inner.height as usize;
    let auto_scroll = wrapped_line_count.saturating_sub(visible_height) as u16;
    let scroll_y = auto_scroll.saturating_sub(state.chat_scroll_offset);

    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll_y, 0)),
        inner,
    );
}

fn render_arc_hierarchy(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let border_col = if state.active_panel_index == 1 {
        CYAN_NEON
    } else {
        BORDER_SLATE
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_col).bg(BG_DARK))
        .title_top(Span::styled(
            "─ MEMORY & CACHE HIERARCHY (ARC ENGINE) ─",
            Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = vec![
        Line::from(vec![
            Span::styled(" L1 VRAM [GPU0+1] ", Style::default().fg(TEXT_MUTED)),
            Span::styled("▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱ ", Style::default().fg(CYAN_NEON)),
            Span::styled(
                format!(
                    "{:.1}/{:.0} GB",
                    state.header.vram_used_gb, state.header.vram_total_gb
                ),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled(" L2 Slab [Pinned] ", Style::default().fg(TEXT_MUTED)),
            Span::styled("▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱ ", Style::default().fg(INDIGO_GLOW)),
            Span::styled(
                format!("{:.1}/{:.0} GB", state.l2_used_gb, state.l2_total_gb),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled(" L3 NVMe [Stream] ", Style::default().fg(TEXT_MUTED)),
            Span::styled("▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱ ", Style::default().fg(GOLD_BADGE)),
            Span::styled(
                format!("{:.1} GB", state.l3_stream_gb),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Hit Ratio: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.1}%", state.header.hit_rate),
                Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" ── Miss Penalty: {:.1} ms", state.miss_penalty_ms),
                Style::default().fg(TEXT_MUTED),
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_nvme_smart(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let border_col = if state.active_panel_index == 2 {
        CYAN_NEON
    } else {
        BORDER_SLATE
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_col).bg(BG_DARK))
        .title_top(Span::styled(
            "─ NVMe I/O & S.M.A.R.T. [Samsung 990 PRO] ─",
            Style::default()
                .fg(INDIGO_GLOW)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines = vec![
        Line::from(vec![
            Span::styled(" Read BW: ", Style::default().fg(TEXT_MUTED)),
            Span::styled("▁▂▃▅▇██▆▄▂▁▁▃▆█▇▄▂▁  ", Style::default().fg(CYAN_NEON)),
            Span::styled(
                format!("{:.2} GB/s", state.nvme_read_gbs),
                Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled(" I/O QD:  ", Style::default().fg(TEXT_MUTED)),
            Span::styled("▁▁▂▃▄▆▄▂▁▁▁▁▂▄▆▅▃▁▁  ", Style::default().fg(INDIGO_GLOW)),
            Span::styled(
                format!("QD: {:>2} / {}", state.queue_depth, state.max_queue_depth),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Thermal: ", Style::default().fg(TEXT_MUTED)),
            Span::styled("[███████████░░░░░░░] ", Style::default().fg(EMERALD_OK)),
            Span::styled(
                format!("{}°C (Nominal)", state.header.nvme_temp_c),
                Style::default().fg(EMERALD_OK),
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

/// Encodes 8 expert activation intensities into a single Unicode Braille character (`U+2800..=U+28FF`)
pub fn encode_braille_cell(intensities: &[u16], threshold: u16) -> char {
    // Standard 2x4 Braille dot bit positions:
    // (0,0)->0x01, (0,1)->0x02, (0,2)->0x04, (1,0)->0x08, (1,1)->0x10, (1,2)->0x20, (0,3)->0x40, (1,3)->0x80
    const DOT_BITS: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];
    let mut mask = 0u8;
    for (i, &bit) in DOT_BITS.iter().enumerate() {
        let val = intensities.get(i).copied().unwrap_or(0);
        if val >= threshold {
            mask |= bit;
        }
    }
    char::from_u32(0x2800 + mask as u32).unwrap_or('⠀')
}

fn render_braille_heatmap(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let border_col = if state.active_panel_index == 3 {
        CYAN_NEON
    } else {
        BORDER_SLATE
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_col).bg(BG_DARK))
        .title_top(Span::styled(
            format!(
                "─ EXPERT ACTIVATION HEATMAP (21,504 / LAYER {:02}) ─",
                state.active_layer
            ),
            Style::default().fg(GOLD_BADGE).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let active_count = state
        .expert_heatmap
        .iter()
        .filter(|&&v| v > 0)
        .count()
        .max(8);

    // Dynamically encode 3 rows of 14 Braille characters (each char = 8 experts) from `state.expert_heatmap`
    let mut rows_braille = [
        String::with_capacity(18),
        String::with_capacity(18),
        String::with_capacity(18),
    ];
    for (r_idx, row_str) in rows_braille.iter_mut().enumerate() {
        row_str.push(' ');
        for c_idx in 0..14 {
            let base = ((r_idx * 14 + c_idx) * 8) % state.expert_heatmap.len().max(8);
            let mut slice = [0u16; 8];
            for (k, slot_val) in slice.iter_mut().enumerate() {
                let raw_val = state.expert_heatmap[(base + k) % state.expert_heatmap.len()];
                // Combine real expert hits with a subtle travelling wave across layers
                let wave_hit = u16::from((base + k + state.tick as usize).is_multiple_of(7));
                *slot_val = raw_val.saturating_add(wave_hit);
            }
            row_str.push(encode_braille_cell(&slice, 1));
        }
        row_str.push(' ');
    }

    let lines = vec![
        Line::from(vec![
            Span::styled(rows_braille[0].clone(), Style::default().fg(CYAN_NEON)),
            Span::styled(
                "Mode: Code/Rust [Cluster #07]",
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled(rows_braille[1].clone(), Style::default().fg(INDIGO_GLOW)),
            Span::styled(
                format!("Active: {} / 288 per layer", active_count),
                Style::default().fg(TEXT_MUTED),
            ),
        ]),
        Line::from(vec![
            Span::styled(rows_braille[2].clone(), Style::default().fg(CYAN_NEON)),
            Span::styled("Look-ahead (L+2): Ready", Style::default().fg(EMERALD_OK)),
        ]),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_ticket_queue(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let border_col = if state.active_panel_index == 4 {
        CYAN_NEON
    } else {
        BORDER_SLATE
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_col).bg(BG_DARK))
        .title_top(Span::styled(
            "─ TICKET QUEUE (I/O-AWARE CONTINUOUS BATCHING SCHEDULER) ─",
            Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = Vec::with_capacity(state.live_tickets.len());
    for (badge, summary, is_active) in &state.live_tickets {
        let icon = if *is_active { " ▸ " } else { " ▹ " };
        let badge_style = if *is_active {
            Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(GOLD_BADGE)
        };
        let text_style = if *is_active {
            Style::default().fg(Color::White)
        } else {
            Style::default().fg(TEXT_MUTED)
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{icon}{badge:<8} "), badge_style),
            Span::styled(summary.clone(), text_style),
        ]));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}
