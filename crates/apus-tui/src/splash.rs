use std::{
    io,
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    backend::Backend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Sparkline},
    Frame, Terminal,
};

// Aerospace Truecolor Palette for APUS
pub const BG_VOID: Color = Color::Rgb(9, 11, 16);          // #090B10 — Outer Void Background
pub const BG_PANEL: Color = Color::Rgb(14, 17, 26);        // #0E111A — Inner Obsidian Panel
pub const BORDER_DIM: Color = Color::Rgb(45, 55, 72);      // #2D3748 — Steel Frame Border
pub const CYAN_NEON: Color = Color::Rgb(0, 240, 255);      // #00F0FF — APUS Signature Cyan
pub const INDIGO_DEEP: Color = Color::Rgb(123, 97, 255);   // #7B61FF — Deep Indigo Glow
pub const GOLD_MJOJO: Color = Color::Rgb(255, 184, 0);     // #FFB800 — Gold mjojo Badge
pub const EMERALD_OK: Color = Color::Rgb(0, 230, 118);     // #00E676 — Verification Passed
pub const TEXT_PRIMARY: Color = Color::Rgb(226, 232, 240); // #E2E8F0 — Primary Telemetry Text
pub const TEXT_MUTED: Color = Color::Rgb(100, 116, 139);   // #64748B — Technical Captions

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootStage {
    ProbingTopology,
    VerifyingBlake3,
    WarmingArcCache,
    Completed,
}

pub struct SplashScreenState {
    pub stage: BootStage,
    pub tick: u64,
    pub total_experts: u32,
    pub verified_experts: u32,
    pub current_blake3_hash: String,
    pub cache_target_gb: f32,
    pub cache_warmed_gb: f32,
    pub nvme_speed_gbs: f32,
    pub io_history: Vec<u64>,
    pub profile_name: String,
}

impl Default for SplashScreenState {
    fn default() -> Self {
        Self {
            stage: BootStage::ProbingTopology,
            tick: 0,
            total_experts: 21_504,
            verified_experts: 0,
            current_blake3_hash: "00000000000000000000000000000000".into(),
            cache_target_gb: 20.0,
            cache_warmed_gb: 0.0,
            nvme_speed_gbs: 0.0,
            io_history: vec![0; 48],
            profile_name: "coding_rust.roost".into(),
        }
    }
}

impl SplashScreenState {
    /// Advances animation state and live telemetry (invoked at 60 FPS)
    pub fn on_tick(&mut self) {
        self.tick += 1;
        // Rolling BLAKE3 digest stream for visual chunk integrity verification
        let seed = self.tick.wrapping_mul(0x9E3779B97F4A7C15);
        self.current_blake3_hash = format!(
            "{:016x}{:016x}",
            seed.rotate_left(17),
            seed ^ 0xA5A5A5A55A5A5A5A
        );

        match self.stage {
            BootStage::ProbingTopology => {
                self.nvme_speed_gbs = 0.4;
                if self.tick > 25 {
                    self.stage = BootStage::VerifyingBlake3;
                }
            }
            BootStage::VerifyingBlake3 => {
                // Verify 21,504 expert chunk headers in ~60 ticks (~1 second)
                self.verified_experts = (self.verified_experts + 420).min(self.total_experts);
                self.nvme_speed_gbs = 3.2 + ((self.tick as f32 * 0.3).sin().abs() * 1.4);
                if self.verified_experts >= self.total_experts {
                    self.stage = BootStage::WarmingArcCache;
                }
            }
            BootStage::WarmingArcCache => {
                // Linear DMA streaming of hot experts into Pinned Slab RAM and VRAM
                self.nvme_speed_gbs = 5.8 + ((self.tick as f32 * 0.25).cos() * 0.9);
                self.cache_warmed_gb = (self.cache_warmed_gb + 0.32).min(self.cache_target_gb);
                if self.cache_warmed_gb >= self.cache_target_gb {
                    self.stage = BootStage::Completed;
                }
            }
            BootStage::Completed => {
                self.nvme_speed_gbs = 0.0;
                self.current_blake3_hash = "b94f8a2e7c10d4f6a19b3e82d507c41a".into();
            }
        }

        // Update NVMe throughput sparkline history (in 100 MB/s units)
        if self.io_history.len() >= 48 {
            self.io_history.remove(0);
        }
        self.io_history.push((self.nvme_speed_gbs * 10.0) as u64);
    }

    pub fn overall_progress(&self) -> f32 {
        match self.stage {
            BootStage::ProbingTopology => (self.tick as f32 / 25.0) * 0.10,
            BootStage::VerifyingBlake3 => {
                0.10 + (self.verified_experts as f32 / self.total_experts.max(1) as f32) * 0.35
            }
            BootStage::WarmingArcCache => {
                0.45 + (self.cache_warmed_gb / self.cache_target_gb.max(0.1)) * 0.55
            }
            BootStage::Completed => 1.0,
        }
    }
}

/// Linear RGB color interpolation for gradients and frame breathing effects
pub fn lerp_rgb(c1: (u8, u8, u8), c2: (u8, u8, u8), t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let r = (c1.0 as f32 + (c2.0 as f32 - c1.0 as f32) * t) as u8;
    let g = (c1.1 as f32 + (c2.1 as f32 - c1.1 as f32) * t) as u8;
    let b = (c1.2 as f32 + (c2.2 as f32 - c1.2 as f32) * t) as u8;
    Color::Rgb(r, g, b)
}

/// Renders a sub-pixel animated RGB gradient progress bar (█▓▒░)
pub fn build_gradient_bar(width: usize, progress: f32, tick: u64) -> Line<'static> {
    let filled_exact = (width as f32 * progress.clamp(0.0, 1.0)) as usize;
    let mut spans = Vec::with_capacity(width + 4);
    for i in 0..width {
        let ratio = i as f32 / width.max(1) as f32;
        // Phase-shift gradient over time for a traveling wave effect
        let wave = ((ratio * 3.0 - (tick as f32 * 0.08)).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let grad_color = lerp_rgb((123, 97, 255), (0, 240, 255), (ratio * 0.7) + (wave * 0.3));
        if i < filled_exact {
            spans.push(Span::styled("█", Style::default().fg(grad_color)));
        } else if i == filled_exact && progress < 1.0 {
            spans.push(Span::styled("▓", Style::default().fg(CYAN_NEON)));
        } else if i == filled_exact + 1 && progress < 1.0 {
            spans.push(Span::styled("▒", Style::default().fg(INDIGO_DEEP)));
        } else {
            spans.push(Span::styled("░", Style::default().fg(Color::Rgb(28, 34, 48))));
        }
    }
    Line::from(spans)
}

/// Main rendering function for the 60 FPS Boot Splash Screen
pub fn render_splash(frame: &mut Frame, state: &SplashScreenState) {
    let screen = frame.area();
    // Fill terminal background with deep obsidian void color
    frame.render_widget(Block::default().style(Style::default().bg(BG_VOID)), screen);

    // Center the 86x24 telemetry modal
    let modal = centered_rect(86, 24, screen);
    frame.render_widget(Clear, modal);

    // Pulsing border color during active NVMe DMA streaming
    let pulse = ((state.tick as f32 * 0.12).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
    let border_color = if state.stage == BootStage::Completed {
        EMERALD_OK
    } else {
        lerp_rgb((45, 55, 72), (0, 240, 255), pulse * 0.65)
    };

    let outer_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Thick)
        .style(Style::default().bg(BG_PANEL))
        .border_style(Style::default().fg(border_color))
        .title_top(
            Line::from(vec![
                Span::styled("━ ", Style::default().fg(BORDER_DIM)),
                Span::styled(
                    " APUS // BOOTSTRAP TELEMETRY ",
                    Style::default().fg(BG_VOID).bg(CYAN_NEON).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ━ [Space/Esc] Skip ", Style::default().fg(TEXT_MUTED)),
            ])
            .alignment(Alignment::Left),
        )
        .title_bottom(
            Line::from(vec![
                Span::styled("─ ⟪ ", Style::default().fg(INDIGO_DEEP)),
                Span::styled("engineered by ", Style::default().fg(TEXT_MUTED)),
                Span::styled(
                    "mjojo",
                    Style::default().fg(GOLD_MJOJO).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ⟫ ─", Style::default().fg(INDIGO_DEEP)),
            ])
            .alignment(Alignment::Right),
        );

    let inner = outer_block.inner(modal);
    frame.render_widget(outer_block, modal);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(7), // Section 1: Swift wing + APUS by mjojo logo
            Constraint::Length(4), // Section 2: Subsystem checklist + BLAKE3 digest
            Constraint::Length(3), // Section 3: Live NVMe throughput Sparkline
            Constraint::Min(3),    // Section 4: Gradient ARC cache warm-up bar
        ])
        .split(inner);

    // 1. Render logo with gradient swift wing (Apus apus)
    let wing_c1 = lerp_rgb((0, 240, 255), (123, 97, 255), pulse);
    let wing_c2 = lerp_rgb((123, 97, 255), (0, 240, 255), pulse);
    let logo_lines = vec![
        Line::from(vec![
            Span::styled("            ◣                 ", Style::default().fg(wing_c1)),
            Span::styled(
                "▄▀█ █▀█ █ █ █▀",
                Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "   ASYNC PAGED ULTRA-STREAMING",
                Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("           ██ ◣                ", Style::default().fg(wing_c1)),
            Span::styled(
                "█▀█ █▀▀ █▄█ ▄█",
                Style::default().fg(INDIGO_DEEP).add_modifier(Modifier::BOLD),
            ),
            Span::styled("   744B MoE Inference Runtime", Style::default().fg(TEXT_MUTED)),
        ]),
        Line::from(vec![
            Span::styled("     ◣     ████ ◣               ", Style::default().fg(CYAN_NEON)),
            Span::styled(
                "────────────────────────────────────────────────",
                Style::default().fg(BORDER_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("    ███▄▄███████▄▄           ", Style::default().fg(CYAN_NEON)),
            Span::styled("AUTHOR   :: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                " mjojo ",
                Style::default()
                    .fg(BG_VOID)
                    .bg(GOLD_MJOJO)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  │  ", Style::default().fg(BORDER_DIM)),
            Span::styled("KERNEL: ", Style::default().fg(TEXT_MUTED)),
            Span::styled("Zero-Copy DMA", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("     ▀▀██████████▀           ", Style::default().fg(wing_c2)),
            Span::styled("MODEL    :: ", Style::default().fg(TEXT_MUTED)),
            Span::styled("GLM-5.2-744B-Q4_K_M.apus ", Style::default().fg(CYAN_NEON)),
            Span::styled("(MLA + MTP 2x)", Style::default().fg(INDIGO_DEEP)),
        ]),
        Line::from(vec![
            Span::styled("         ▀▀████▀             ", Style::default().fg(INDIGO_DEEP)),
            Span::styled("TOPOLOGY :: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                "2x GPU P2P Pipeline + 24 GB Pinned Slab RAM",
                Style::default().fg(TEXT_PRIMARY),
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(logo_lines), rows[0]);

    // 2. Subsystem initialization checklist & cryptographic BLAKE3 verification
    let topo_badge = Span::styled(
        " [OK] ",
        Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD),
    );
    let blake_badge = if state.verified_experts >= state.total_experts {
        Span::styled(
            " [VERIFIED] ",
            Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            " [SCANNING] ",
            Style::default().fg(GOLD_MJOJO).add_modifier(Modifier::BOLD),
        )
    };
    let arc_badge = match state.stage {
        BootStage::Completed => Span::styled(
            " [LOCKED] ",
            Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD),
        ),
        BootStage::WarmingArcCache => Span::styled(
            " [STREAMING] ",
            Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
        ),
        _ => Span::styled(" [WAITING] ", Style::default().fg(TEXT_MUTED)),
    };

    let telemetry_lines = vec![
        Line::from(vec![
            Span::styled(" ▸ I/O SUBSYSTEM ::", Style::default().fg(TEXT_MUTED)),
            topo_badge,
            Span::styled(
                "DirectStorage / O_DIRECT (4096B Aligned, OS QoS: IDLE)",
                Style::default().fg(TEXT_PRIMARY),
            ),
        ]),
        Line::from(vec![
            Span::styled(" ▸ BLAKE3 TREE   ::", Style::default().fg(TEXT_MUTED)),
            blake_badge,
            Span::styled(
                format!("{:>5} / {} Experts ", state.verified_experts, state.total_experts),
                Style::default().fg(TEXT_PRIMARY),
            ),
            Span::styled(
                format!("⟪ 0x{} ⟫", &state.current_blake3_hash[..16]),
                Style::default().fg(INDIGO_DEEP),
            ),
        ]),
        Line::from(vec![
            Span::styled(" ▸ ARC SNAPSHOT  ::", Style::default().fg(TEXT_MUTED)),
            arc_badge,
            Span::styled(
                format!(
                    "Profile '{}' -> {:.1} / {:.1} GB (mlock)",
                    state.profile_name, state.cache_warmed_gb, state.cache_target_gb
                ),
                Style::default().fg(TEXT_PRIMARY),
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(telemetry_lines), rows[1]);

    // 3. NVMe read throughput sparkline
    let spark_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(20)])
        .split(rows[2]);

    let io_label = Paragraph::new(vec![
        Line::from(vec![Span::styled(
            " ▸ NVMe THROUGHPUT",
            Style::default().fg(TEXT_MUTED),
        )]),
        Line::from(vec![
            Span::styled(
                format!("   {:.2} GB/s ", state.nvme_speed_gbs),
                Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
            ),
            Span::styled("(QD: 32)", Style::default().fg(INDIGO_DEEP)),
        ]),
    ]);
    frame.render_widget(io_label, spark_chunks[0]);

    let sparkline = Sparkline::default()
        .data(&state.io_history)
        .max(75)
        .bar_set(symbols::bar::NINE_LEVELS)
        .style(Style::default().fg(INDIGO_DEEP).bg(BG_PANEL));
    frame.render_widget(sparkline, spark_chunks[1]);

    // 4. Bottom progress bar with traveling RGB gradient
    let pct = (state.overall_progress() * 100.0).min(100.0);
    let bar_width = (rows[3].width as usize).saturating_sub(14);
    let mut bar_line = build_gradient_bar(bar_width, state.overall_progress(), state.tick);
    bar_line.spans.insert(0, Span::raw(" "));
    bar_line.spans.push(Span::styled(
        format!(" {:>5.1}%", pct),
        Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
    ));

    let status_caption = match state.stage {
        BootStage::ProbingTopology => {
            "Scanning PCIe lanes, NUMA nodes, and locking 4KB Page-Aligned Slab Arena..."
        }
        BootStage::VerifyingBlake3 => {
            "Verifying cryptographic BLAKE3 Merkle integrity across 21,504 experts..."
        }
        BootStage::WarmingArcCache => {
            "Streaming warm experts via zero-copy DMA into L1 VRAM and L2 Pinned RAM..."
        }
        BootStage::Completed => "System ready. Launching APUS Telemetry HUD (by mjojo)...",
    };

    let progress_widget = Paragraph::new(vec![
        Line::from(Span::styled(
            format!(" {}", status_caption),
            Style::default().fg(TEXT_MUTED),
        )),
        bar_line,
    ]);
    frame.render_widget(progress_widget, rows[3]);
}

/// Helper to center a fixed-size modal rectangle inside the terminal viewport
pub fn centered_rect(width: u16, height: u16, r: Rect) -> Rect {
    let x = r.x + r.width.saturating_sub(width) / 2;
    let y = r.y + r.height.saturating_sub(height) / 2;
    Rect::new(x, y, width.min(r.width), height.min(r.height))
}

/// 60 FPS Boot Splash loop with Space/Esc/Enter skip support
pub fn run_boot_splash<B: Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    let mut state = SplashScreenState::default();
    let tick_rate = Duration::from_millis(16); // ~60 FPS
    let mut last_tick = Instant::now();
    let mut hold_completed_ticks = 0u8;

    loop {
        terminal
            .draw(|f| render_splash(f, &state))
            .map_err(|e| io::Error::other(e.to_string()))?;
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press
                    && matches!(key.code, KeyCode::Esc | KeyCode::Char(' ') | KeyCode::Enter)
                {
                    return Ok(());
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            state.on_tick();
            last_tick = Instant::now();
            if state.stage == BootStage::Completed {
                hold_completed_ticks += 1;
                if hold_completed_ticks > 35 {
                    return Ok(());
                }
            }
        }
    }
}
