use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
};

// Premium Truecolor Palette (Cyberpunk / Aerospace HUD)
pub const BG_DARK: Color = Color::Rgb(13, 15, 23); // #0D0F17 — Deep Obsidian
pub const BORDER_SLATE: Color = Color::Rgb(56, 65, 82); // #384152 — Muted Steel
pub const CYAN_NEON: Color = Color::Rgb(0, 240, 255); // #00F0FF — APUS Neon Cyan Accent
pub const INDIGO_GLOW: Color = Color::Rgb(123, 97, 255); // #7B61FF — Secondary Indigo Gradient
pub const GOLD_BADGE: Color = Color::Rgb(255, 184, 0); // #FFB800 — mjojo Author Badge Accent
pub const EMERALD_OK: Color = Color::Rgb(0, 230, 118); // #00E676 — Nominal Telemetry Status
pub const TEXT_MUTED: Color = Color::Rgb(148, 163, 184); // #94A3B8 — Metric Labels

#[derive(Debug, Clone)]
pub struct TelemetryHeaderState {
    pub model_name: String,
    pub io_mode: String,
    pub tok_per_sec: f32,
    pub hit_rate: f32,
    pub nvme_temp_c: u8,
    pub vram_used_gb: f32,
    pub vram_total_gb: f32,
}

impl Default for TelemetryHeaderState {
    fn default() -> Self {
        Self {
            model_name: "GLM-5.2-744B-Q4_K_M.apus".into(),
            io_mode: "DirectStorage + O_DIRECT (OS IDLE)".into(),
            tok_per_sec: 6.42,
            hit_rate: 91.8,
            nvme_temp_c: 54,
            vram_used_gb: 14.2,
            vram_total_gb: 16.0,
        }
    }
}

pub fn render_header(frame: &mut Frame, area: Rect, state: &TelemetryHeaderState) {
    // Outer thick-bordered frame with inverse header status pills
    let outer_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Thick)
        .border_style(Style::default().fg(BORDER_SLATE).bg(BG_DARK))
        .title_top(
            Line::from(vec![
                Span::styled("━ ", Style::default().fg(BORDER_SLATE)),
                Span::styled(
                    " APUS // ULTRA-STREAMING MoE ",
                    Style::default()
                        .fg(BG_DARK)
                        .bg(CYAN_NEON)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ━ ", Style::default().fg(BORDER_SLATE)),
                Span::styled(
                    format!(" {} ", state.model_name),
                    Style::default()
                        .fg(CYAN_NEON)
                        .bg(Color::Rgb(26, 32, 48))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ━", Style::default().fg(BORDER_SLATE)),
            ])
            .alignment(Alignment::Left),
        )
        .title_top(
            Line::from(vec![
                Span::styled(" ⟪ ", Style::default().fg(INDIGO_GLOW)),
                Span::styled("engineered by ", Style::default().fg(TEXT_MUTED)),
                Span::styled(
                    "mjojo",
                    Style::default().fg(GOLD_BADGE).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ⟫ ", Style::default().fg(INDIGO_GLOW)),
            ])
            .alignment(Alignment::Right),
        );

    let inner_area = outer_block.inner(area);
    frame.render_widget(outer_block, area);

    // Split inner header into 3 columns: Swift Wing Logo | Stream & Cache | I/O & Memory
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(31),
            Constraint::Percentage(46),
            Constraint::Percentage(54),
        ])
        .split(inner_area);

    // 1. Compact two-line emblem with the swift wing glyph
    let logo = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" ◥ ██ ◣  ", Style::default().fg(INDIGO_GLOW)),
            Span::styled(
                "▄▀█ █▀█ █ █ █▀",
                Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ◥ █▀ ", Style::default().fg(CYAN_NEON)),
            Span::styled(
                "█▀█ █▀▀ █▄█ ▄█",
                Style::default()
                    .fg(INDIGO_GLOW)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" by mjojo", Style::default().fg(GOLD_BADGE)),
        ]),
    ]);
    frame.render_widget(logo, chunks[0]);

    // 2. Center column: streaming throughput & ARC hit rate telemetry
    let stream_metrics = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("▸ STREAM   :: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.2} tok/s ", state.tok_per_sec),
                Style::default().fg(EMERALD_OK).add_modifier(Modifier::BOLD),
            ),
            Span::styled("[MTP 2x Active]", Style::default().fg(INDIGO_GLOW)),
        ]),
        Line::from(vec![
            Span::styled("▸ ARC HIT  :: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.1}% ", state.hit_rate),
                Style::default().fg(CYAN_NEON).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ▰▰▰▰▰▰▰▰▰▱ ", Style::default().fg(CYAN_NEON)),
        ]),
    ]);
    frame.render_widget(stream_metrics, chunks[1]);

    // 3. Right column: I/O subsystem & S.M.A.R.T. thermal status
    let temp_color = if state.nvme_temp_c < 65 {
        EMERALD_OK
    } else {
        GOLD_BADGE
    };
    let hw_metrics = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("▸ I/O PATH :: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                &state.io_mode,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("▸ NVMe QoS :: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{}°C ", state.nvme_temp_c),
                Style::default().fg(temp_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "│ VRAM: {:.1}/{:.0} GB",
                    state.vram_used_gb, state.vram_total_gb
                ),
                Style::default().fg(TEXT_MUTED),
            ),
        ]),
    ]);
    frame.render_widget(hw_metrics, chunks[2]);
}
