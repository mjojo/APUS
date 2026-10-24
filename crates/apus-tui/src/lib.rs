pub mod dashboard;
pub mod header;
pub mod splash;

pub use dashboard::{DashboardState, buffer_to_plain_text, render_dashboard};
pub use header::{TelemetryHeaderState, render_header};
pub use splash::{
    BootStage, SplashScreenState, build_gradient_bar, centered_rect, lerp_rgb, render_splash,
    run_boot_splash,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn test_splash_and_dashboard_render_without_panic() {
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut splash = SplashScreenState::default();
        for _ in 0..10 {
            splash.on_tick();
        }
        terminal.draw(|f| render_splash(f, &splash)).unwrap();

        let mut dash = DashboardState {
            prompt_input: "Test prompt".into(),
            ..Default::default()
        };
        dash.submit_prompt();
        dash.toggle_nvme_cap();
        for _ in 0..25 {
            dash.on_tick();
        }
        terminal.draw(|f| render_dashboard(f, &dash)).unwrap();
        let text = buffer_to_plain_text(terminal.backend().buffer());
        assert!(text.contains("APUS // ULTRA-STREAMING MoE"));
        assert!(text.contains("mjojo"));
        assert!(text.contains("Test prompt"));
        assert!(text.contains("NVMe Cap (100%)"));
    }
}
