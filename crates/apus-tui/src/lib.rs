pub mod dashboard;
pub mod header;
pub mod splash;

pub use dashboard::{render_dashboard, DashboardState};
pub use header::{render_header, TelemetryHeaderState};
pub use splash::{
    build_gradient_bar, centered_rect, lerp_rgb, render_splash, run_boot_splash, BootStage,
    SplashScreenState,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    #[test]
    fn test_splash_and_dashboard_render_without_panic() {
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut splash = SplashScreenState::default();
        for _ in 0..10 {
            splash.on_tick();
        }
        terminal.draw(|f| render_splash(f, &splash)).unwrap();

        let mut dash = DashboardState::default();
        dash.on_tick();
        terminal.draw(|f| render_dashboard(f, &dash)).unwrap();
    }
}
