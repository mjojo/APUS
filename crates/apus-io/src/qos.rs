use parking_lot::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoPriorityClass {
    /// Background I/O priority (IOPRIO_CLASS_IDLE on Linux / IoPriorityHintVeryLow on Windows)
    IdleBackground,
    Normal,
    HighDedicatedNvme,
}

#[derive(Debug, Clone)]
pub struct QosMetrics {
    pub os_drive_mode: bool,
    pub priority: IoPriorityClass,
    pub bandwidth_cap_gbs: f32,
    pub current_throughput_gbs: f32,
    pub active_queue_depth: usize,
    pub max_queue_depth: usize,
    pub throttled_events: u64,
}

struct QosInternal {
    bytes_in_window: u64,
    window_start: Instant,
    current_throughput_gbs: f32,
    active_qd: usize,
    throttled_events: u64,
}

/// System Drive Quality-of-Service (OS Drive QoS) Controller.
/// Caps peak NVMe utilization (e.g., to 5.5 GB/s out of 7.0 GB/s),
/// reserving at least 20% of bandwidth and command queue depth (QD) for the host OS kernel.
pub struct OsDriveQosController {
    os_drive_mode: bool,
    priority: IoPriorityClass,
    bandwidth_cap_gbs: f32,
    max_queue_depth: usize,
    state: Mutex<QosInternal>,
}

impl OsDriveQosController {
    pub fn new(
        os_drive_mode: bool,
        priority: IoPriorityClass,
        bandwidth_cap_gbs: f32,
        max_queue_depth: usize,
    ) -> Self {
        Self {
            os_drive_mode,
            priority,
            bandwidth_cap_gbs: bandwidth_cap_gbs.max(0.5),
            max_queue_depth: max_queue_depth.clamp(1, 128),
            state: Mutex::new(QosInternal {
                bytes_in_window: 0,
                window_start: Instant::now(),
                current_throughput_gbs: 0.0,
                active_qd: 0,
                throttled_events: 0,
            }),
        }
    }

    /// Registers a read operation of `bytes` and computes a micro-sleep pacing delay
    /// when throughput exceeds the OS drive bandwidth cap to prevent system freezes.
    pub fn register_read_and_compute_delay(&self, bytes: usize) -> Duration {
        let mut st = self.state.lock();
        let elapsed = st.window_start.elapsed().as_secs_f32();
        if elapsed >= 0.10 {
            st.current_throughput_gbs = (st.bytes_in_window as f32 / 1e9) / elapsed;
            st.bytes_in_window = 0;
            st.window_start = Instant::now();
        }
        st.bytes_in_window += bytes as u64;
        st.active_qd = (st.active_qd + 1).min(self.max_queue_depth);

        if self.os_drive_mode && st.current_throughput_gbs > self.bandwidth_cap_gbs {
            st.throttled_events += 1;
            let excess_ratio =
                (st.current_throughput_gbs - self.bandwidth_cap_gbs) / self.bandwidth_cap_gbs;
            let micros = ((excess_ratio * 250.0) as u64).clamp(20, 2_000);
            Duration::from_micros(micros)
        } else {
            Duration::ZERO
        }
    }

    pub fn complete_read(&self) {
        let mut st = self.state.lock();
        st.active_qd = st.active_qd.saturating_sub(1);
    }

    pub fn metrics(&self) -> QosMetrics {
        let st = self.state.lock();
        QosMetrics {
            os_drive_mode: self.os_drive_mode,
            priority: self.priority,
            bandwidth_cap_gbs: self.bandwidth_cap_gbs,
            current_throughput_gbs: st.current_throughput_gbs,
            active_queue_depth: st.active_qd,
            max_queue_depth: self.max_queue_depth,
            throttled_events: st.throttled_events,
        }
    }
}
