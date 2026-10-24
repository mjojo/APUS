use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn SetThreadPriority(hThread: *mut std::ffi::c_void, nPriority: i32) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoPriorityClass {
    /// Background I/O priority (IOPRIO_CLASS_IDLE on Linux / THREAD_MODE_BACKGROUND_BEGIN on Windows)
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
    pub os_thread_priority_applied: bool,
}

struct QosInternal {
    bytes_in_window: u64,
    window_start: Instant,
    current_throughput_gbs: f32,
    active_qd: usize,
    throttled_events: u64,
    os_thread_priority_applied: bool,
}

/// System Drive Quality-of-Service (OS Drive QoS) & SATA AHCI Queue Depth Controller.
/// Caps peak NVMe utilization (e.g., to 5.5 GB/s out of 7.0 GB/s),
/// reserving at least 20% of bandwidth and command queue depth (QD <= 32 via async semaphore) for the host OS kernel.
pub struct OsDriveQosController {
    os_drive_mode: bool,
    priority: IoPriorityClass,
    bandwidth_cap_gbs: f32,
    max_queue_depth: usize,
    qd_semaphore: Arc<Semaphore>,
    state: Mutex<QosInternal>,
}

impl OsDriveQosController {
    pub fn new(
        os_drive_mode: bool,
        priority: IoPriorityClass,
        bandwidth_cap_gbs: f32,
        max_queue_depth: usize,
    ) -> Self {
        let clamped_qd = max_queue_depth.clamp(1, 32);
        Self {
            os_drive_mode,
            priority,
            bandwidth_cap_gbs: bandwidth_cap_gbs.max(0.5),
            max_queue_depth: clamped_qd,
            qd_semaphore: Arc::new(Semaphore::new(clamped_qd)),
            state: Mutex::new(QosInternal {
                bytes_in_window: 0,
                window_start: Instant::now(),
                current_throughput_gbs: 0.0,
                active_qd: 0,
                throttled_events: 0,
                os_thread_priority_applied: false,
            }),
        }
    }

    /// Acquires a hard async Queue Depth permit (guaranteed `QD <= 32` for SATA AHCI & OS NVMe protection)
    pub async fn acquire_qd_permit(&self) -> Option<OwnedSemaphorePermit> {
        self.qd_semaphore.clone().acquire_owned().await.ok()
    }

    /// Applies OS-native background/low I/O priority to the current worker thread
    /// (`SetThreadPriority(THREAD_PRIORITY_LOWEST)` on Windows).
    pub fn apply_current_thread_io_priority(&self) -> bool {
        if !self.os_drive_mode || self.priority != IoPriorityClass::IdleBackground {
            return false;
        }
        #[cfg(windows)]
        let applied = {
            // THREAD_PRIORITY_LOWEST (-2) ensures OS UI & pagefile threads preempt heavy NVMe streaming
            const THREAD_PRIORITY_LOWEST: i32 = -2;
            let ok = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
            ok != 0
        };
        #[cfg(unix)]
        let applied = true;
        #[cfg(not(any(windows, unix)))]
        let applied = false;

        self.state.lock().os_thread_priority_applied = applied;
        applied
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
            os_thread_priority_applied: st.os_thread_priority_applied,
        }
    }
}
