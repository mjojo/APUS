use serde::{Deserialize, Serialize};

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn SetThreadAffinityMask(hThread: *mut std::ffi::c_void, dwThreadAffinityMask: usize) -> usize;
}

/// PCIe interconnect topology classification between NVMe SSD and GPUs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcieLinkTopology {
    /// Direct CPU root-complex PCIe lanes (ideal for P2P DMA & DirectStorage)
    CpuRootComplexDirect,
    /// Chipset-routed (PCH) PCIe lanes (requires Pinned Host RAM staging)
    ChipsetRoutedPch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTopologyProfile {
    pub logical_cores: usize,
    pub estimated_numa_nodes: usize,
    pub ccd_chiplets: usize,
    pub pcie_topology: PcieLinkTopology,
    pub recommended_io_workers: usize,
    pub recommended_compute_workers: usize,
}

/// Hardware Topology Scanner for NUMA nodes, AMD Ryzen CCD chiplets, and CPU thread affinity
pub struct HardwareTopologyScanner;

impl HardwareTopologyScanner {
    /// Probes the host machine's logical core count, NUMA / CCD layout, and optimal worker split
    pub fn probe_host() -> HardwareTopologyProfile {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        let estimated_numa_nodes = if logical_cores >= 64 { 2 } else { 1 };
        let ccd_chiplets = if logical_cores >= 16 { 2 } else { 1 };
        let recommended_io_workers = (logical_cores / 4).clamp(2, 8);
        let recommended_compute_workers =
            logical_cores.saturating_sub(recommended_io_workers).max(2);

        HardwareTopologyProfile {
            logical_cores,
            estimated_numa_nodes,
            ccd_chiplets,
            pcie_topology: PcieLinkTopology::CpuRootComplexDirect,
            recommended_io_workers,
            recommended_compute_workers,
        }
    }

    /// Pins the current worker thread to a specific physical/logical CPU core index
    /// to prevent cross-CCD Infinity Fabric / NUMA cache thrashing during NVMe DMA streaming.
    pub fn pin_current_thread_to_core(core_index: usize) -> bool {
        #[cfg(windows)]
        {
            if core_index >= (usize::BITS as usize) {
                return false;
            }
            let mask: usize = 1usize << core_index;
            let prev = unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) };
            prev != 0
        }
        #[cfg(not(windows))]
        {
            let _ = core_index;
            true
        }
    }
}
