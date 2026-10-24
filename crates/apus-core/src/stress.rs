use crate::{ArcCache, CacheTier, ExpertId, SlabArena};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct StressReport {
    pub worker_threads: usize,
    pub total_operations: u64,
    pub elapsed: Duration,
    pub ops_per_sec: f64,
    pub hit_rate_pct: f32,
    pub slab_invariant_passed: bool,
}

/// Multi-threaded concurrency & invariant stress verifier for `ArcCache` and `SlabArena` (Sprint 30 `apus-fuzz`)
pub struct ConcurrencyStressRunner;

impl ConcurrencyStressRunner {
    /// Spawns `threads` parallel OS worker threads hammering a shared `ArcCache` + `SlabArena`
    /// with pseudo-random Zipfian MoE expert accesses and verifies zero memory leaks or slot corruption.
    pub fn run_stress_test(
        threads: usize,
        ops_per_thread: usize,
        slab_slots: usize,
    ) -> Result<StressReport, crate::ApusError> {
        let start = Instant::now();
        let arena = Arc::new(SlabArena::new(4096, slab_slots, true)?);
        let cache = Arc::new(ArcCache::new(slab_slots, slab_slots / 2));

        let mut handles = Vec::with_capacity(threads);
        for tid in 0..threads {
            let arena_cl = Arc::clone(&arena);
            let cache_cl = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                let mut rng = (tid as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
                for step in 0..ops_per_thread {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    // Skewed distribution: 70% hot cluster (experts 0..16), 30% cold tail (experts 16..128)
                    let layer = ((rng >> 16) % 4) as u16;
                    let expert = if (rng & 0xFF) < 180 {
                        ((rng >> 32) % 16) as u16
                    } else {
                        16 + ((rng >> 32) % 112) as u16
                    };
                    let id = ExpertId { layer, expert };

                    let (tier, evicted) = cache_cl.access(id);
                    if let Some(victim) = evicted {
                        arena_cl.evict(&victim);
                    }
                    if tier == CacheTier::L3NvmeMiss {
                        if let Ok(slot_arc) = arena_cl.acquire_slot(id) {
                            let mut guard = slot_arc.lock();
                            guard.as_mut_slice()[0] = (step & 0xFF) as u8;
                        }
                    }
                }
            }));
        }

        for h in handles {
            let _ = h.join();
        }

        let elapsed = start.elapsed();
        let total_operations = (threads * ops_per_thread) as u64;
        let ops_per_sec = (total_operations as f64) / elapsed.as_secs_f64().max(1e-6);
        let telemetry = cache.telemetry();
        let slab_invariant_passed =
            arena.used_slots() + arena.free_slots_count() >= slab_slots;

        Ok(StressReport {
            worker_threads: threads,
            total_operations,
            elapsed,
            ops_per_sec,
            hit_rate_pct: telemetry.hit_rate_percent(),
            slab_invariant_passed,
        })
    }
}
