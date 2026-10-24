use crate::ExpertId;
use parking_lot::Mutex;
use std::collections::{HashSet, VecDeque};

/// Memory residency tier for an expert in the APUS cache hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    /// Hot cache resident in GPU memory (L1 VRAM)
    L1Vram,
    /// Pinned pool resident in system memory (L2 Pinned Slab RAM)
    L2SlabRam,
    /// Cache miss — requires streaming from NVMe storage (L3 SSD)
    L3NvmeMiss,
}

/// Telemetry counters for the 3-tier Adaptive Replacement Cache (ARC)
#[derive(Debug, Clone, Copy, Default)]
pub struct ArcTelemetry {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub ssd_misses: u64,
    pub t1_len: usize,
    pub t2_len: usize,
    pub l1_resident: usize,
    pub p_adaptation: usize,
}

impl ArcTelemetry {
    pub fn total_requests(&self) -> u64 {
        self.l1_hits + self.l2_hits + self.ssd_misses
    }

    pub fn hit_rate_percent(&self) -> f32 {
        let total = self.total_requests();
        if total == 0 {
            return 0.0;
        }
        ((self.l1_hits + self.l2_hits) as f32 / total as f32) * 100.0
    }

    pub fn l1_hit_percent(&self) -> f32 {
        let total = self.total_requests();
        if total == 0 {
            return 0.0;
        }
        (self.l1_hits as f32 / total as f32) * 100.0
    }

    pub fn l2_hit_percent(&self) -> f32 {
        let total = self.total_requests();
        if total == 0 {
            return 0.0;
        }
        (self.l2_hits as f32 / total as f32) * 100.0
    }

    pub fn ssd_miss_percent(&self) -> f32 {
        let total = self.total_requests();
        if total == 0 {
            return 0.0;
        }
        (self.ssd_misses as f32 / total as f32) * 100.0
    }
}

struct ArcState {
    capacity: usize,
    l1_capacity: usize,
    p: usize,
    /// T1: recently accessed experts (Recency)
    t1: VecDeque<ExpertId>,
    /// T2: frequently accessed ("hot") experts (Frequency)
    t2: VecDeque<ExpertId>,
    /// B1: ghost list of entries recently evicted from T1
    b1: VecDeque<ExpertId>,
    /// B2: ghost list of entries recently evicted from T2
    b2: VecDeque<ExpertId>,
    /// Set of experts promoted into L1 VRAM
    l1_set: HashSet<ExpertId>,
    l1_hits: u64,
    l2_hits: u64,
    ssd_misses: u64,
}

/// Adaptive Replacement Cache (ARC) manager spanning L1 VRAM and L2 Pinned RAM
pub struct ArcCache {
    state: Mutex<ArcState>,
}

impl ArcCache {
    pub fn new(total_ram_slots: usize, l1_vram_slots: usize) -> Self {
        let cap = total_ram_slots.max(2);
        Self {
            state: Mutex::new(ArcState {
                capacity: cap,
                l1_capacity: l1_vram_slots.min(cap),
                p: 0,
                t1: VecDeque::with_capacity(cap),
                t2: VecDeque::with_capacity(cap),
                b1: VecDeque::with_capacity(cap),
                b2: VecDeque::with_capacity(cap),
                l1_set: HashSet::with_capacity(l1_vram_slots),
                l1_hits: 0,
                l2_hits: 0,
                ssd_misses: 0,
            }),
        }
    }

    /// Accesses an expert. Returns the hit tier (L1, L2, or Miss) and an optional evicted victim.
    pub fn access(&self, id: ExpertId) -> (CacheTier, Option<ExpertId>) {
        let mut st = self.state.lock();

        // Case 1: Cache Hit in T1 or T2 (expert is already resident in RAM / VRAM)
        if let Some(pos) = st.t1.iter().position(|x| *x == id) {
            st.t1.remove(pos);
            st.t2.push_back(id);
            let tier = if st.l1_set.contains(&id) {
                st.l1_hits += 1;
                CacheTier::L1Vram
            } else {
                st.l2_hits += 1;
                if st.l1_set.len() < st.l1_capacity {
                    st.l1_set.insert(id);
                }
                CacheTier::L2SlabRam
            };
            return (tier, None);
        }

        if let Some(pos) = st.t2.iter().position(|x| *x == id) {
            st.t2.remove(pos);
            st.t2.push_back(id);
            let tier = if st.l1_set.contains(&id) {
                st.l1_hits += 1;
                CacheTier::L1Vram
            } else {
                st.l2_hits += 1;
                if st.l1_set.len() < st.l1_capacity {
                    st.l1_set.insert(id);
                }
                CacheTier::L2SlabRam
            };
            return (tier, None);
        }

        // Cache Miss — requires NVMe streaming
        st.ssd_misses += 1;
        let mut evicted = None;

        // Case 2: Hit in ghost list B1 (increase target size p for T1)
        if let Some(pos) = st.b1.iter().position(|x| *x == id) {
            let delta = if st.b1.len() >= st.b2.len() {
                1
            } else {
                st.b2.len() / st.b1.len().max(1)
            };
            st.p = (st.p + delta).min(st.capacity);
            evicted = Self::replace(&mut st, false);
            st.b1.remove(pos);
            st.t2.push_back(id);
            if st.l1_set.len() < st.l1_capacity {
                st.l1_set.insert(id);
            }
            return (CacheTier::L3NvmeMiss, evicted);
        }

        // Case 3: Hit in ghost list B2 (decrease target size p, favoring T2)
        if let Some(pos) = st.b2.iter().position(|x| *x == id) {
            let delta = if st.b2.len() >= st.b1.len() {
                1
            } else {
                st.b1.len() / st.b2.len().max(1)
            };
            st.p = st.p.saturating_sub(delta);
            evicted = Self::replace(&mut st, true);
            st.b2.remove(pos);
            st.t2.push_back(id);
            if st.l1_set.len() < st.l1_capacity {
                st.l1_set.insert(id);
            }
            return (CacheTier::L3NvmeMiss, evicted);
        }

        // Case 4: Cold miss (unseen expert)
        let l1_len = st.t1.len() + st.b1.len();
        if l1_len == st.capacity {
            if st.t1.len() < st.capacity {
                st.b1.pop_front();
                evicted = Self::replace(&mut st, false);
            } else {
                evicted = st.t1.pop_front();
                if let Some(ev) = evicted {
                    st.l1_set.remove(&ev);
                }
            }
        } else if l1_len < st.capacity {
            let total_len = st.t1.len() + st.t2.len() + st.b1.len() + st.b2.len();
            if total_len >= st.capacity {
                if total_len == 2 * st.capacity {
                    st.b2.pop_front();
                }
                evicted = Self::replace(&mut st, false);
            }
        }

        st.t1.push_back(id);
        (CacheTier::L3NvmeMiss, evicted)
    }

    fn replace(st: &mut ArcState, in_b2: bool) -> Option<ExpertId> {
        let evict_from_t1 = !st.t1.is_empty() && ((in_b2 && st.t1.len() == st.p) || (st.t1.len() > st.p));
        let victim = if evict_from_t1 {
            let v = st.t1.pop_front();
            if let Some(id) = v {
                st.b1.push_back(id);
            }
            v
        } else {
            let v = st.t2.pop_front();
            if let Some(id) = v {
                st.b2.push_back(id);
            }
            v
        };
        if let Some(v) = victim {
            st.l1_set.remove(&v);
        }
        victim
    }

    /// Exports the list of hot experts for persistence in a `*.roost` session snapshot
    pub fn export_warm_profile(&self) -> Vec<ExpertId> {
        let st = self.state.lock();
        let mut hot: Vec<ExpertId> = st.t2.iter().copied().collect();
        for id in &st.t1 {
            if !hot.contains(id) {
                hot.push(*id);
            }
        }
        hot
    }

    /// Pre-warms the cache from a `*.roost` snapshot profile without incurring cold miss penalties
    pub fn import_warm_profile(&self, experts: &[ExpertId]) {
        let mut st = self.state.lock();
        for &id in experts.iter().take(st.capacity) {
            if !st.t2.contains(&id) {
                st.t2.push_back(id);
                if st.l1_set.len() < st.l1_capacity {
                    st.l1_set.insert(id);
                }
            }
        }
    }

    pub fn telemetry(&self) -> ArcTelemetry {
        let st = self.state.lock();
        ArcTelemetry {
            l1_hits: st.l1_hits,
            l2_hits: st.l2_hits,
            ssd_misses: st.ssd_misses,
            t1_len: st.t1.len(),
            t2_len: st.t2.len(),
            l1_resident: st.l1_set.len(),
            p_adaptation: st.p,
        }
    }
}
