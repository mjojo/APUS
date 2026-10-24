use crate::{ApusError, ExpertId, PageAlignedSlot};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Custom Slab Memory Arena with fixed-size 4096-byte aligned slots
/// to prevent RAM heap fragmentation across multi-hour streaming sessions.
pub struct SlabArena {
    slot_size_bytes: usize,
    capacity_slots: usize,
    free_slots: Mutex<Vec<PageAlignedSlot>>,
    active_allocations: Mutex<HashMap<ExpertId, Arc<Mutex<PageAlignedSlot>>>>,
    locked_in_ram: bool,
}

impl SlabArena {
    /// Pre-allocates a pool of 4KB-aligned slots in system RAM
    pub fn new(slot_size_bytes: usize, capacity_slots: usize, lock_in_ram: bool) -> Result<Self, ApusError> {
        let mut free_slots = Vec::with_capacity(capacity_slots);
        for _ in 0..capacity_slots {
            let mut slot = PageAlignedSlot::new(slot_size_bytes)?;
            if lock_in_ram {
                slot.pin_memory();
            }
            free_slots.push(slot);
        }
        Ok(Self {
            slot_size_bytes,
            capacity_slots,
            free_slots: Mutex::new(free_slots),
            active_allocations: Mutex::new(HashMap::with_capacity(capacity_slots)),
            locked_in_ram: lock_in_ram,
        })
    }

    pub fn slot_size_bytes(&self) -> usize {
        self.slot_size_bytes
    }

    pub fn capacity_slots(&self) -> usize {
        self.capacity_slots
    }

    pub fn is_locked_in_ram(&self) -> bool {
        self.locked_in_ram
    }

    pub fn used_slots(&self) -> usize {
        self.active_allocations.lock().len()
    }

    pub fn free_slots_count(&self) -> usize {
        self.free_slots.lock().len()
    }

    /// Retrieves an already resident expert slot from the arena
    pub fn get(&self, id: &ExpertId) -> Option<Arc<Mutex<PageAlignedSlot>>> {
        self.active_allocations.lock().get(id).cloned()
    }

    /// Acquires a slot for an expert (or returns the existing one).
    /// If no free slots remain, the caller must evict a victim via `evict` first.
    pub fn acquire_slot(&self, id: ExpertId) -> Result<Arc<Mutex<PageAlignedSlot>>, ApusError> {
        let mut active = self.active_allocations.lock();
        if let Some(existing) = active.get(&id) {
            return Ok(Arc::clone(existing));
        }

        let mut free = self.free_slots.lock();
        let slot = free.pop().ok_or_else(|| {
            ApusError::AllocationFailed(format!(
                "SlabArena exhausted ({} slots occupied)",
                self.capacity_slots
            ))
        })?;
        let arc_slot = Arc::new(Mutex::new(slot));
        active.insert(id, Arc::clone(&arc_slot));
        Ok(arc_slot)
    }

    /// Evicts an expert from the arena, recycling its 4KB-aligned buffer back into the pool without deallocation
    pub fn evict(&self, id: &ExpertId) -> bool {
        let removed = self.active_allocations.lock().remove(id);
        if let Some(arc_slot) = removed {
            if let Ok(mutex_slot) = Arc::try_unwrap(arc_slot) {
                let mut slot = mutex_slot.into_inner();
                slot.as_mut_slice().fill(0);
                self.free_slots.lock().push(slot);
            } else {
                // If the slot is still referenced by a worker thread, allocate a replacement in the pool
                if let Ok(mut replacement) = PageAlignedSlot::new(self.slot_size_bytes) {
                    if self.locked_in_ram {
                        replacement.pin_memory();
                    }
                    self.free_slots.lock().push(replacement);
                }
            }
            true
        } else {
            false
        }
    }
}
