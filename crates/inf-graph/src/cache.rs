//! Content-addressed node-output cache (xxh3).
//!
//! De-geo'd port of GeoCanvas `nodegraph::cache`. The cache key folds every
//! upstream node's output hash into the current node's hash, so **subgraph
//! caching is emergent**: a param edit invalidates exactly its downstream
//! cone, and identical subgraphs (even across different graphs) share entries.
//! No dirty tracking — incremental re-execution falls out of cache hits.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use crate::model::ParamMap;
use crate::value::GraphValue;

/// The cache key for a node's output: `xxh3(type_id ‖ 0xff ‖
/// canonical-params-json ‖ upstream-hashes-LE)`. `ParamMap` is a `BTreeMap`,
/// so its JSON is key-sorted and canonical → identical params hash identically.
pub fn node_hash(type_id: &str, resolved_params: &ParamMap, upstream: &[u64]) -> u64 {
    let mut h = Xxh3::new();
    h.update(type_id.as_bytes());
    h.update(&[0xff]);
    // BTreeMap → deterministic key order → canonical string.
    let json = serde_json::to_string(resolved_params).unwrap_or_default();
    h.update(json.as_bytes());
    for u in upstream {
        h.update(&u.to_le_bytes());
    }
    h.digest()
}

/// Hash of raw content bytes (e.g. an external input's current contents).
pub fn content_hash(bytes: &[u8]) -> u64 {
    xxh3_64(bytes)
}

/// A cheap identity hash for a file-backed input: path + length + mtime nanos.
/// An edited file changes its mtime → dirties its downstream cone without
/// reading the (possibly huge) contents.
pub fn file_identity_hash(path: &Path) -> u64 {
    let mut h = Xxh3::new();
    h.update(path.to_string_lossy().as_bytes());
    if let Ok(meta) = std::fs::metadata(path) {
        h.update(&meta.len().to_le_bytes());
        if let Ok(modified) = meta.modified() {
            if let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
                h.update(&dur.as_nanos().to_le_bytes());
            }
        }
    }
    h.digest()
}

struct Entry<V> {
    value: V,
    ram: usize,
    last_use: u64,
}

struct Inner<V> {
    map: BTreeMap<u64, Entry<V>>,
    ram_total: usize,
    tick: u64,
}

/// A session-scoped, content-addressed, byte-capped LRU cache of node outputs.
/// Keyed by [`node_hash`]. Thread-safe.
pub struct NodeCache<V> {
    inner: Mutex<Inner<V>>,
    ram_cap: usize,
}

impl<V: GraphValue> NodeCache<V> {
    /// A cache holding up to `ram_cap` bytes of live values.
    pub fn new(ram_cap: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: BTreeMap::new(),
                ram_total: 0,
                tick: 0,
            }),
            ram_cap,
        }
    }

    /// Look up a cached output, bumping its LRU recency on a hit.
    pub fn get(&self, key: u64) -> Option<V> {
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;
        let entry = inner.map.get_mut(&key)?;
        entry.last_use = tick;
        Some(entry.value.clone())
    }

    /// Store an output. A value larger than the whole cap is skipped (never
    /// evicts everything for one giant entry); otherwise LRU-evicts to fit.
    pub fn insert(&self, key: u64, value: V) {
        let ram = value.ram_bytes().max(1);
        let mut inner = self.inner.lock().unwrap();
        if ram > self.ram_cap {
            return;
        }
        inner.tick += 1;
        let tick = inner.tick;
        if let Some(old) = inner.map.remove(&key) {
            inner.ram_total = inner.ram_total.saturating_sub(old.ram);
        }
        inner.map.insert(
            key,
            Entry {
                value,
                ram,
                last_use: tick,
            },
        );
        inner.ram_total += ram;
        while inner.ram_total > self.ram_cap {
            let Some((&victim, _)) = inner.map.iter().min_by_key(|(_, e)| e.last_use) else {
                break;
            };
            if let Some(e) = inner.map.remove(&victim) {
                inner.ram_total = inner.ram_total.saturating_sub(e.ram);
            }
        }
    }

    /// Drop everything (e.g. on project switch).
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.map.clear();
        inner.ram_total = 0;
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ParamValue;

    impl GraphValue for i64 {
        fn ram_bytes(&self) -> usize {
            64
        }
    }

    #[test]
    fn node_hash_is_deterministic_and_param_sensitive() {
        let mut p1 = ParamMap::new();
        p1.insert("a".into(), ParamValue::Int(1));
        p1.insert("b".into(), ParamValue::Int(2));
        // Same content, different insertion order → same hash (BTreeMap).
        let mut p2 = ParamMap::new();
        p2.insert("b".into(), ParamValue::Int(2));
        p2.insert("a".into(), ParamValue::Int(1));
        assert_eq!(node_hash("t", &p1, &[]), node_hash("t", &p2, &[]));

        // A changed param changes the hash.
        let mut p3 = p1.clone();
        p3.insert("a".into(), ParamValue::Int(9));
        assert_ne!(node_hash("t", &p1, &[]), node_hash("t", &p3, &[]));

        // Upstream hashes fold in.
        assert_ne!(node_hash("t", &p1, &[7]), node_hash("t", &p1, &[8]));
    }

    #[test]
    fn cache_stores_and_evicts_lru() {
        let cache: NodeCache<i64> = NodeCache::new(128); // 2 entries at 64B
        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.get(1), Some(10));
        // Insert a third → evicts the least-recently-used (key 2; key 1 was just touched).
        cache.insert(3, 30);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(1), Some(10));
        assert_eq!(cache.get(3), Some(30));
        assert_eq!(cache.get(2), None);
    }
}
