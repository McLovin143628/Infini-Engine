//! The sans-io reliability [`Endpoint`]: a pure state machine that turns an
//! unreliable, unordered, duplicating datagram pipe into per-channel
//! **exactly-once, in-order** delivery for reliable messages and **best-effort,
//! deduplicated** delivery for unreliable ones.
//!
//! It performs **no IO**. You feed it messages to send ([`send_reliable`]/
//! [`send_unreliable`](Endpoint::send_unreliable)), pull [`Packet`]s to put on the
//! wire ([`poll_send`](Endpoint::poll_send)), hand it received packets
//! ([`recv`](Endpoint::recv)), and drain delivered messages
//! ([`poll_recv`](Endpoint::poll_recv)). Time is injected via
//! [`update`](Endpoint::update) so retransmission is deterministic and testable
//! (no wall clock — the §2.5 rule applied to the network layer).
//!
//! This layer is transport-agnostic: it works over raw UDP, over QUIC unreliable
//! datagrams, or in-process (the loss/reorder/dup tests below run it over an
//! in-memory link). The quinn transport (feature `quic`) offers an *alternative*
//! reliable path by mapping reliable frames onto QUIC streams; this Endpoint is
//! the path used when only a datagram pipe is available.
//!
//! ## Ack scheme
//!
//! Classic sequence + redundant-bitfield acking. Every [`Packet`] carries its own
//! `seq`, the highest peer `seq` we have seen (`ack`), and a 64-bit `ack_bits`
//! window of the 64 packets before `ack` (bit `i` set ⇒ `ack - 1 - i` received).
//! A reliable message is retransmitted (under a fresh packet seq) until *some*
//! packet that carried it is acked. Delivery order is recovered on the receive
//! side by buffering out-of-order reliable ids and releasing them from
//! `reliable_recv_next` upward.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::frame::ChannelId;

/// An application message inside the reliability layer (a [`Frame`](crate::Frame)
/// minus its reliability tag, which is implied by the send method / packet slot).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub channel: ChannelId,
    pub payload: Vec<u8>,
}

/// A wire packet: an ack header plus the reliable/unreliable messages it carries.
/// Serialized (bincode) to a single datagram by the transport.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Packet {
    /// This packet's sequence number (monotonic per endpoint, starts at 1).
    pub seq: u64,
    /// Highest peer packet seq seen so far (0 ⇒ none).
    pub ack: u64,
    /// Redundant ack bitfield: bit `i` ⇒ packet `ack - 1 - i` was received.
    pub ack_bits: u64,
    /// Reliable messages, each with its stable reliable id.
    pub reliable: Vec<(u64, Message)>,
    /// Unreliable messages, each with its dedup sequence id.
    pub unreliable: Vec<(u64, Message)>,
}

impl Packet {
    /// A packet carries application data (not a pure ack) if it has any message.
    pub fn has_messages(&self) -> bool {
        !self.reliable.is_empty() || !self.unreliable.is_empty()
    }
}

/// Encode a packet to datagram bytes.
pub fn encode_packet(packet: &Packet) -> Vec<u8> {
    bincode::serde::encode_to_vec(packet, bincode::config::standard())
        .expect("a Packet is always encodable")
}

/// Decode a packet from datagram bytes.
pub fn decode_packet(bytes: &[u8]) -> Result<Packet, crate::NetError> {
    let (packet, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
    Ok(packet)
}

/// Exact serialized size of one reliable slot `(id, msg)` as it rides a packet —
/// the byte cost [`poll_send`](Endpoint::poll_send) charges against the budget. An
/// (impossible for our types) encode failure counts as oversize so the message
/// still ships alone rather than wedging the queue.
fn reliable_entry_size(id: u64, msg: &Message) -> usize {
    bincode::serde::encode_to_vec((id, msg), bincode::config::standard())
        .map(|v| v.len())
        .unwrap_or(usize::MAX)
}

/// Conservative per-packet **reliable** payload budget, in serialized bytes. A
/// [`Packet`] is one datagram (module doc), so its reliable messages must fit a
/// safe MTU; [`poll_send`](Endpoint::poll_send) fills each packet with due reliable
/// messages up to this budget (oldest-first) and leaves the rest for the next poll.
/// Sizes are measured exactly with bincode `serialized_size`, so this is a real
/// byte budget, not a message count. It is kept well under a 1200-byte safe MTU to
/// leave headroom for the ack header, an unreliable piggyback, and IP/UDP overhead.
/// A single message larger than the budget still ships alone (oversize-single — its
/// datagram size is then the transport's problem, e.g. IP fragmentation).
const RELIABLE_PAYLOAD_BUDGET: usize = 1100;

/// Endpoint tuning.
#[derive(Clone, Copy, Debug)]
pub struct EndpointConfig {
    /// Seconds after a reliable message's last send before it is retransmitted.
    pub resend_timeout: f64,
    /// Cap on retained sent-packet ack records (memory bound).
    pub sent_history: usize,
    /// Max reliable messages **in flight** (assigned an id, awaiting an ack) at
    /// once. Sends beyond this are held in a FIFO backlog (`pending_reliable`) and
    /// promoted into flight as acks free slots, so the in-flight set — and thus the
    /// per-packet resend volume — is bounded under sustained loss.
    pub max_in_flight: usize,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            resend_timeout: 0.1,
            sent_history: 1024,
            max_in_flight: 256,
        }
    }
}

struct ReliableEntry {
    msg: Message,
    last_sent: Option<f64>,
}

struct SentInfo {
    reliable_ids: Vec<u64>,
}

/// The reliability state machine (one per peer connection). Bidirectional: it both
/// sends and acknowledges.
pub struct Endpoint {
    cfg: EndpointConfig,
    now: f64,

    // ── send side ──
    next_packet_seq: u64,
    next_reliable_id: u64,
    next_unreliable_id: u64,
    reliable_unacked: BTreeMap<u64, ReliableEntry>,
    /// Reliable messages queued behind the in-flight cap, oldest-first. They get a
    /// reliable id only on promotion into `reliable_unacked` (which happens after
    /// every earlier send already took its id), so global send order is preserved.
    pending_reliable: VecDeque<Message>,
    pending_unreliable: Vec<(u64, Message)>,
    sent_packets: BTreeMap<u64, SentInfo>,

    // ── ack generation (what we tell the peer we've received) ──
    recv_highest: u64,
    recv_bits: u64,
    ack_dirty: bool,

    // ── reliable receive (ordering + dedup) ──
    reliable_recv_next: u64,
    reliable_recv_buffer: BTreeMap<u64, Message>,

    // ── unreliable receive (dedup only) ──
    unreliable_recv_highest: u64,
    unreliable_recv_bits: u64,

    // ── delivered, awaiting the app ──
    incoming: VecDeque<Message>,
}

impl Endpoint {
    pub fn new(cfg: EndpointConfig) -> Self {
        Self {
            cfg,
            now: 0.0,
            next_packet_seq: 1,
            next_reliable_id: 1,
            next_unreliable_id: 1,
            reliable_unacked: BTreeMap::new(),
            pending_reliable: VecDeque::new(),
            pending_unreliable: Vec::new(),
            sent_packets: BTreeMap::new(),
            recv_highest: 0,
            recv_bits: 0,
            ack_dirty: false,
            reliable_recv_next: 1,
            reliable_recv_buffer: BTreeMap::new(),
            unreliable_recv_highest: 0,
            unreliable_recv_bits: 0,
            incoming: VecDeque::new(),
        }
    }

    /// Advance the endpoint's clock (drives retransmission).
    pub fn update(&mut self, now: f64) {
        self.now = now;
    }

    /// Queue a reliable, ordered message. Retransmitted until acked. If the
    /// in-flight cap ([`EndpointConfig::max_in_flight`]) is reached — or a backlog
    /// already exists — it waits in the FIFO backlog and enters flight as acks free
    /// slots, so a burst never inflates the in-flight (resend) set.
    pub fn send_reliable(&mut self, channel: ChannelId, payload: Vec<u8>) {
        let msg = Message { channel, payload };
        // Preserve order: only send directly when nothing is already queued ahead.
        if self.pending_reliable.is_empty() && self.reliable_unacked.len() < self.cfg.max_in_flight
        {
            self.push_in_flight(msg);
        } else {
            self.pending_reliable.push_back(msg);
        }
    }

    /// Assign the next reliable id and put a message in flight.
    fn push_in_flight(&mut self, msg: Message) {
        let id = self.next_reliable_id;
        self.next_reliable_id += 1;
        self.reliable_unacked.insert(
            id,
            ReliableEntry {
                msg,
                last_sent: None,
            },
        );
    }

    /// Promote backlog into flight while there is room under the cap (FIFO — the
    /// oldest queued message takes the next id, keeping delivery order intact).
    fn promote_pending(&mut self) {
        while self.reliable_unacked.len() < self.cfg.max_in_flight {
            let Some(msg) = self.pending_reliable.pop_front() else {
                break;
            };
            self.push_in_flight(msg);
        }
    }

    /// Queue a best-effort message. Sent once; dropped if lost.
    pub fn send_unreliable(&mut self, channel: ChannelId, payload: Vec<u8>) {
        let id = self.next_unreliable_id;
        self.next_unreliable_id += 1;
        self.pending_unreliable
            .push((id, Message { channel, payload }));
    }

    /// Whether there is anything to put on the wire (data or a pending ack).
    pub fn wants_send(&self) -> bool {
        self.ack_dirty
            || !self.pending_unreliable.is_empty()
            || self.reliable_unacked.values().any(|e| self.due(e))
    }

    fn due(&self, e: &ReliableEntry) -> bool {
        match e.last_sent {
            None => true,
            Some(t) => self.now - t >= self.cfg.resend_timeout,
        }
    }

    /// Produce the next packet to send, or `None` if nothing is pending. Marks the
    /// included reliable messages as sent-at-`now` and records the packet for ack
    /// resolution.
    pub fn poll_send(&mut self) -> Option<Packet> {
        // Reliable messages due for (re)send, oldest-first (BTreeMap key = id =
        // send order).
        let due_ids: Vec<u64> = self
            .reliable_unacked
            .iter()
            .filter(|(_, e)| self.due(e))
            .map(|(id, _)| *id)
            .collect();

        let unreliable = std::mem::take(&mut self.pending_unreliable);

        if due_ids.is_empty() && unreliable.is_empty() && !self.ack_dirty {
            return None;
        }

        // Fill this packet's reliable slot up to the payload budget, oldest-first.
        // The rest of the due set rides the next poll_send call (the one-packet-per-
        // call contract is preserved — callers poll repeatedly). The first message
        // is always included, so a single oversize message still ships (alone).
        let mut reliable = Vec::new();
        let mut used = 0usize;
        for id in &due_ids {
            let Some(sz) = self
                .reliable_unacked
                .get(id)
                .map(|e| reliable_entry_size(*id, &e.msg))
            else {
                continue;
            };
            if !reliable.is_empty() && used + sz > RELIABLE_PAYLOAD_BUDGET {
                break; // defer the remaining due messages to the next poll
            }
            used += sz;
            if let Some(e) = self.reliable_unacked.get_mut(id) {
                e.last_sent = Some(self.now);
                reliable.push((*id, e.msg.clone()));
            }
        }
        let reliable_ids: Vec<u64> = reliable.iter().map(|(id, _)| *id).collect();

        let seq = self.next_packet_seq;
        self.next_packet_seq += 1;
        self.sent_packets.insert(seq, SentInfo { reliable_ids });
        // Bound the ack-record history.
        while self.sent_packets.len() > self.cfg.sent_history {
            let lowest = *self.sent_packets.keys().next().unwrap();
            self.sent_packets.remove(&lowest);
        }

        self.ack_dirty = false;
        Some(Packet {
            seq,
            ack: self.recv_highest,
            ack_bits: self.recv_bits,
            reliable,
            unreliable,
        })
    }

    /// Ingest a received packet: update our ack state, resolve the peer's acks
    /// (dropping acked reliable messages), and deliver in-order reliable +
    /// deduplicated unreliable messages into the [`poll_recv`](Self::poll_recv)
    /// queue.
    pub fn recv(&mut self, packet: &Packet) {
        self.track_received(packet.seq);
        self.resolve_acks(packet.ack, packet.ack_bits);

        if !packet.reliable.is_empty() {
            // Reliable messages require an ack back; flag it (unreliable/pure-ack
            // packets deliberately do not, which prevents an ack storm).
            self.ack_dirty = true;
        }
        for (id, msg) in &packet.reliable {
            self.recv_reliable(*id, msg.clone());
        }
        for (id, msg) in &packet.unreliable {
            self.recv_unreliable(*id, msg.clone());
        }
    }

    /// Pop the next delivered message (reliable in order, unreliable as accepted).
    pub fn poll_recv(&mut self) -> Option<Message> {
        self.incoming.pop_front()
    }

    /// Count of reliable messages **in flight** (sent, awaiting an ack). Bounded by
    /// [`EndpointConfig::max_in_flight`]. `0` with an empty backlog ⇒ all delivered.
    pub fn unacked_reliable(&self) -> usize {
        self.reliable_unacked.len()
    }

    /// Count of reliable messages waiting in the backlog behind the in-flight cap
    /// (not yet assigned an id / put on the wire).
    pub fn pending_reliable(&self) -> usize {
        self.pending_reliable.len()
    }

    // ── internals ──

    /// Record `seq` in our received-window so our outgoing acks reflect it.
    fn track_received(&mut self, seq: u64) {
        if seq == 0 {
            return;
        }
        if self.recv_highest == 0 {
            self.recv_highest = seq;
            self.recv_bits = 0;
            return;
        }
        if seq > self.recv_highest {
            let d = seq - self.recv_highest;
            // **`d == 64` is `u64 << 64`** (round-2, the net MED cluster): a
            // panic in dev and a *masked* `<< 0` in release, which silently
            // keeps every stale bit in the dedup window and makes the acks we
            // send a lie. The sibling four lines down gets it right with
            // `(1..=64)`, which is what makes this one look correct at a
            // glance. A jump of exactly 64 is one burst loss of 64 packets —
            // an ordinary event on a bad link, not an adversarial one.
            self.recv_bits = match d {
                0 => self.recv_bits,
                1..=63 => (self.recv_bits << d) | (1u64 << (d - 1)),
                // The old highest lands on the last bit of the window and
                // everything before it has fallen out of it.
                64 => 1u64 << 63,
                _ => 0,
            };
            self.recv_highest = seq;
        } else {
            let d = self.recv_highest - seq;
            if (1..=64).contains(&d) {
                self.recv_bits |= 1u64 << (d - 1);
            }
        }
    }

    /// Resolve the peer's ack header against our sent packets, dropping any
    /// reliable message that has now been acknowledged by *some* carrying packet.
    fn resolve_acks(&mut self, ack: u64, ack_bits: u64) {
        if ack == 0 {
            return;
        }
        self.ack_one(ack);
        for i in 0..64u64 {
            if ack_bits & (1u64 << i) != 0 {
                // bit i ⇒ packet (ack - 1 - i)
                if ack > i {
                    self.ack_one(ack - 1 - i);
                }
            }
        }
        // Acks freed in-flight slots; pull backlog forward to fill them.
        self.promote_pending();
    }

    fn ack_one(&mut self, seq: u64) {
        if let Some(info) = self.sent_packets.remove(&seq) {
            for id in info.reliable_ids {
                self.reliable_unacked.remove(&id);
            }
        }
    }

    /// Buffer an out-of-order reliable message until its predecessors arrive.
    ///
    /// # The window is bounded by what a peer may legally have in flight
    ///
    /// Round-2 finding **R2-5**. `reliable_recv_buffer` accepted any id at all
    /// and held it for ever: both sibling containers in this struct are capped
    /// (`sent_packets` by `sent_history`, the in-flight set by
    /// `max_in_flight`), and this one — the only one filled by *the peer's*
    /// numbers rather than our own — was not. A peer that sends ids
    /// `1, 3, 5, 7, …` and never the evens fills memory with messages that can
    /// never be released, because the contiguous prefix never advances.
    ///
    /// A well-behaved peer cannot exceed `reliable_recv_next + max_in_flight`
    /// by construction: that is exactly the in-flight cap its own sender obeys,
    /// and an id beyond it means a message we have not seen was already acked,
    /// which cannot happen. So the bound is the protocol's own, not a tuning
    /// constant — and a refusal is safe, because the reliable layer's whole
    /// contract is that an undelivered message is retransmitted.
    ///
    /// There is **no production caller of this crate yet** (P14.3 built the
    /// sans-io core ahead of its consumers), so this is protocol-core debt
    /// rather than a live hazard — which is the reason to fix it before there
    /// is a shipped client whose memory it is.
    fn recv_reliable(&mut self, id: u64, msg: Message) {
        if id < self.reliable_recv_next {
            return; // already delivered
        }
        if id
            >= self
                .reliable_recv_next
                .saturating_add(self.cfg.max_in_flight as u64)
        {
            // Past anything the peer's own in-flight cap allows. Dropped, not
            // buffered: it will be retransmitted once the gap ahead of it
            // closes.
            return;
        }
        if self.reliable_recv_buffer.contains_key(&id) {
            return; // duplicate, still buffered
        }
        self.reliable_recv_buffer.insert(id, msg);
        // Release the contiguous prefix.
        while let Some(m) = self.reliable_recv_buffer.remove(&self.reliable_recv_next) {
            self.incoming.push_back(m);
            self.reliable_recv_next += 1;
        }
    }

    fn recv_unreliable(&mut self, id: u64, msg: Message) {
        if id == 0 {
            return;
        }
        if self.unreliable_recv_highest == 0 {
            self.unreliable_recv_highest = id;
            self.unreliable_recv_bits = 0;
            self.incoming.push_back(msg);
            return;
        }
        if id > self.unreliable_recv_highest {
            let d = id - self.unreliable_recv_highest;
            // The same `u64 << 64` as `track_received` — see the comment there.
            self.unreliable_recv_bits = match d {
                0 => self.unreliable_recv_bits,
                1..=63 => (self.unreliable_recv_bits << d) | (1u64 << (d - 1)),
                64 => 1u64 << 63,
                _ => 0,
            };
            self.unreliable_recv_highest = id;
            self.incoming.push_back(msg);
        } else {
            let d = self.unreliable_recv_highest - id;
            if d == 0 {
                return; // duplicate of the current highest
            }
            if d > 64 {
                return; // too old to dedup — drop conservatively
            }
            let bit = 1u64 << (d - 1);
            if self.unreliable_recv_bits & bit != 0 {
                return; // duplicate
            }
            self.unreliable_recv_bits |= bit;
            self.incoming.push_back(msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch() -> ChannelId {
        ChannelId::REPLICATION
    }

    /// A deterministic pseudo-random link that can lose, reorder, and duplicate
    /// packets. Seeded LCG so every test run is reproducible.
    struct LossyLink {
        rng: u64,
        loss: u32,    // 0..100 drop percent
        dup: u32,     // 0..100 duplicate percent
        reorder: u32, // 0..100 hold-and-shuffle percent
        inflight: Vec<Packet>,
    }

    impl LossyLink {
        fn new(seed: u64, loss: u32, dup: u32, reorder: u32) -> Self {
            Self {
                rng: seed,
                loss,
                dup,
                reorder,
                inflight: Vec::new(),
            }
        }

        fn next(&mut self) -> u32 {
            // Numerical Recipes LCG.
            self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.rng >> 33) & 0xFFFF_FFFF) as u32
        }

        fn chance(&mut self, pct: u32) -> bool {
            self.next() % 100 < pct
        }

        /// Offer a packet to the link; returns the packets it decides to deliver
        /// this step (after loss/dup/reorder).
        fn transmit(&mut self, p: Packet) -> Vec<Packet> {
            if self.chance(self.loss) {
                // dropped
            } else {
                let copies = if self.chance(self.dup) { 2 } else { 1 };
                for _ in 0..copies {
                    self.inflight.push(p.clone());
                }
            }
            // Decide what to release now: sometimes hold to induce reorder.
            if self.inflight.len() > 1 && self.chance(self.reorder) {
                // Swap the two oldest so ordering is perturbed.
                let last = self.inflight.len() - 1;
                self.inflight.swap(0, last);
            }
            std::mem::take(&mut self.inflight)
        }
    }

    #[test]
    fn clean_link_delivers_reliable_in_order_exactly_once() {
        let mut a = Endpoint::new(EndpointConfig::default());
        let mut b = Endpoint::new(EndpointConfig::default());
        for i in 0..50u32 {
            a.send_reliable(ch(), i.to_le_bytes().to_vec());
        }
        // Pump until all reliable acked.
        let mut t = 0.0;
        for _ in 0..500 {
            t += 0.01;
            a.update(t);
            b.update(t);
            while let Some(p) = a.poll_send() {
                b.recv(&p);
            }
            while let Some(p) = b.poll_send() {
                a.recv(&p);
            }
            if a.unacked_reliable() == 0 {
                break;
            }
        }
        assert_eq!(a.unacked_reliable(), 0, "not everything got acked");
        let mut got = Vec::new();
        while let Some(m) = b.poll_recv() {
            got.push(u32::from_le_bytes(m.payload.try_into().unwrap()));
        }
        assert_eq!(got, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn lossy_reordering_duplicating_link_still_exactly_once_in_order() {
        for seed in 0..16u64 {
            let mut a = Endpoint::new(EndpointConfig::default());
            let mut b = Endpoint::new(EndpointConfig::default());
            let mut atob = LossyLink::new(seed * 7 + 1, 30, 20, 40);
            let mut btoa = LossyLink::new(seed * 13 + 3, 30, 20, 40);

            let n = 40u32;
            for i in 0..n {
                a.send_reliable(ch(), i.to_le_bytes().to_vec());
            }

            let mut got = Vec::new();
            let mut t = 0.0;
            for _ in 0..5000 {
                t += 0.01;
                a.update(t);
                b.update(t);
                while let Some(p) = a.poll_send() {
                    for d in atob.transmit(p) {
                        b.recv(&d);
                    }
                }
                while let Some(p) = b.poll_send() {
                    for d in btoa.transmit(p) {
                        a.recv(&d);
                    }
                }
                while let Some(m) = b.poll_recv() {
                    got.push(u32::from_le_bytes(m.payload.clone().try_into().unwrap()));
                }
                if a.unacked_reliable() == 0 {
                    break;
                }
            }
            assert_eq!(
                a.unacked_reliable(),
                0,
                "seed {seed}: reliable never fully acked"
            );
            assert_eq!(
                got,
                (0..n).collect::<Vec<_>>(),
                "seed {seed}: reliable delivery not exactly-once-in-order"
            );
        }
    }

    #[test]
    fn unreliable_is_best_effort_deduped_no_reorder_guarantee() {
        // Clean link: all unreliable delivered exactly once.
        let mut a = Endpoint::new(EndpointConfig::default());
        let mut b = Endpoint::new(EndpointConfig::default());
        for i in 0..20u32 {
            a.send_unreliable(ch(), i.to_le_bytes().to_vec());
        }
        let mut t = 0.0;
        let mut got = Vec::new();
        for _ in 0..50 {
            t += 0.01;
            a.update(t);
            b.update(t);
            while let Some(p) = a.poll_send() {
                // Duplicate every packet to prove receive-side dedup.
                b.recv(&p);
                b.recv(&p);
            }
            while let Some(m) = b.poll_recv() {
                got.push(u32::from_le_bytes(m.payload.clone().try_into().unwrap()));
            }
        }
        got.sort();
        assert_eq!(
            got,
            (0..20).collect::<Vec<_>>(),
            "unreliable lost or duplicated"
        );
    }

    #[test]
    fn unreliable_drops_under_loss_without_blocking() {
        let mut a = Endpoint::new(EndpointConfig::default());
        let mut b = Endpoint::new(EndpointConfig::default());
        let mut link = LossyLink::new(99, 50, 0, 0);
        let mut t = 0.0;
        let mut got = std::collections::BTreeSet::new();
        // Send one unreliable per step so each rides its own packet and loss is
        // per-message (not all-or-nothing in a single batched datagram).
        for i in 0..100u32 {
            a.send_unreliable(ch(), i.to_le_bytes().to_vec());
            t += 0.01;
            a.update(t);
            b.update(t);
            while let Some(p) = a.poll_send() {
                for d in link.transmit(p) {
                    b.recv(&d);
                }
            }
            while let Some(m) = b.poll_recv() {
                got.insert(u32::from_le_bytes(m.payload.clone().try_into().unwrap()));
            }
        }
        // Best-effort: a strict subset arrives (some lost), never a duplicate,
        // never a phantom id.
        assert!(got.len() < 100, "expected some unreliable loss");
        assert!(got.iter().all(|v| *v < 100));
    }

    #[test]
    fn packet_round_trips() {
        let p = Packet {
            seq: 5,
            ack: 4,
            ack_bits: 0b101,
            reliable: vec![(
                1,
                Message {
                    channel: ch(),
                    payload: vec![9],
                },
            )],
            unreliable: vec![],
        };
        let bytes = encode_packet(&p);
        assert_eq!(decode_packet(&bytes).unwrap(), p);
    }

    #[test]
    fn oversized_reliable_backlog_splits_across_packets_by_budget() {
        // A backlog whose combined size far exceeds one packet's budget must not be
        // packed into a single oversized datagram — poll_send fills to the budget
        // and defers the rest.
        let cfg = EndpointConfig {
            max_in_flight: 1000,
            ..Default::default()
        };
        let mut a = Endpoint::new(cfg);
        let n = 50usize;
        for _ in 0..n {
            a.send_reliable(ch(), vec![0u8; 200]); // ~200 B each ⇒ budget hit in ~5
        }
        a.update(0.0);
        let p = a.poll_send().expect("a packet is produced");
        assert!(
            p.reliable.len() < n,
            "budget splits the backlog ({} of {n})",
            p.reliable.len()
        );
        assert!(
            encode_packet(&p).len() <= 1400,
            "one datagram stays near a safe MTU"
        );
    }

    #[test]
    fn a_single_oversize_reliable_message_still_ships_alone() {
        let mut a = Endpoint::new(EndpointConfig::default());
        a.send_reliable(ch(), vec![7u8; 4000]); // larger than the whole budget
        a.update(0.0);
        let p = a.poll_send().expect("a packet is produced");
        assert_eq!(
            p.reliable.len(),
            1,
            "an oversize message goes alone rather than wedging the queue forever"
        );
    }

    #[test]
    fn sustained_loss_bounds_in_flight_and_delivers_all_after_recovery() {
        // The M1 gate: under total loss the in-flight (resend) set stays ≤ cap while
        // the backlog holds the rest; once loss clears, every message is delivered
        // exactly once, in order.
        let cfg = EndpointConfig {
            max_in_flight: 8,
            ..Default::default()
        };
        let mut a = Endpoint::new(cfg);
        let mut b = Endpoint::new(cfg);

        let n = 100u32;
        for i in 0..n {
            a.send_reliable(ch(), i.to_le_bytes().to_vec());
        }
        assert!(a.unacked_reliable() <= 8, "cap holds from the first send");
        assert_eq!(
            a.pending_reliable(),
            n as usize - 8,
            "the rest is backlogged"
        );

        let mut t = 0.0;
        // Phase 1 — total loss a→b: nothing acks, so in-flight is pinned at the cap.
        for _ in 0..50 {
            t += 0.01;
            a.update(t);
            b.update(t);
            while a.poll_send().is_some() { /* dropped */ }
            assert!(a.unacked_reliable() <= 8, "in-flight never exceeds the cap");
        }

        // Phase 2 — loss clears: pump both directions to completion.
        let mut got = Vec::new();
        for _ in 0..10_000 {
            t += 0.01;
            a.update(t);
            b.update(t);
            while let Some(p) = a.poll_send() {
                b.recv(&p);
            }
            while let Some(p) = b.poll_send() {
                a.recv(&p);
            }
            while let Some(m) = b.poll_recv() {
                got.push(u32::from_le_bytes(m.payload.clone().try_into().unwrap()));
            }
            assert!(
                a.unacked_reliable() <= 8,
                "in-flight stays bounded throughout"
            );
            if a.unacked_reliable() == 0 && a.pending_reliable() == 0 {
                break;
            }
        }
        assert_eq!(a.unacked_reliable(), 0, "everything acked");
        assert_eq!(a.pending_reliable(), 0, "backlog drained");
        assert_eq!(
            got,
            (0..n).collect::<Vec<_>>(),
            "exactly-once, in order after recovery"
        );
    }

    /// **Round-2, the net MED cluster**: a burst loss of exactly 64 is
    /// `u64 << 64` — a panic in dev, and in release a *masked* `<< 0` that
    /// keeps every stale bit in the dedup window.
    ///
    /// The window itself is asserted, not the delivery: a forward jump always
    /// delivers, so the only observable of the shift is the bitfield the acks
    /// are built from. (Measured: an arm that drove packets through the public
    /// path and checked what arrived left the `<< 0` mutation green.)
    #[test]
    fn a_gap_of_exactly_sixty_four_does_not_shift_a_u64_by_its_own_width() {
        // `track_received` (the packet-seq window). Start with two received
        // packets so there is a stale bit for a masked shift to keep.
        let mut e = Endpoint::new(EndpointConfig::default());
        e.track_received(1);
        e.track_received(2);
        assert_eq!(e.recv_bits, 1, "packet 1 sits one below the highest");

        e.track_received(2 + 64);
        assert_eq!(e.recv_highest, 66, "the window did not advance");
        assert_eq!(
            e.recv_bits,
            1u64 << 63,
            "a gap of exactly 64 must leave ONLY the old highest, at the far end              of the window; a masked `<< 0` keeps every stale bit and makes the              acks we send a lie"
        );

        // 63 and 65 bracket it, so the arm cannot pass by accident.
        let mut e = Endpoint::new(EndpointConfig::default());
        e.track_received(1);
        e.track_received(2);
        e.track_received(2 + 63);
        assert_eq!(e.recv_bits, (1u64 << 63) | (1u64 << 62));

        let mut e = Endpoint::new(EndpointConfig::default());
        e.track_received(1);
        e.track_received(2);
        e.track_received(2 + 65);
        assert_eq!(e.recv_bits, 0, "past the window, nothing survives");

        // The unreliable mirror, same shape and the same bug.
        let mut e = Endpoint::new(EndpointConfig::default());
        let m = |i: u8| Message {
            channel: ch(),
            payload: vec![i],
        };
        e.recv_unreliable(1, m(1));
        e.recv_unreliable(2, m(2));
        assert_eq!(e.unreliable_recv_bits, 1);
        e.recv_unreliable(2 + 64, m(3));
        assert_eq!(
            e.unreliable_recv_bits,
            1u64 << 63,
            "the unreliable window shifted by its own width"
        );
    }

    /// **Round-2 finding R2-5**: the reliable reorder buffer is bounded by what
    /// the peer's own in-flight cap allows.
    ///
    /// Both sibling containers in this struct are capped — `sent_packets` by
    /// `sent_history`, the in-flight set by `max_in_flight`. This one, the only
    /// one filled by the PEER's numbers rather than our own, was not. A peer
    /// that sends only odd ids fills memory with messages that can never be
    /// released, because the contiguous prefix never advances.
    #[test]
    fn the_reliable_reorder_buffer_cannot_grow_past_the_in_flight_cap() {
        let cfg = EndpointConfig {
            max_in_flight: 8,
            ..EndpointConfig::default()
        };
        let mut e = Endpoint::new(cfg);

        // Ids 2..=200, all out of order (1 never arrives), so nothing is ever
        // released and every one of them wants to be buffered.
        for id in 2..=200u64 {
            e.recv_reliable(
                id,
                Message {
                    channel: ch(),
                    payload: vec![id as u8],
                },
            );
        }
        assert!(
            e.reliable_recv_buffer.len() <= cfg.max_in_flight,
            "the reorder buffer holds {} of an 8-message window — it accepted ids              the peer cannot legally have in flight",
            e.reliable_recv_buffer.len()
        );
        assert!(
            !e.reliable_recv_buffer.is_empty(),
            "nothing was buffered at all, so this arm is vacuous"
        );

        // …and the messages inside the window are still delivered once the gap
        // closes: a bound that loses legitimate traffic is not a bound.
        e.recv_reliable(
            1,
            Message {
                channel: ch(),
                payload: vec![1],
            },
        );
        let mut seen = Vec::new();
        while let Some(m) = e.poll_recv() {
            seen.push(m.payload[0]);
        }
        assert_eq!(
            seen,
            (1..=8u8).collect::<Vec<_>>(),
            "the in-window prefix was not released in order"
        );
    }
}
