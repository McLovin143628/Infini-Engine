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

/// Endpoint tuning.
#[derive(Clone, Copy, Debug)]
pub struct EndpointConfig {
    /// Seconds after a reliable message's last send before it is retransmitted.
    pub resend_timeout: f64,
    /// Cap on retained sent-packet ack records (memory bound).
    pub sent_history: usize,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            resend_timeout: 0.1,
            sent_history: 1024,
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

    /// Queue a reliable, ordered message. Retransmitted until acked.
    pub fn send_reliable(&mut self, channel: ChannelId, payload: Vec<u8>) {
        let id = self.next_reliable_id;
        self.next_reliable_id += 1;
        self.reliable_unacked.insert(
            id,
            ReliableEntry {
                msg: Message { channel, payload },
                last_sent: None,
            },
        );
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
        // Reliable messages due for (re)send.
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

        let mut reliable = Vec::with_capacity(due_ids.len());
        for id in &due_ids {
            if let Some(e) = self.reliable_unacked.get_mut(id) {
                e.last_sent = Some(self.now);
                reliable.push((*id, e.msg.clone()));
            }
        }

        let seq = self.next_packet_seq;
        self.next_packet_seq += 1;
        self.sent_packets.insert(
            seq,
            SentInfo {
                reliable_ids: due_ids,
            },
        );
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

    /// Count of reliable messages still awaiting an ack (0 ⇒ everything delivered).
    pub fn unacked_reliable(&self) -> usize {
        self.reliable_unacked.len()
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
            if d <= 64 {
                self.recv_bits = (self.recv_bits << d) | (1u64 << (d - 1));
            } else {
                self.recv_bits = 0;
            }
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
    }

    fn ack_one(&mut self, seq: u64) {
        if let Some(info) = self.sent_packets.remove(&seq) {
            for id in info.reliable_ids {
                self.reliable_unacked.remove(&id);
            }
        }
    }

    fn recv_reliable(&mut self, id: u64, msg: Message) {
        if id < self.reliable_recv_next {
            return; // already delivered
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
            if d <= 64 {
                self.unreliable_recv_bits = (self.unreliable_recv_bits << d) | (1u64 << (d - 1));
            } else {
                self.unreliable_recv_bits = 0;
            }
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
}
