//! Message framing: a channel id + reliability class wrapping an opaque payload.
//!
//! A [`Frame`] is the transport-neutral unit every path (the sans-io
//! [`crate::reliability::Endpoint`] and the quinn transport) carries. The
//! reliability class tells a multiplexing transport which sub-pipe to use
//! (QUIC stream vs datagram); the channel id lets a game keep independent ordered
//! streams (e.g. "world snapshots" separate from "chat") without head-of-line
//! blocking between them.

use serde::{Deserialize, Serialize};

/// A logical channel. Games assign meaning (0 = replication, 1 = rpc, …); the
/// protocol only uses it to keep per-channel ordering independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChannelId(pub u8);

impl ChannelId {
    pub const REPLICATION: ChannelId = ChannelId(0);
    pub const RPC: ChannelId = ChannelId(1);
}

/// Delivery guarantee for a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reliability {
    /// Exactly-once, in per-channel order. Retransmitted until acked. Backed by
    /// the [`crate::reliability::Endpoint`] over a raw datagram pipe, or by a QUIC
    /// stream in the quinn transport.
    ReliableOrdered,
    /// Best-effort. Sent once, deduplicated on receipt, never retransmitted, no
    /// ordering guarantee across frames. For high-rate state (transform snapshots)
    /// where the freshest datagram supersedes a lost one.
    Unreliable,
}

/// A framed application message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frame {
    pub channel: ChannelId,
    pub reliability: Reliability,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn reliable(channel: ChannelId, payload: Vec<u8>) -> Self {
        Self {
            channel,
            reliability: Reliability::ReliableOrdered,
            payload,
        }
    }

    pub fn unreliable(channel: ChannelId, payload: Vec<u8>) -> Self {
        Self {
            channel,
            reliability: Reliability::Unreliable,
            payload,
        }
    }
}

/// Encode a frame to bytes (bincode, engine-standard config).
pub fn encode_frame(frame: &Frame) -> Vec<u8> {
    bincode::serde::encode_to_vec(frame, bincode::config::standard())
        .expect("a Frame is always encodable")
}

/// Decode a frame from bytes.
pub fn decode_frame(bytes: &[u8]) -> Result<Frame, crate::NetError> {
    let (frame, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trips() {
        let f = Frame::reliable(ChannelId::RPC, vec![1, 2, 3]);
        let bytes = encode_frame(&f);
        assert_eq!(decode_frame(&bytes).unwrap(), f);
    }

    #[test]
    fn reliability_helpers() {
        assert_eq!(
            Frame::unreliable(ChannelId::REPLICATION, vec![]).reliability,
            Reliability::Unreliable
        );
    }
}
