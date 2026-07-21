//! Infinity Engine networking (ROADMAP P14.3).
//!
//! Two layers, cleanly split so the deterministic sim never pulls an async runtime:
//!
//! - **The protocol core (this crate's default features, sans-io, pure):**
//!   [`frame`] (channel + reliability framing), [`reliability`] (the exactly-once
//!   / best-effort [`Endpoint`] state machine), [`snapshot`] (Guid-keyed transform
//!   replication with delta-against-baseline), and [`rpc`] (numbered RPCs with
//!   bincode args). No sockets, no tokio, no TLS — every piece is a pure function
//!   or a fed-in/polled-out state machine, so it is fully unit-testable and safe
//!   for Ring-0 consumers.
//! - **The quinn (QUIC) transport (`quic` feature, off by default):** [`transport`]
//!   maps reliable frames onto QUIC streams and unreliable frames onto QUIC
//!   datagrams. It owns a tokio runtime and rustls/ring, so it is enabled only by
//!   Ring-2 apps (player, dedicated server) and the integration test.
//!
//! ## Ring placement (the §2.5 reasoning)
//!
//! `quinn` requires `tokio`, and tokio is a Ring-2-only dependency by rule (async
//! is for IO concurrency, not the compute sim). So networking is split: the
//! **protocol** is Ring-0-safe (sans-io, deterministic, this crate's default) and
//! is what `inf-runtime`'s replication glue links against; the **transport** lives
//! behind the `quic` feature that only Ring-2 apps turn on. Replication itself
//! happens at snapshot boundaries (a [`snapshot::NetSnapshot`] is produced from a
//! post-step world), so the transport is IO that never enters the fixed-step loop
//! — the determinism guarantee is preserved.

pub mod frame;
pub mod reliability;
pub mod rpc;
pub mod snapshot;

#[cfg(feature = "quic")]
pub mod transport;

pub use frame::{decode_frame, encode_frame, ChannelId, Frame, Reliability};
pub use reliability::{Endpoint, EndpointConfig, Message, Packet};
pub use rpc::{RpcCall, RpcId, RpcRegistry};
pub use snapshot::{NetId, NetSnapshot, SnapshotDelta, TransformState};

/// Errors surfaced by the protocol core (encode/decode) and the transport.
#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("encode error: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    #[error("decode error: {0}")]
    Decode(#[from] bincode::error::DecodeError),
    #[cfg(feature = "quic")]
    #[error("transport error: {0}")]
    Transport(String),
}
