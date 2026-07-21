//! The quinn (QUIC) transport (ROADMAP P14.3 item 1) — feature `quic` only.
//!
//! Maps the protocol's reliability classes onto QUIC's native primitives:
//! **`ReliableOrdered` → a QUIC bidirectional stream** (QUIC guarantees reliable,
//! ordered delivery per stream), **`Unreliable` → a QUIC datagram** (best-effort,
//! unordered, small). This is the *alternative* to running the sans-io
//! [`Endpoint`](crate::Endpoint) over raw datagrams: when QUIC is the substrate we
//! let it own reliability and use its multiplexed streams to avoid head-of-line
//! blocking between channels.
//!
//! This module owns tokio + rustls(+ring). It is Ring-2 only (players, dedicated
//! servers); Ring-0 crates never enable `quic` (§2.5). Certs are self-signed via
//! `rcgen` for localhost/dev; production deployments supply real certs.

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::{ClientConfig, Connection, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};

use crate::frame::{decode_frame, encode_frame, Frame, Reliability};
use crate::NetError;

/// Max bytes we will read for a single reliable frame stream (a snapshot can be
/// large; this bounds a malicious/mis-sized peer).
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

impl From<quinn::ConnectionError> for NetError {
    fn from(e: quinn::ConnectionError) -> Self {
        NetError::Transport(e.to_string())
    }
}

/// Ensure the process-wide rustls crypto provider is the **ring** provider (not
/// the aws-lc-rs default). Idempotent — safe to call from every entry point.
pub fn install_crypto_provider() {
    // Errors mean a provider is already installed; that is fine.
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// A listening QUIC server plus its self-signed certificate (hand this DER to the
/// client so it can trust the connection).
pub struct QuicServer {
    pub endpoint: Endpoint,
    pub cert: CertificateDer<'static>,
}

impl QuicServer {
    /// Bind a QUIC server to `addr` (use `127.0.0.1:0` to get an ephemeral port,
    /// then read [`Endpoint::local_addr`]). Generates a self-signed cert for
    /// `localhost`.
    pub fn bind(addr: SocketAddr) -> Result<Self, NetError> {
        install_crypto_provider();
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| NetError::Transport(format!("cert gen: {e}")))?;
        let cert_der: CertificateDer<'static> = cert.cert.der().clone();
        let key = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

        let server_config = ServerConfig::with_single_cert(vec![cert_der.clone()], key.into())
            .map_err(|e| NetError::Transport(format!("server config: {e}")))?;

        let endpoint = Endpoint::server(server_config, addr)
            .map_err(|e| NetError::Transport(format!("endpoint::server: {e}")))?;
        Ok(Self {
            endpoint,
            cert: cert_der,
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, NetError> {
        self.endpoint
            .local_addr()
            .map_err(|e| NetError::Transport(e.to_string()))
    }

    /// Await the next inbound connection.
    pub async fn accept(&self) -> Result<Connection, NetError> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| NetError::Transport("endpoint closed".into()))?;
        Ok(incoming.await?)
    }
}

/// Build a QUIC client endpoint that trusts `server_cert` (the self-signed cert
/// the server generated). Bind to `127.0.0.1:0` for an ephemeral local port.
pub fn client_endpoint(
    bind: SocketAddr,
    server_cert: CertificateDer<'static>,
) -> Result<Endpoint, NetError> {
    install_crypto_provider();
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(server_cert)
        .map_err(|e| NetError::Transport(format!("add root: {e}")))?;
    let client_config = ClientConfig::with_root_certificates(Arc::new(roots))
        .map_err(|e| NetError::Transport(format!("client config: {e}")))?;
    let mut endpoint = Endpoint::client(bind)
        .map_err(|e| NetError::Transport(format!("endpoint::client: {e}")))?;
    endpoint.set_default_client_config(client_config);
    Ok(endpoint)
}

/// Connect to a server; `server_name` must match the cert (`"localhost"`).
pub async fn connect(
    endpoint: &Endpoint,
    server_addr: SocketAddr,
    server_name: &str,
) -> Result<Connection, NetError> {
    let connecting = endpoint
        .connect(server_addr, server_name)
        .map_err(|e| NetError::Transport(format!("connect: {e}")))?;
    Ok(connecting.await?)
}

/// Send one frame, routing by its reliability class:
/// reliable → a fresh bidirectional stream (written then finished); unreliable →
/// a datagram.
pub async fn send_frame(conn: &Connection, frame: &Frame) -> Result<(), NetError> {
    let bytes = encode_frame(frame);
    match frame.reliability {
        Reliability::ReliableOrdered => {
            let (mut send, _recv) = conn.open_bi().await?;
            send.write_all(&bytes)
                .await
                .map_err(|e| NetError::Transport(format!("write: {e}")))?;
            send.finish()
                .map_err(|e| NetError::Transport(format!("finish: {e}")))?;
            // finish() queues the FIN; quinn's connection driver flushes the
            // stream to the peer even after `send` drops (the Connection outlives
            // it). We deliberately do NOT await `stopped()` — a well-behaved peer
            // reads to end without sending STOP_SENDING, so `stopped()` would
            // never resolve and hang the send.
        }
        Reliability::Unreliable => {
            conn.send_datagram(bytes::Bytes::from(bytes))
                .map_err(|e| NetError::Transport(format!("datagram: {e}")))?;
        }
    }
    Ok(())
}

/// Receive the next reliable frame (accepts a bi-stream and reads it to end).
pub async fn recv_reliable(conn: &Connection) -> Result<Frame, NetError> {
    let (_send, mut recv) = conn.accept_bi().await?;
    let bytes = recv
        .read_to_end(MAX_FRAME_BYTES)
        .await
        .map_err(|e| NetError::Transport(format!("read: {e}")))?;
    decode_frame(&bytes)
}

/// Receive the next unreliable frame (a datagram).
pub async fn recv_unreliable(conn: &Connection) -> Result<Frame, NetError> {
    let bytes = conn.read_datagram().await?;
    decode_frame(&bytes)
}
