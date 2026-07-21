//! quinn (QUIC) transport integration test (ROADMAP P14.3 item 1).
//!
//! Runs a real server + client over `127.0.0.1` inside a tokio runtime and proves
//! the two headline paths end-to-end:
//!   1. **Transform replication client-ward** — the authoritative server pushes a
//!      100-entity [`NetSnapshot`] over a QUIC *stream* (reliable); the client
//!      decodes it and it equals what was sent.
//!   2. **RPC round-trip** — the client calls a numbered RPC over a reliable
//!      stream; the server dispatches it through an [`RpcRegistry`] and replies;
//!      the client's decoded response is correct.
//!   3. **Unreliable datagram** — a small best-effort frame over a QUIC datagram
//!      (mapping the `Unreliable` class), delivered on loopback.
//!
//! Only built/run with `--features quic`.
#![cfg(feature = "quic")]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

use inf_net::frame::Frame;
use inf_net::snapshot::{NetId, NetSnapshot, TransformState};
use inf_net::transport::{self, QuicServer};
use inf_net::{rpc, ChannelId, RpcId, RpcRegistry};

fn make_snapshot(n: u128) -> NetSnapshot {
    let mut snap = NetSnapshot::new(42);
    for i in 1..=n {
        snap.insert(
            NetId(i),
            TransformState {
                translation: [i as f64, i as f64 * 0.5, -(i as f64)],
                rotation: [0.0, (i as f64) * 3.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
        );
    }
    snap
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quic_replicates_transforms_and_round_trips_rpc() {
    let result = tokio::time::timeout(Duration::from_secs(20), run()).await;
    result.expect("quic integration test timed out");
}

async fn run() {
    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let server = QuicServer::bind(bind).expect("server bind");
    let server_addr = server.local_addr().unwrap();
    let cert = server.cert.clone();

    let sent_snapshot = make_snapshot(100);
    let server_snapshot = sent_snapshot.clone();

    // ── server task ──
    let server_fut = async move {
        let conn = server.accept().await.expect("accept");

        // 1. push the snapshot (reliable stream).
        let frame = Frame::reliable(ChannelId::REPLICATION, server_snapshot.encode());
        transport::send_frame(&conn, &frame)
            .await
            .expect("send snap");

        // 2. answer one RPC.
        let mut registry = RpcRegistry::new();
        registry.register::<(u32, String), u32, _>(RpcId(7), |(n, s)| Some(n + s.len() as u32));
        let call_frame = transport::recv_reliable(&conn)
            .await
            .expect("recv rpc call");
        let resp = registry
            .dispatch(&call_frame.payload)
            .expect("dispatch")
            .expect("rpc produced a response");
        let resp_frame = Frame::reliable(ChannelId::RPC, resp);
        transport::send_frame(&conn, &resp_frame)
            .await
            .expect("send rpc resp");

        // 3. receive one unreliable datagram.
        let dg = transport::recv_unreliable(&conn)
            .await
            .expect("recv datagram");
        conn.closed().await;
        dg.payload
    };

    // ── client task ──
    let client_fut = async move {
        let endpoint = transport::client_endpoint("127.0.0.1:0".parse().unwrap(), cert)
            .expect("client endpoint");
        let conn = transport::connect(&endpoint, server_addr, "localhost")
            .await
            .expect("connect");

        // 1. receive the replicated snapshot.
        let snap_frame = transport::recv_reliable(&conn).await.expect("recv snap");
        let received = NetSnapshot::decode(&snap_frame.payload).expect("decode snap");

        // 2. RPC call → response.
        let call = rpc::encode_call(RpcId(7), 1, &(10u32, "hello".to_string())).unwrap();
        transport::send_frame(&conn, &Frame::reliable(ChannelId::RPC, call))
            .await
            .expect("send rpc");
        let resp_frame = transport::recv_reliable(&conn)
            .await
            .expect("recv rpc resp");
        let (rpc_val, _): (u32, usize) =
            bincode::serde::decode_from_slice(&resp_frame.payload, bincode::config::standard())
                .unwrap();

        // 3. send an unreliable datagram.
        transport::send_frame(&conn, &Frame::unreliable(ChannelId::RPC, vec![7, 7, 7]))
            .await
            .expect("send datagram");

        // Give the datagram a moment, then close.
        tokio::time::sleep(Duration::from_millis(50)).await;
        conn.close(0u32.into(), b"done");
        endpoint.wait_idle().await;
        (received, rpc_val)
    };

    let (dg_payload, (received, rpc_val)) = tokio::join!(server_fut, client_fut);

    // 1. transform replication: received == sent, all 100.
    assert_eq!(received.transforms.len(), 100);
    assert_eq!(received, sent_snapshot);
    assert_eq!(
        received.transforms.keys().copied().collect::<Vec<_>>(),
        sent_snapshot.transforms.keys().copied().collect::<Vec<_>>()
    );
    // Not a trivial default map.
    assert_ne!(received.transforms, BTreeMap::new());

    // 2. RPC round-trip: 10 + len("hello") == 15.
    assert_eq!(rpc_val, 15);

    // 3. unreliable datagram arrived on loopback.
    assert_eq!(dg_payload, vec![7, 7, 7]);
}
