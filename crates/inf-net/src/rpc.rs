//! A minimal RPC skeleton: numbered remote procedures with bincode-encoded args.
//!
//! An RPC is identified by a stable [`RpcId`] (`u16`). The caller encodes its
//! typed args into an [`RpcCall`] frame payload ([`encode_call`]); the receiver
//! dispatches it through an [`RpcRegistry`] onto a registered handler, which
//! decodes the args and optionally returns a bincode-encoded response.
//!
//! This is the *wire + dispatch* half only — it is transport-agnostic (send an
//! [`RpcCall`] payload as a reliable [`Frame`](crate::Frame) on the RPC channel).
//! Request/response correlation (matching a reply to a call) is carried by
//! [`RpcCall::seq`] so a caller can pair responses; wiring that into a live
//! request/await future is a Ring-2 concern left to the transport layer.

use std::collections::HashMap;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// A stable RPC identifier. Games assign one per remote procedure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RpcId(pub u16);

/// A wire RPC call: which procedure, a correlation seq, and bincode arg bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RpcCall {
    pub id: RpcId,
    /// Caller-assigned correlation id (echoed in a response), `0` = fire-and-forget.
    pub seq: u32,
    pub args: Vec<u8>,
}

impl RpcCall {
    pub fn encode(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("an RpcCall is always encodable")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, crate::NetError> {
        let (call, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(call)
    }
}

/// Build the frame payload for a typed RPC call.
pub fn encode_call<T: Serialize>(
    id: RpcId,
    seq: u32,
    args: &T,
) -> Result<Vec<u8>, crate::NetError> {
    let args = bincode::serde::encode_to_vec(args, bincode::config::standard())?;
    Ok(RpcCall { id, seq, args }.encode())
}

/// Decode typed args out of a received [`RpcCall`].
pub fn decode_args<T: DeserializeOwned>(call: &RpcCall) -> Result<T, crate::NetError> {
    let (args, _) = bincode::serde::decode_from_slice(&call.args, bincode::config::standard())?;
    Ok(args)
}

type Handler = Box<dyn Fn(&RpcCall) -> Result<Option<Vec<u8>>, crate::NetError> + Send + Sync>;

/// A registry mapping [`RpcId`]s to handlers. Dispatch decodes the call and
/// invokes the matching handler, returning its optional response bytes.
#[derive(Default)]
pub struct RpcRegistry {
    handlers: HashMap<RpcId, Handler>,
}

impl RpcRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a typed handler. `f` receives the decoded args and returns an
    /// optional typed response.
    pub fn register<A, R, F>(&mut self, id: RpcId, f: F)
    where
        A: DeserializeOwned,
        R: Serialize,
        F: Fn(A) -> Option<R> + Send + Sync + 'static,
    {
        let handler: Handler = Box::new(move |call: &RpcCall| {
            let args: A = decode_args(call)?;
            match f(args) {
                Some(resp) => {
                    let bytes = bincode::serde::encode_to_vec(&resp, bincode::config::standard())?;
                    Ok(Some(bytes))
                }
                None => Ok(None),
            }
        });
        self.handlers.insert(id, handler);
    }

    /// Whether an id is registered.
    pub fn contains(&self, id: RpcId) -> bool {
        self.handlers.contains_key(&id)
    }

    /// Dispatch a received call frame payload. Returns:
    /// - `Ok(Some(bytes))` — handler ran and produced a response,
    /// - `Ok(None)` — handler ran with no response (fire-and-forget), or no
    ///   handler registered for the id,
    /// - `Err` — malformed call / arg-decode failure.
    pub fn dispatch(&self, payload: &[u8]) -> Result<Option<Vec<u8>>, crate::NetError> {
        let call = RpcCall::decode(payload)?;
        match self.handlers.get(&call.id) {
            Some(h) => h(&call),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn call_dispatches_to_handler_with_decoded_args() {
        let seen = Arc::new(AtomicU32::new(0));
        let seen2 = seen.clone();
        let mut reg = RpcRegistry::new();
        // RPC 7: takes (u32, String), returns u32.
        reg.register::<(u32, String), u32, _>(RpcId(7), move |(n, s)| {
            seen2.store(n, Ordering::SeqCst);
            Some(n + s.len() as u32)
        });

        let payload = encode_call(RpcId(7), 42, &(10u32, "hello".to_string())).unwrap();
        let resp = reg
            .dispatch(&payload)
            .unwrap()
            .expect("expected a response");
        let (val, _): (u32, usize) =
            bincode::serde::decode_from_slice(&resp, bincode::config::standard()).unwrap();
        assert_eq!(val, 15);
        assert_eq!(seen.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn correlation_seq_survives() {
        let call = RpcCall::decode(&encode_call(RpcId(1), 99, &()).unwrap()).unwrap();
        assert_eq!(call.seq, 99);
        assert_eq!(call.id, RpcId(1));
    }

    #[test]
    fn unregistered_id_is_none_not_error() {
        let reg = RpcRegistry::new();
        let payload = encode_call(RpcId(3), 0, &5u8).unwrap();
        assert!(reg.dispatch(&payload).unwrap().is_none());
    }

    #[test]
    fn fire_and_forget_returns_none() {
        let mut reg = RpcRegistry::new();
        reg.register::<u8, (), _>(RpcId(2), |_v| None);
        let payload = encode_call(RpcId(2), 0, &1u8).unwrap();
        assert!(reg.dispatch(&payload).unwrap().is_none());
    }
}
