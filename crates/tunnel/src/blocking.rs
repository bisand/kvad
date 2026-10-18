//! The tunnel for a caller with no async runtime: the CLI, whose HTTP client
//! (`ureq`) is blocking.
//!
//! A [`Tunnel`] owns a small runtime of its own that holds the connection.
//! [`Tunnel::open`] hands back one end of a Unix socket pair, and a task on
//! that runtime copies between the other end and a stream through the
//! tunnel. So the CLI's requests go through the tunnel without a listening
//! port anywhere on the machine: nothing else on it can use the connection.

use super::{Address, Client, Key, Path, Res, Voucher};

/// One pairing request to `to`, as the key in `key` (made on first use),
/// and the server's answer.
pub fn pair(key: &FsPath, to: &Address, request: &[u8], voucher: Option<&Voucher>) -> Res<Vec<u8>> {
    let (key, _) = Key::load(key)?;
    pair_as(&key, to, request, voucher)
}

/// The same, as `key`.
pub fn pair_as(key: &Key, to: &Address, request: &[u8], voucher: Option<&Voucher>) -> Res<Vec<u8>> {
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(|e| format!("starting the tunnel: {e}"))?;
    runtime.block_on(Client::pair(key, to, request, voucher))
}
use std::os::unix::net::UnixStream;
use std::path::Path as FsPath;

pub struct Tunnel {
    runtime: tokio::runtime::Runtime,
    client: std::sync::Arc<Client>,
    address: Address,
}

impl Tunnel {
    /// Connects to `to` with the key in `key` (made on first use), showing
    /// relays `voucher` if there is one.
    pub fn connect(key: &FsPath, to: &Address, voucher: Option<&Voucher>) -> Res<Tunnel> {
        let (key, _) = Key::load(key)?;
        Tunnel::connect_as(&key, to, voucher)
    }

    /// The same, as `key`.
    pub fn connect_as(key: &Key, to: &Address, voucher: Option<&Voucher>) -> Res<Tunnel> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("kvad-tunnel")
            .enable_all()
            .build()
            .map_err(|e| format!("starting the tunnel: {e}"))?;
        let client = runtime.block_on(Client::connect_vouched(key, to, voucher))?;
        Ok(Tunnel { runtime, client: std::sync::Arc::new(client), address: to.clone() })
    }

    /// A new stream through the tunnel, as a socket this process alone holds.
    pub fn open(&self) -> Res<UnixStream> {
        let stream = self.runtime.block_on(self.client.open())?;
        let (ours, theirs) = UnixStream::pair().map_err(|e| format!("a socket pair: {e}"))?;
        theirs.set_nonblocking(true).map_err(|e| e.to_string())?;
        self.runtime.spawn(async move {
            let Ok(mut theirs) = tokio::net::UnixStream::from_std(theirs) else { return };
            let mut stream = stream;
            let _ = tokio::io::copy_bidirectional(&mut theirs, &mut stream).await;
        });
        Ok(ours)
    }

    pub fn path(&self) -> Option<Path> {
        self.client.path()
    }

    pub fn address(&self) -> &Address {
        &self.address
    }
}

impl Drop for Tunnel {
    fn drop(&mut self) {
        // Says goodbye rather than leave the server to time the connection
        // out; bounded, so that a dead link cannot hold the CLI's exit.
        let client = self.client.clone();
        let _ = self.runtime.block_on(async move { tokio::time::timeout(std::time::Duration::from_secs(1), client.close()).await });
    }
}
