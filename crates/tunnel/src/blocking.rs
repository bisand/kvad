//! The tunnel for a caller with no async runtime: the CLI, whose HTTP client
//! (`ureq`) is blocking.
//!
//! A [`Tunnel`] owns a small runtime of its own that holds the connection.
//! [`Tunnel::open`] hands back one end of a Unix socket pair, and a task on
//! that runtime copies between the other end and a stream through the
//! tunnel. So the CLI's requests go through the tunnel without a listening
//! port anywhere on the machine: nothing else on it can use the connection.
//!
//! A connection that has ended, as one does when the server is restarted or
//! the link was down for longer than QUIC waits, is dialled again by the
//! next `open`. A phone holds its `Tunnel` for as long as it is connected, so
//! without that it would keep a dead connection until someone connected
//! again by hand.

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
use std::sync::{Arc, Mutex};

pub struct Tunnel {
    runtime: tokio::runtime::Runtime,
    /// The connection now: replaced when it has ended.
    client: Mutex<Arc<Client>>,
    /// Held by whoever is dialling again, so that two streams opened at once
    /// on a dead connection make one new one.
    dialling: Mutex<()>,
    address: Address,
    /// What the next connection is dialled with.
    key: Key,
    voucher: Option<Voucher>,
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
        Ok(Tunnel {
            runtime,
            client: Mutex::new(Arc::new(client)),
            dialling: Mutex::new(()),
            address: to.clone(),
            key: Key::from_bytes(&key.to_bytes()),
            voucher: voucher.cloned(),
        })
    }

    fn client(&self) -> Arc<Client> {
        self.client.lock().unwrap().clone()
    }

    /// A new stream through the tunnel, as a socket this process alone holds.
    /// On a connection that has ended, the server is dialled again first.
    pub fn open(&self) -> Res<UnixStream> {
        let client = self.client();
        let stream = match self.runtime.block_on(client.open()) {
            Ok(stream) => stream,
            Err(_) if client.is_closed() => {
                let fresh = self.dial_again(&client)?;
                self.runtime.block_on(fresh.open())?
            }
            Err(e) => return Err(e),
        };
        let (ours, theirs) = UnixStream::pair().map_err(|e| format!("a socket pair: {e}"))?;
        theirs.set_nonblocking(true).map_err(|e| e.to_string())?;
        self.runtime.spawn(async move {
            let Ok(mut theirs) = tokio::net::UnixStream::from_std(theirs) else { return };
            let mut stream = stream;
            let _ = tokio::io::copy_bidirectional(&mut theirs, &mut stream).await;
        });
        Ok(ours)
    }

    /// A new connection in place of `ended`, unless another thread has made
    /// one meanwhile.
    fn dial_again(&self, ended: &Arc<Client>) -> Res<Arc<Client>> {
        let _one = self.dialling.lock().unwrap();
        let now = self.client();
        if !Arc::ptr_eq(&now, ended) && !now.is_closed() {
            return Ok(now);
        }
        tracing::info!("the tunnel to {} ended; dialling again", self.address.label());
        let fresh = Arc::new(self.runtime.block_on(Client::connect_vouched(&self.key, &self.address, self.voucher.as_ref()))?);
        *self.client.lock().unwrap() = fresh.clone();
        Ok(fresh)
    }

    pub fn path(&self) -> Option<Path> {
        self.client().path()
    }

    /// See [`Client::network_changed`]. Returns at once; the new search runs
    /// on the tunnel's own threads.
    pub fn network_changed(&self) {
        let client = self.client();
        self.runtime.spawn(async move { client.network_changed().await });
    }

    pub fn address(&self) -> &Address {
        &self.address
    }
}

impl Drop for Tunnel {
    fn drop(&mut self) {
        // Says goodbye rather than leave the server to time the connection
        // out; bounded, so that a dead link cannot hold the CLI's exit.
        let client = self.client();
        let _ = self.runtime.block_on(async move { tokio::time::timeout(std::time::Duration::from_secs(1), client.close()).await });
    }
}
