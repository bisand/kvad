//! ureq's connections, made of streams through the tunnel: each one end of
//! a Unix socket pair that [`crate::blocking::Tunnel`] copies to a QUIC
//! stream, so that no port is opened anywhere for a client to talk through.
//! Shared by the CLI and Kvad for iPhone, both of which speak HTTP with ureq.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use ureq::unversioned::transport::{Buffers, ConnectionDetails, LazyBuffers, NextTimeout, Transport};

pub struct Connector(pub Arc<crate::blocking::Tunnel>);

impl std::fmt::Debug for Connector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Connector({})", self.0.address().label())
    }
}

impl ureq::unversioned::transport::Connector<()> for Connector {
    type Out = Socket;

    fn connect(&self, details: &ConnectionDetails, _: Option<()>) -> Result<Option<Socket>, ureq::Error> {
        let stream = self.0.open().map_err(|e| ureq::Error::Io(std::io::Error::other(e)))?;
        let buffers = LazyBuffers::new(details.config.input_buffer_size(), details.config.output_buffer_size());
        Ok(Some(Socket { stream, buffers }))
    }
}

pub struct Socket {
    stream: UnixStream,
    buffers: LazyBuffers,
}

impl std::fmt::Debug for Socket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Socket(tunnel)")
    }
}

fn timed_out(e: &std::io::Error) -> bool {
    matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)
}

impl Transport for Socket {
    fn buffers(&mut self) -> &mut dyn Buffers {
        &mut self.buffers
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), ureq::Error> {
        self.stream.set_write_timeout(timeout.not_zero().map(|t| *t))?;
        let output = &self.buffers.output()[..amount];
        match self.stream.write_all(output) {
            Err(e) if timed_out(&e) => Err(ureq::Error::Timeout(timeout.reason)),
            other => Ok(other?),
        }
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
        self.stream.set_read_timeout(timeout.not_zero().map(|t| *t))?;
        let input = self.buffers.input_append_buf();
        let amount = match self.stream.read(input) {
            Err(e) if timed_out(&e) => return Err(ureq::Error::Timeout(timeout.reason)),
            other => other?,
        };
        self.buffers.input_appended(amount);
        Ok(amount > 0)
    }

    fn is_open(&mut self) -> bool {
        // A peek would say whether the other end has gone; a pooled
        // stream the server closed is found out at the next request
        // instead, which ureq retries.
        true
    }
}

/// An agent with `config` whose every connection is a stream through
/// `tunnel`. Request URLs still need a host; one that is an address, such as
/// `http://127.0.0.1:5823`, is never looked up and never dialled.
pub fn agent(tunnel: Arc<crate::blocking::Tunnel>, config: ureq::config::Config) -> ureq::Agent {
    ureq::Agent::with_parts(config, Connector(tunnel), ureq::unversioned::resolver::DefaultResolver::default())
}
