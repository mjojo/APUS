use std::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Wire packet for distributed LAN pipeline inference (Sprint 24 `apus-net`).
/// Compresses hidden-state activation vectors into FP8/INT8 (4x smaller than FP32)
/// to minimize LAN latency when chaining a Desktop PC, Laptop, and Mac Mini.
#[derive(Debug, Clone, PartialEq)]
pub struct Fp8ActivationPacket {
    pub ticket_id: u64,
    pub next_layer: u16,
    pub scale: f32,
    pub quantized_activations: Vec<i8>,
}

impl Fp8ActivationPacket {
    /// Compresses an FP32 hidden state slice into an FP8/INT8 LAN packet
    pub fn from_hidden_state(ticket_id: u64, next_layer: u16, hidden: &[f32]) -> Self {
        let max_abs = hidden.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
        let scale = max_abs / 127.0;
        let quantized_activations = hidden
            .iter()
            .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Self {
            ticket_id,
            next_layer,
            scale,
            quantized_activations,
        }
    }

    /// Reconstructs the FP32 hidden state vector from the received LAN packet
    pub fn into_hidden_state(&self) -> Vec<f32> {
        self.quantized_activations
            .iter()
            .map(|&q| (q as f32) * self.scale)
            .collect()
    }

    /// Serializes the packet into a compact binary wire frame
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let len = self.quantized_activations.len() as u32;
        let mut buf = Vec::with_capacity(18 + len as usize);
        buf.extend_from_slice(&self.ticket_id.to_le_bytes());
        buf.extend_from_slice(&self.next_layer.to_le_bytes());
        buf.extend_from_slice(&self.scale.to_le_bytes());
        buf.extend_from_slice(&len.to_le_bytes());
        for &q in &self.quantized_activations {
            buf.push(q as u8);
        }
        buf
    }

    /// Deserializes a packet from a binary wire frame
    pub fn from_wire_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 18 {
            return None;
        }
        let ticket_id = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let next_layer = u16::from_le_bytes(bytes[8..10].try_into().ok()?);
        let scale = f32::from_le_bytes(bytes[10..14].try_into().ok()?);
        let len = u32::from_le_bytes(bytes[14..18].try_into().ok()?) as usize;
        if bytes.len() < 18 + len {
            return None;
        }
        let quantized_activations = bytes[18..18 + len].iter().map(|&b| b as i8).collect();
        Some(Self {
            ticket_id,
            next_layer,
            scale,
            quantized_activations,
        })
    }
}

/// Asynchronous TCP RPC node for LAN Pipeline Parallelism
pub struct LanRpcNode;

impl LanRpcNode {
    /// Sends an FP8-compressed activation packet to the next node in the LAN ring/pipeline
    /// and awaits the transformed activation packet back.
    pub async fn send_and_receive(
        endpoint: &str,
        packet: &Fp8ActivationPacket,
    ) -> io::Result<Fp8ActivationPacket> {
        let mut stream = TcpStream::connect(endpoint).await?;
        let raw = packet.to_wire_bytes();
        stream.write_u32_le(raw.len() as u32).await?;
        stream.write_all(&raw).await?;
        stream.flush().await?;

        let resp_len = stream.read_u32_le().await? as usize;
        let mut resp_buf = vec![0u8; resp_len];
        stream.read_exact(&mut resp_buf).await?;

        Fp8ActivationPacket::from_wire_bytes(&resp_buf)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid FP8 RPC response"))
    }

    /// Spawns a single-shot worker listener on `listener` that receives an activation packet,
    /// advances `next_layer` by `layers_processed`, and returns the updated FP8 packet.
    pub async fn serve_one_hop(
        listener: TcpListener,
        layers_processed: u16,
    ) -> io::Result<Fp8ActivationPacket> {
        let (mut socket, _) = listener.accept().await?;
        let req_len = socket.read_u32_le().await? as usize;
        let mut req_buf = vec![0u8; req_len];
        socket.read_exact(&mut req_buf).await?;

        let pkt = Fp8ActivationPacket::from_wire_bytes(&req_buf).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Malformed FP8 RPC packet")
        })?;

        let mut hidden = pkt.into_hidden_state();
        for v in &mut hidden {
            *v = (*v * 1.02).tanh();
        }
        let out_pkt = Fp8ActivationPacket::from_hidden_state(
            pkt.ticket_id,
            pkt.next_layer + layers_processed,
            &hidden,
        );
        let out_raw = out_pkt.to_wire_bytes();
        socket.write_u32_le(out_raw.len() as u32).await?;
        socket.write_all(&out_raw).await?;
        socket.flush().await?;
        Ok(out_pkt)
    }
}
