//! **A WAV that is written as it is rendered** (wave VEH3e) — the file half of
//! the render-to-file door ([`crate::AudioEngine::offline`]).
//!
//! Stereo, 16-bit, at the engine's rate. The RIFF sizes are patched every
//! [`WavCapture::patch`] and on drop, so a capture whose process is killed
//! mid-drive (a demo script ending a PIE session) is still a WAV every player
//! opens, up to the last patch.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// **A stereo 16-bit WAV being written.**
pub struct WavCapture {
    file: File,
    frames: u64,
}

impl WavCapture {
    /// Create `path` and write a header with zero sizes.
    pub fn create(path: &Path, sample_rate: u32) -> std::io::Result<Self> {
        let mut file = File::create(path)?;
        let (channels, bits) = (2u16, 16u16);
        let block = channels * bits / 8;
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&36u32.to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&channels.to_le_bytes());
        h.extend_from_slice(&sample_rate.to_le_bytes());
        h.extend_from_slice(&(sample_rate * u32::from(block)).to_le_bytes());
        h.extend_from_slice(&block.to_le_bytes());
        h.extend_from_slice(&bits.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&0u32.to_le_bytes());
        file.write_all(&h)?;
        Ok(Self { file, frames: 0 })
    }

    /// Append interleaved `L R` samples in `[-1, 1]` (clamped).
    pub fn append(&mut self, interleaved: &[f32]) -> std::io::Result<()> {
        let mut out = Vec::with_capacity(interleaved.len() * 2);
        for s in interleaved {
            let v = (f64::from(*s) * 32_767.0)
                .round()
                .clamp(-32_768.0, 32_767.0) as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
        self.file.write_all(&out)?;
        self.frames += (interleaved.len() / 2) as u64;
        Ok(())
    }

    /// How many stereo frames have been written.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Write the RIFF and data sizes for what has been appended so far.
    pub fn patch(&mut self) -> std::io::Result<()> {
        let data = (self.frames * 4).min(u64::from(u32::MAX - 36)) as u32;
        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&(36 + data).to_le_bytes())?;
        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&data.to_le_bytes())?;
        self.file.seek(SeekFrom::End(0))?;
        self.file.flush()
    }
}

impl Drop for WavCapture {
    fn drop(&mut self) {
        let _ = self.patch();
    }
}
