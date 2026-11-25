use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context};
use serde_json::Value;

struct Tensor {
    dtype: String,
    shape: Vec<u64>,
    range: (u64, u64),
}

pub struct MtpFile {
    path: PathBuf,
    file: File,
    tensors: HashMap<String, Tensor>,
}

impl MtpFile {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path).with_context(|| path.display().to_string())?;
        let mut length = [0u8; 8];
        file.read_exact_at(&mut length, 0)?;
        let length = u64::from_le_bytes(length);
        ensure!(length < 1 << 20, "{}: invalid MTP header", path.display());
        let mut header = vec![0u8; length as usize];
        file.read_exact_at(&mut header, 8)?;
        let header: Value = serde_json::from_slice(&header)?;
        let mut tensors = HashMap::new();
        for (name, value) in header.as_object().context("MTP header object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = value["dtype"].as_str().context("MTP tensor dtype")?;
            ensure!(dtype == "BF16", "{name}: expected BF16, got {dtype}");
            let shape: Vec<u64> = value["shape"]
                .as_array()
                .context("MTP tensor shape")?
                .iter()
                .map(|v| v.as_u64().context("MTP tensor dimension"))
                .collect::<anyhow::Result<_>>()?;
            let offsets = value["data_offsets"].as_array().context("MTP offsets")?;
            let start = offsets[0].as_u64().context("MTP offset")? + 8 + length;
            let end = offsets[1].as_u64().context("MTP offset")? + 8 + length;
            ensure!(end >= start, "{name}: invalid offsets");
            tensors.insert(
                name.clone(),
                Tensor {
                    dtype: dtype.into(),
                    shape,
                    range: (start, end),
                },
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            file,
            tensors,
        })
    }

    fn read_range(&self, name: &str, start: u64, bytes: usize) -> anyhow::Result<Vec<u8>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{}: missing {name}", self.path.display()))?;
        ensure!(
            start >= tensor.range.0 && start + bytes as u64 <= tensor.range.1,
            "{name}: slice outside tensor"
        );
        let mut output = vec![0u8; bytes];
        self.file.read_exact_at(&mut output, start)?;
        Ok(output)
    }

    pub fn whole(&self, name: &str, shape: &[u64]) -> anyhow::Result<Vec<u8>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{}: missing {name}", self.path.display()))?;
        ensure!(
            tensor.shape == shape,
            "{name}: got {:?}, expected {shape:?}",
            tensor.shape
        );
        ensure!(tensor.dtype == "BF16");
        self.read_range(
            name,
            tensor.range.0,
            (tensor.range.1 - tensor.range.0) as usize,
        )
    }

    pub fn rows(
        &self,
        name: &str,
        shape: &[u64],
        start: u64,
        rows: u64,
    ) -> anyhow::Result<Vec<u8>> {
        let columns = shape[1..].iter().product::<u64>();
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{}: missing {name}", self.path.display()))?;
        ensure!(tensor.shape == shape && start + rows <= shape[0]);
        self.read_range(
            name,
            tensor.range.0 + start * columns * 2,
            (rows * columns * 2) as usize,
        )
    }

    pub fn columns(
        &self,
        name: &str,
        rows: u64,
        columns: u64,
        start: u64,
        width: u64,
    ) -> anyhow::Result<Vec<u8>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{}: missing {name}", self.path.display()))?;
        ensure!(tensor.shape == [rows, columns] && start + width <= columns);
        let mut output = Vec::with_capacity((rows * width * 2) as usize);
        for row in 0..rows {
            output.extend(self.read_range(
                name,
                tensor.range.0 + (row * columns + start) * 2,
                (width * 2) as usize,
            )?);
        }
        Ok(output)
    }
}
