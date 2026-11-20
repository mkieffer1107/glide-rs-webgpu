//! Memory-mapped quantized and floating-point weights.
use crate::bf16;
use crate::quant::{tiled_weight, TileQuant, BLOCK_COLS, TILE_ROWS};
use anyhow::{anyhow, bail, Context, Result};
use half::f16;
#[cfg(not(target_arch="wasm32"))]
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(not(target_arch="wasm32"))]
use std::{fs::File, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: usize,
    pub nbytes: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Index {
    pub version: u32,
    pub tensors: HashMap<String, TensorMeta>,
}

enum Storage {
    #[cfg(not(target_arch="wasm32"))]
    Mapped(Mmap),
    Owned(Vec<u8>),
}
impl Storage {
    fn bytes(&self) -> &[u8] { match self {
        #[cfg(not(target_arch="wasm32"))]
        Self::Mapped(m) => m,
        Self::Owned(v) => v
    } }
}

pub struct Weights { storage: Storage, pub index: Index }
impl Weights {
    #[cfg(not(target_arch="wasm32"))]
    pub fn open(dir: &Path) -> Result<Self> {
        let index: Index = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
        let file = File::open(dir.join("weights.bin"))?;
        // SAFETY: weights.bin is immutable while this process reads it. Do not overwrite in place.
        let storage = Storage::Mapped(unsafe { Mmap::map(&file)? });
        Self::new(index, storage)
    }
    pub fn from_bytes(index_json: &str, bytes: Vec<u8>) -> Result<Self> {
        Self::new(serde_json::from_str(index_json)?, Storage::Owned(bytes))
    }
    fn new(index: Index, storage: Storage) -> Result<Self> {
        if index.version != 1 { bail!("unsupported weight index format"); }
        for (name, t) in &index.tensors {
            if t.offset.checked_add(t.nbytes).map_or(true, |end| end > storage.bytes().len()) {
                bail!("out of range tensor {name}");
            }
            if t.shape.is_empty() || t.shape.iter().any(|&s| s == 0) { bail!("invalid shape: {name}"); }
            let count = t.shape.iter().try_fold(1usize, |acc,&s| acc.checked_mul(s)).context("shape overflow")?;
            let want = match t.dtype.as_str() {
                "f32" => count.checked_mul(4),
                "f16" | "bf16" => count.checked_mul(2),
                "bf16x16" if t.shape.len() == 2 => bf16::packed_size(t.shape[0], t.shape[1]),
                "q4" | "q4x16" | "q4nx16" | "q6x16" | "q8x16" if t.shape.len() == 2 => {
                    let group_bytes = if let Some(quant) = TileQuant::from_dtype(&t.dtype) {
                        if t.shape[0] % TILE_ROWS != 0 {
                            bail!("row count must be a multiple of {TILE_ROWS} for {name}");
                        }
                        quant.block_bytes() / TILE_ROWS
                    } else {
                        18
                    };
                    t.shape[1].div_ceil(BLOCK_COLS).checked_mul(group_bytes)
                        .and_then(|x| x.checked_mul(t.shape[0]))
                },
                _ => return Err(anyhow!("unsupported dtype for {name}: {}", t.dtype)),
            }.context("tensor size overflow")?;
            if t.nbytes != want { bail!("wrong byte length for {name}: {} != {want}",t.nbytes); }
        }
        Ok(Self { index, storage })
    }
    pub fn has(&self, name: &str) -> bool { self.index.tensors.contains_key(name) }
    pub fn get(&self, name: &str) -> Result<Tensor<'_>> {
        let meta = self.index.tensors.get(name).ok_or_else(|| anyhow!("missing tensor: {name}"))?;
        let data = &self.storage.bytes()[meta.offset..meta.offset+meta.nbytes];
        Ok(Tensor { meta, data })
    }
}

pub struct Tensor<'a> { meta: &'a TensorMeta, data: &'a [u8] }
impl<'a> Tensor<'a> {
    pub fn shape(&self) -> &[usize] { &self.meta.shape }
    pub(crate) fn dtype(&self) -> &str { &self.meta.dtype }
    pub fn at(&self, idx: usize) -> f32 {
        match self.meta.dtype.as_str() {
            "f32" => f32::from_le_bytes(self.data[idx*4..idx*4+4].try_into().unwrap()),
            "f16" => f16::from_bits(u16::from_le_bytes(self.data[idx*2..idx*2+2].try_into().unwrap())).to_f32(),
            "bf16" => half::bf16::from_le_bytes(self.data[idx*2..idx*2+2].try_into().unwrap()).to_f32(),
            "bf16x16" => {
                let cols = self.meta.shape[1];
                bf16::weight(self.data, cols, idx / cols, idx % cols)
            },
            "q4" => {
                let cols = self.meta.shape[1];
                let row = idx / cols;
                let col = idx % cols;
                let blocks = (cols + 31)/32;
                let start = (row*blocks + col/32)*18;
                let scale = f16::from_bits(u16::from_le_bytes(self.data[start..start+2].try_into().unwrap())).to_f32();
                let b = self.data[start+2+(col%32)/2];
                let n = if col%2==0 { b&15 } else { b>>4 };
                (n as i32 - 8) as f32 * scale
            },
            "q4x16" | "q4nx16" | "q6x16" | "q8x16" => {
                let quant = TileQuant::from_dtype(&self.meta.dtype).unwrap();
                let cols = self.meta.shape[1];
                let row = idx / cols;
                let col = idx % cols;
                let group = row / TILE_ROWS * cols.div_ceil(BLOCK_COLS) + col / BLOCK_COLS;
                let block_bytes = quant.block_bytes();
                let offset = group * block_bytes;
                tiled_weight(&self.data[offset..offset + block_bytes], row % TILE_ROWS, col % BLOCK_COLS, quant)
            },
            _ => unreachable!(),
        }
    }
    pub fn vector(&self) -> Vec<f32> {
        (0..self.meta.shape.iter().product()).map(|i| self.at(i)).collect()
    }
    pub fn row(&self, row: usize) -> Vec<f32> {
        let cols = *self.meta.shape.last().unwrap();
        (0..cols).map(|i|self.at(row*cols+i)).collect()
    }
    #[inline]
    fn dot_row(&self, row: usize, input: &[f32]) -> f32 {
        let cols = self.meta.shape[1];
        debug_assert_eq!(cols, input.len());
        match self.meta.dtype.as_str() {
            "q4" => {
                let blocks = (cols+31)/32;
                let start = row*blocks*18;
                #[cfg(target_arch="x86_64")]
                if std::arch::is_x86_feature_detected!("avx2") {
                    // Identical packed Q4 format on AVX2 and portable WASM paths.
                    return unsafe { dot_q4_avx2(&self.data[start..start+blocks*18], input) };
                }
                let mut sum = 0f32;
                for b in 0..blocks {
                    let offset = start + b*18;
                    let scale = f16::from_bits(u16::from_le_bytes([self.data[offset],self.data[offset+1]])).to_f32();
                    let end = (cols - b*32).min(32);
                    let mut acc = 0f32;
                    for j in 0..end {
                        let packed = self.data[offset+2+j/2];
                        let q = if j&1==0 { packed&15 } else { packed>>4 };
                        acc += (q as i32 - 8) as f32 * input[b*32+j];
                    }
                    sum += scale*acc;
                }
                sum
            },
            "bf16x16" | "bf16" => {
                let mut sum = 0.0;
                for (j, &x) in input.iter().enumerate() {
                    sum += self.at(row * cols + j) * bf16::round(x);
                }
                bf16::round(sum)
            },
            "q4x16" | "q4nx16" | "q6x16" | "q8x16" | "f16" | "f32" => {
                let mut sum = 0f32;
                for j in 0..cols { sum += self.at(row*cols+j)*input[j]; }
                sum
            },
            _ => unreachable!()
        }
    }
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>> {
        if self.meta.shape.len()!=2 || self.meta.shape[1]!=input.len() {
            bail!("matvec shape mismatch: {:?} with {}",self.meta.shape,input.len());
        }
        let rows = self.meta.shape[0];
        let mut out = vec![0.; rows];
        #[cfg(not(target_arch="wasm32"))]
        {
            use rayon::prelude::*;
            if rows >= 512 {
                out.par_iter_mut().enumerate().for_each(|(i,y)| *y=self.dot_row(i,input));
            } else {
                for (i,y) in out.iter_mut().enumerate() { *y=self.dot_row(i,input); }
            }
        }
        #[cfg(target_arch="wasm32")]
        for (i,y) in out.iter_mut().enumerate() { *y=self.dot_row(i,input); }
        Ok(out)
    }
    pub fn matmul_tokens(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if self.meta.shape.len() != 2 || x.iter().any(|token| token.len() != self.meta.shape[1]) {
            bail!("matrix shape mismatch");
        }
        #[cfg(target_arch = "x86_64")]
        if self.meta.dtype == "bf16x16" {
            if let Some(result) = bf16::matmul(self.data, self.meta.shape[0], self.meta.shape[1], x) {
                return Ok(result);
            }
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(quant) = TileQuant::from_dtype(&self.meta.dtype) {
            if let Some(result) = crate::quant::matmul(quant, self.data, self.meta.shape[0], self.meta.shape[1], x) {
                return Ok(result);
            }
        }
        #[cfg(not(target_arch="wasm32"))]
        {
            use rayon::prelude::*;
            if x.len()>=8 {
                // Parallelize over tokens; avoids creating an enormous activation matrix.
                return Ok(x.par_iter().map(|token| self.matvec_serial(token)).collect());
            }
        }
        Ok(x.iter().map(|token| self.matvec_serial(token)).collect())
    }
    fn matvec_serial(&self, x: &[f32]) -> Vec<f32> {
        (0..self.meta.shape[0]).map(|r| self.dot_row(r,x)).collect()
    }
}


/// Decode 32 Q4 values into four AVX2 lanes, multiply FP32 activations.
/// The Python converter stores adjacent values in the low and high nibbles.
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn dot_q4_avx2(packed: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let blocks = (x.len()+31)/32;
    let mut total = _mm256_setzero_ps();
    let bias = _mm_set1_epi8(8);
    let mask = _mm_set1_epi8(15);
    for b in 0..blocks {
        let pos=b*18;
        let scale=f16::from_bits(u16::from_le_bytes([packed[pos], packed[pos+1]])).to_f32();
        let bytes=_mm_loadu_si128(packed.as_ptr().add(pos+2) as *const __m128i);
        let lo=_mm_and_si128(bytes,mask);
        let hi=_mm_and_si128(_mm_srli_epi16(bytes,4),mask);
        let first=_mm_sub_epi8(_mm_unpacklo_epi8(lo,hi),bias);
        let second=_mm_sub_epi8(_mm_unpackhi_epi8(lo,hi),bias);
        let input_offset=b*32;
        let mut tail=[0f32;32];
        let input_ptr = if input_offset+32<=x.len() {
            x.as_ptr().add(input_offset)
        } else {
            tail[..x.len()-input_offset].copy_from_slice(&x[input_offset..]);
            tail.as_ptr()
        };
        let scale_vec=_mm256_set1_ps(scale);
        for part in 0..4 {
            let vec8=match part {
                0=>_mm256_cvtepi8_epi32(first),
                1=>_mm256_cvtepi8_epi32(_mm_srli_si128(first,8)),
                2=>_mm256_cvtepi8_epi32(second),
                _=>_mm256_cvtepi8_epi32(_mm_srli_si128(second,8)),
            };
            let w=_mm256_mul_ps(_mm256_cvtepi32_ps(vec8),scale_vec);
            let xv=_mm256_loadu_ps(input_ptr.add(part*8));
            total=_mm256_add_ps(total,_mm256_mul_ps(w,xv));
        }
    }
    let mut lanes=[0f32;8];
    _mm256_storeu_ps(lanes.as_mut_ptr(),total);
    lanes.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_matmul_keeps_odd_shapes_and_token_tiles() {
        let (rows, cols) = (55_usize, 1027_usize);
        let pairs = cols.div_ceil(2);
        let mut data = vec![0; bf16::packed_size(rows, cols).unwrap()];
        let weight = |r: usize, c: usize| ((r * 3 + c * 7) as i32 % 17 - 8) as f32 * 0.125;
        for row in 0..rows {
            for col in 0..cols {
                let pair = row / 16 * pairs + col / 2;
                let offset = pair * 64 + (row % 16 * 2 + col % 2) * 2;
                data[offset..offset + 2].copy_from_slice(&half::bf16::from_f32(weight(row, col)).to_le_bytes());
            }
        }
        let meta = TensorMeta {
            shape: vec![rows, cols], dtype: "bf16x16".into(), offset: 0, nbytes: data.len(),
        };
        let matrix = Tensor { meta: &meta, data: &data };
        let tokens: Vec<Vec<f32>> = (0..131).map(|t| {
            (0..cols).map(|c| ((t * 5 + c * 11) as i32 % 19 - 9) as f32 * 0.125).collect()
        }).collect();
        for input in [&tokens[..1], &tokens[..]] {
            let actual = matrix.matmul_tokens(input).unwrap();
            for (activation, result) in input.iter().zip(actual) {
                for (row, value) in result.into_iter().enumerate() {
                    let expected = activation.iter().enumerate().map(|(c, x)| weight(row, c) * x).sum();
                    assert_eq!(value, bf16::round(expected));
                }
            }
        }
    }

    #[test]
    fn tiled_matmul_with_partial_token_and_column_tiles() {
        let (rows, cols) = (32_usize, 47_usize);
        let groups = cols.div_ceil(BLOCK_COLS);
        for (kind, dtype, bias) in [
            (TileQuant::Q4, "q4x16", 8),
            (TileQuant::Q4Codebook, "q4nx16", 8),
            (TileQuant::Q6, "q6x16", 32),
            (TileQuant::Q8, "q8x16", 128),
        ] {
            let tile_bytes = kind.block_bytes();
            let mut data = vec![0; rows / TILE_ROWS * groups * tile_bytes];
            let quant = |row, col| (row * 7 + col * 3) % (2 * bias);
            for r in 0..rows {
                for g in 0..groups {
                    let base = (r / TILE_ROWS * groups + g) * tile_bytes;
                    let lane = r % TILE_ROWS;
                    let scale = f16::from_f32(0.25 * (g + 1) as f32).to_le_bytes();
                    data[base + lane * 2..base + lane * 2 + 2].copy_from_slice(&scale);
                    for c in 0..BLOCK_COLS {
                        let value = quant(r, g * BLOCK_COLS + c) as u8;
                        let row_data = base + 2 * TILE_ROWS + lane * 4 + c % 4;
                        match kind {
                            TileQuant::Q4 | TileQuant::Q4Codebook => {
                                data[row_data + c / 8 * 64] |=
                                    if c % 8 < 4 { value } else { value << 4 };
                            }
                            TileQuant::Q6 => {
                                let part = base + 2 * TILE_ROWS + c / 8 * 96;
                                let half = c % 8 / 4;
                                data[part + lane * 4 + c % 4] |= (value & 15) << (half * 4);
                                data[part + 64 + half * 16 + lane] |= (value >> 4) << (2 * (c % 4));
                            }
                            TileQuant::Q8 => data[row_data + c / 4 * 64] = value,
                        }
                    }
                }
            }
            let meta = TensorMeta {
                shape: vec![rows, cols],
                dtype: dtype.into(),
                offset: 0,
                nbytes: data.len(),
            };
            let tokens: Vec<Vec<f32>> = (0..9)
                .map(|t| {
                    (0..cols)
                        .map(|c| match (t, c % BLOCK_COLS) {
                            (8, _) => 0.0,
                            (_, 0) => 127.0,
                            _ => ((t * 11 + c * 23) as i32 % 255 - 127) as f32,
                        })
                        .collect()
                })
                .collect();
            let matrix = Tensor {
                meta: &meta,
                data: &data,
            };
            let projected = matrix.matmul_tokens(&tokens).unwrap();
            for (activation, output) in tokens.iter().zip(projected) {
                for (r, actual) in output.into_iter().enumerate() {
                    let expected = activation
                        .iter()
                        .enumerate()
                        .map(|(c, a)| {
                            let code = quant(r, c);
                            let weight = match kind {
                                TileQuant::Q4Codebook => i32::from(crate::quant::Q4_CODEBOOK[code]),
                                _ => code as i32 - bias as i32,
                            };
                            weight as f32 * 0.25 * (c / BLOCK_COLS + 1) as f32 * a
                        })
                        .sum::<f32>();
                    assert_eq!(actual, expected);
                    assert_eq!(matrix.dot_row(r, activation), expected);
                }
            }
        }
    }

    #[test]
    fn q4_values_and_dot() {
        let mut tensors=HashMap::new();
        tensors.insert("w".into(), TensorMeta { shape:vec![1,32], dtype:"q4".into(), offset:0, nbytes:18 });
        let mut bytes=vec![0u8;18];
        bytes[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        for i in 0..16 { bytes[i+2]=0x97; } // -1 in low nibble, +1 in high nibble
        let store=Weights::new(Index{version:1,tensors}, Storage::Owned(bytes)).unwrap();
        let w=store.get("w").unwrap();
        assert!((w.at(0)+0.5).abs()<1e-5);
        assert!((w.at(1)-0.5).abs()<1e-5);
        assert!(w.matvec(&vec![1f32;32]).unwrap()[0].abs()<1e-5);
        #[cfg(target_arch="x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            let probe=(0..32).map(|i|i as f32/13.).collect::<Vec<_>>();
            let accelerated=w.matvec(&probe).unwrap()[0];
            let reference=(0..32).map(|i|w.at(i)*probe[i]).sum::<f32>();
            assert!((accelerated-reference).abs()<1e-4);
        }
    }
}
