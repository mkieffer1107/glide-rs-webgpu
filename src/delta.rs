//! FP32 recurrent update for a single Gated DeltaNet head.

pub(crate) fn update(
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    value: &mut [f32],
    output: &mut [f32],
    decay: f32,
    beta: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if state.len() == 128 * 128
        && key.len() == 128
        && query.len() == 128
        && value.len() == 128
        && output.len() == 128
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        // SAFETY: The slices have the model's checked 128-by-128 head
        // dimensions, and the CPU feature is available on this process.
        unsafe { update_avx512(state, key, query, value, output, decay, beta) };
        return;
    }

    let width = value.len();
    for (row, &k) in state.chunks_exact_mut(width).zip(key) {
        for (previous, error) in row.iter_mut().zip(&mut *value) {
            *previous *= decay;
            *error -= *previous * k;
        }
    }
    for error in &mut *value {
        *error *= beta;
    }
    for ((row, &k), &q) in state.chunks_exact_mut(width).zip(key).zip(query) {
        for ((previous, &error), out) in row.iter_mut().zip(&*value).zip(&mut *output) {
            *previous += k * error;
            *out += *previous * q;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn update_avx512(
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    value: &[f32],
    output: &mut [f32],
    decay: f32,
    beta: f32,
) {
    use std::arch::x86_64::*;

    // Each 128-by-64 slab holds two consecutive passes' working set.
    // Recomputing the decayed value preserves the scalar operation order
    // while avoiding an intermediate write to the recurrent state.
    let decay = _mm512_set1_ps(decay);
    let beta = _mm512_set1_ps(beta);
    for (part, state) in state.chunks_exact_mut(128 * 64).enumerate() {
        let mut delta: [__m512; 4] =
            std::array::from_fn(|j| _mm512_loadu_ps(value.as_ptr().add(part * 64 + j * 16)));
        for (row, &k) in state.chunks_exact(64).zip(key) {
            let k = _mm512_set1_ps(k);
            for (j, d) in delta.iter_mut().enumerate() {
                let previous = _mm512_loadu_ps(row.as_ptr().add(j * 16));
                *d = _mm512_sub_ps(*d, _mm512_mul_ps(_mm512_mul_ps(previous, decay), k));
            }
        }
        for d in &mut delta {
            *d = _mm512_mul_ps(*d, beta);
        }
        let mut result = [_mm512_setzero_ps(); 4];
        for ((row, &k), &q) in state.chunks_exact_mut(64).zip(key).zip(query) {
            let k = _mm512_set1_ps(k);
            let q = _mm512_set1_ps(q);
            for (j, out) in result.iter_mut().enumerate() {
                let lane = row.as_mut_ptr().add(j * 16);
                let old = _mm512_mul_ps(decay, _mm512_loadu_ps(lane));
                let next = _mm512_add_ps(old, _mm512_mul_ps(k, delta[j]));
                *out = _mm512_add_ps(*out, _mm512_mul_ps(q, next));
                _mm512_storeu_ps(lane, next);
            }
        }
        for (j, sum) in result.into_iter().enumerate() {
            _mm512_storeu_ps(output.as_mut_ptr().add(part * 64 + j * 16), sum);
        }
    }
}
