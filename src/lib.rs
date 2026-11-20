#[cfg(target_arch = "x86_64")]
mod attention;
mod bf16;
mod delta;
pub mod model;
mod profile;
mod quant;
pub mod weights;
pub use model::GlideModel;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    pub struct WasmGlide { inner: GlideModel }
    #[wasm_bindgen]
    impl WasmGlide {
        #[wasm_bindgen(constructor)]
        pub fn new(config: &str, decision_config: &str, index: &str, weight_bytes: Vec<u8>) -> Result<WasmGlide, JsValue> {
            GlideModel::from_parts(config, decision_config, index, weight_bytes)
                .map(|inner| Self {inner})
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
        pub fn predict(&self, token_ids: Vec<u32>, option_count: usize) -> Result<JsValue, JsValue> {
            let scores = self.inner.probabilities(&token_ids, option_count)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&scores).map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }
}
