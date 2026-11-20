use std::{collections::BTreeMap, time::Instant};

#[derive(Default, serde::Serialize)]
struct Stage {
    calls: usize,
    wall_ms: f64,
    rss_after_max_kib: Option<u64>,
    process_hwm_after_kib: Option<u64>,
}

pub(crate) struct Profile {
    start: Option<Instant>,
    sample_rss: bool,
    stages: BTreeMap<&'static str, Stage>,
}

impl Profile {
    pub(crate) fn new() -> Self {
        let mode = if cfg!(not(target_arch = "wasm32")) {
            std::env::var("GLIDE_PROFILE").ok()
        } else {
            None
        };
        Self {
            start: matches!(mode.as_deref(), Some("1" | "rss")).then(Instant::now),
            sample_rss: mode.as_deref() == Some("rss"),
            stages: BTreeMap::new(),
        }
    }

    #[inline]
    pub(crate) fn start(&self) -> Option<Instant> {
        self.start.map(|_| Instant::now())
    }

    #[inline]
    pub(crate) fn end(&mut self, name: &'static str, start: Option<Instant>) {
        if let Some(start) = start {
            let stage = self.stages.entry(name).or_default();
            stage.calls += 1;
            stage.wall_ms += start.elapsed().as_secs_f64() * 1000.0;
            if self.sample_rss {
                let (rss, high_water) = process_memory();
                stage.rss_after_max_kib = stage.rss_after_max_kib.max(rss);
                stage.process_hwm_after_kib = stage.process_hwm_after_kib.max(high_water);
            }
        }
    }

    pub(crate) fn report(&self, prompts: &[&[u32]]) {
        if let Some(start) = self.start {
            let (rss, high_water) = process_memory();
            eprintln!(
                "GLIDE_PROFILE {}",
                serde_json::json!({
                    "prompt_lengths": prompts.iter().map(|row| row.len()).collect::<Vec<_>>(),
                    "wall_ms": start.elapsed().as_secs_f64() * 1000.0,
                    "stages": self.stages,
                    "rss_after_kib": rss,
                    "process_hwm_kib": high_water,
                })
            );
        }
    }
}

fn process_memory() -> (Option<u64>, Option<u64>) {
    if !cfg!(target_os = "linux") {
        return (None, None);
    }
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    let value = |key: &str| {
        status.lines().find_map(|line| {
            line.strip_prefix(key)?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
    };
    (value("VmRSS:"), value("VmHWM:"))
}
