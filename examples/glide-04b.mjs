import { Glide04bWebGPU } from "../webgpu/encoder.mjs";

const load = document.querySelector("#load");
const run = document.querySelector("#run");
const status = document.querySelector("#status");
const progress = document.querySelector("#progress");
const result = document.querySelector("#result");
let model;

load.onclick = async () => {
    load.disabled = true;
    status.textContent = "Preparing WebGPU.";
    let device;
    try {
        const adapter = await navigator.gpu?.requestAdapter({powerPreference: "high-performance"});
        if (!adapter) throw new Error("No WebGPU adapter found. Use a supported browser on localhost or HTTPS.");
        device = await adapter.requestDevice();
        progress.hidden = false;
        const loaded = await Glide04bWebGPU.load("/model/webgpu.json", {
            adapter, device,
            onProgress(bytes, total) {
                progress.value = bytes / total;
                status.textContent = `Loaded ${Math.round(bytes / 1e6)} of ${Math.round(total / 1e6)} MB.`;
            },
        });
        model = loaded;
        status.textContent = adapter.info.isFallbackAdapter
            ? "Model loaded on a software WebGPU adapter."
            : "Model loaded on the GPU.";
        run.disabled = false;
        device.lost.then(() => {
            if (model !== loaded) return;
            model = undefined;
            run.disabled = true;
            load.disabled = false;
            status.textContent = "WebGPU device lost. Reload the model to continue.";
            void loaded.destroy();
        });
    } catch (error) {
        device?.destroy();
        status.textContent = error.message;
        load.disabled = false;
    } finally {
        progress.hidden = true;
    }
};

run.onclick = async () => {
    run.disabled = true;
    result.hidden = true;
    status.textContent = "Running inference.";
    try {
        const file = document.querySelector("#request").files[0];
        const request = file ? JSON.parse(await file.text())
            : await (await fetch("/examples/glide-04b.tokens.json")).json();
        const entries = Object.entries(request.questions);
        if (!entries.length) throw new Error("The request needs at least one question.");
        const start = performance.now();
        const logits = await model.predictLogitsBatch(entries.map(([, q]) => q.input_ids));
        const ms = performance.now() - start;
        const answers = Object.fromEntries(entries.map(([name, question], i) => {
            const scores = model.probabilities(logits[i], question.labels.length);
            const best = scores.indexOf(Math.max(...scores));
            return [name, {
                choice: question.labels[best],
                probabilities: Object.fromEntries(question.labels.map((label, j) => [label, scores[j]])),
            }];
        }));
        status.textContent = `Inference finished in ${ms.toFixed(1)} ms.`;
        result.textContent = JSON.stringify(answers, null, 2);
        result.hidden = false;
    } catch (error) {
        status.textContent = error.message;
    } finally {
        run.disabled = !model;
    }
};
