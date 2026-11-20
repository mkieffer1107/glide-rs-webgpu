import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const mime = {
    ".html": "text/html; charset=utf-8", ".mjs": "text/javascript", ".js": "text/javascript",
    ".json": "application/json", ".wasm": "application/wasm", ".bin": "application/octet-stream",
};

// A local server for the example and its benchmark. Serve only known assets.
export function createGlideServer(modelDir, wasmDir) {
    const files = new Map([
        ["/", resolve(root, "examples/glide-04b.html")],
        ["/examples/glide-04b.mjs", resolve(root, "examples/glide-04b.mjs")],
        ["/examples/glide-04b.tokens.json", resolve(root, "examples/glide-04b.tokens.json")],
    ]);
    for (const name of ["encoder.mjs", "encoder-kernels.mjs", "weights.mjs", "projection.mjs"]) {
        files.set(`/webgpu/${name}`, resolve(root, "webgpu", name));
    }
    for (const name of ["webgpu.json", "config.json", "decision_config.json", "weights.bin"]) {
        files.set(`/model/${name}`, resolve(modelDir, name));
    }
    if (wasmDir) {
        for (const name of ["glide_cpu.js", "glide_cpu_bg.wasm"]) {
            files.set(`/pkg/${name}`, resolve(wasmDir, name));
        }
    }
    return createServer((req, res) => {
        if (req.method !== "GET" && req.method !== "HEAD") return res.writeHead(405).end();
        const file = files.get(new URL(req.url, "http://localhost").pathname);
        if (!file) return res.writeHead(404).end();
        const stat = statSync(file, {throwIfNoEntry: false});
        if (!stat?.isFile()) return res.writeHead(404).end();
        const headers = {
            "Content-Type": mime[file.slice(file.lastIndexOf("."))],
            "Content-Length": stat.size,
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        };
        let start = 0;
        let end = stat.size - 1;
        if (req.headers.range !== undefined) {
            const match = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range);
            start = Number(match?.[1]);
            end = Number(match?.[2]);
            if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || end < start || end >= stat.size) {
                return res.writeHead(416, {"Content-Range": `bytes */${stat.size}`}).end();
            }
            headers["Content-Range"] = `bytes ${start}-${end}/${stat.size}`;
            headers["Content-Length"] = end - start + 1;
        }
        res.writeHead(headers["Content-Range"] ? 206 : 200, headers);
        if (req.method === "HEAD" || stat.size === 0) return res.end();
        const stream = createReadStream(file, {start, end});
        res.on("close", () => stream.destroy());
        stream.on("error", error => res.destroy(error));
        stream.pipe(res);
    });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    const [modelDir = "model-04b-q8", port = "8080"] = process.argv.slice(2);
    for (const name of ["webgpu.json", "config.json", "decision_config.json", "weights.bin"]) {
        if (!statSync(resolve(modelDir, name), {throwIfNoEntry: false})?.isFile()) {
            throw new Error(`Missing ${resolve(modelDir, name)}`);
        }
    }
    createGlideServer(modelDir).listen(Number(port), "127.0.0.1", function () {
        console.log(`GLiDE WebGPU is ready at http://127.0.0.1:${this.address().port}/`);
    });
}
