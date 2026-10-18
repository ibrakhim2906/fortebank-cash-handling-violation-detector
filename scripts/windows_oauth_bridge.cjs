"use strict";
// Windows loopback only. OAuth URL/query/code stays in pipes, never logs or arguments.
const http = require("node:http");
const { spawn } = require("node:child_process");

const port = Number(process.argv[2]);
const callbackPath = process.argv[3];
if (!Number.isInteger(port) || port < 1024 || port > 65535 || !/^\/callback\/[A-Za-z0-9_-]+$/.test(callbackPath || "")) {
  throw new Error("Invalid callback bridge configuration");
}
const server = http.createServer({ maxHeaderSize: 8192 }, (request, response) => {
  if (request.method === "GET" && request.url === "/bridge-health") {
    response.writeHead(204, { "Cache-Control": "no-store" }); response.end(); return;
  }
  if (request.method !== "GET" || request.url.length > 8192 || request.url.split("?", 1)[0] !== callbackPath) {
    response.writeHead(404); response.end(); return;
  }
  const child = spawn("C:\\Windows\\System32\\wsl.exe", [
    "-d", "Ubuntu-24.04", "--cd", "/home/ibrak/datathon-kit", "--exec", "/usr/bin/env",
    "UV_CACHE_DIR=/tmp/datathon-uv-cache", "/home/ibrak/.local/bin/uv", "run", "--offline", "python",
    "-m", "scripts.oauth_callback_proxy", "--port", String(port), "--path", callbackPath,
  ], { stdio: ["pipe", "pipe", "ignore"], windowsHide: true });
  let buffer = "";
  const timer = setTimeout(() => { child.kill(); response.writeHead(502); response.end("Local callback timed out"); }, 20000);
  child.stdout.on("data", part => { buffer += part; if (buffer.length > 100000) child.kill(); });
  child.on("error", () => { clearTimeout(timer); if (!response.writableEnded) { response.writeHead(502); response.end("Local bridge unavailable"); } });
  child.on("close", code => {
    clearTimeout(timer);
    if (response.writableEnded) return;
    try {
      if (code !== 0) throw new Error("callback failed");
      const result = JSON.parse(buffer);
      response.writeHead(result.status, { "Content-Type": result.type, "Cache-Control": "no-store", "Referrer-Policy": "no-referrer" });
      response.end(Buffer.from(result.body, "base64"));
    } catch { response.writeHead(502); response.end("Local callback forwarding failed"); }
  });
  child.stdin.on("error", () => {});
  child.stdin.end(JSON.stringify({ path: request.url }) + "\n");
});
server.on("error", () => { process.stderr.write("Windows callback listener unavailable\n"); process.exitCode = 1; });
server.listen(port, "127.0.0.1", () => process.stdout.write("Windows OAuth callback bridge ready\n"));
