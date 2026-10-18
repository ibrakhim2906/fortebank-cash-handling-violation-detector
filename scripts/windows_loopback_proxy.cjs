/* Windows-only localhost bridge to the existing WSL localhost service.
 * No keys or footage are logged, placed in arguments, or written to disk.
 */
"use strict";

const http = require("node:http");
const { spawn } = require("node:child_process");
const { once } = require("node:events");

const MAX_BODY = 64 * 1024 * 1024 + 128 * 1024;
const MAX_RESULT = 8 * 1024 * 1024;
const MAX_HEADER = 16 * 1024;
const MAX_ACTIVE = 6;
const TIMEOUT_MS = 60_000;
const RESPONSE_HEADERS = new Set([
  "content-type", "cache-control", "content-security-policy", "referrer-policy",
  "x-content-type-options", "x-frame-options", "retry-after",
]);

function options(argv) {
  const result = { distro: "Ubuntu-24.04", root: "/home/ibrak/datathon-kit", port: 8001, service: "video" };
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!["--distro", "--root", "--port", "--service"].includes(name) || value === undefined) {
      throw new Error("Invalid bridge arguments");
    }
    result[name.slice(2)] = value;
  }
  result.port = Number(result.port);
  if (!/^[A-Za-z0-9._-]{1,80}$/.test(result.distro)
      || !result.root.startsWith("/") || /[\r\n\0]/.test(result.root)
      || !["video", "review"].includes(result.service)
      || !Number.isInteger(result.port) || result.port < 1024 || result.port > 65535) {
    throw new Error("Invalid bridge configuration");
  }
  return result;
}

function allowed(method, path, service = "video") {
  if (service === "review") {
    return (method === "GET" && (["/", "/api/review/tasks", "/api/review/summary",
      "/api/review/transfer/tasks", "/api/review/transfer/summary",
      "/api/review/pretrained/tasks", "/api/review/pretrained/summary"].includes(path)
      || /^\/api\/review\/(?:(?:transfer|pretrained)\/)?media\/[0-9a-f]{64}$/.test(path)))
      || (method === "POST" && ["/api/review/labels", "/api/review/transfer/labels"].includes(path));
  }
  return (method === "GET" && (path === "/" || /^\/api\/v1\/jobs\/[0-9a-f]+$/.test(path)))
    || (method === "POST" && path === "/api/v1/analyze");
}

function safeHeaders(headers) {
  const result = {};
  if (!headers || typeof headers !== "object" || Array.isArray(headers)) {
    throw new Error("Invalid upstream headers");
  }
  for (const [rawName, value] of Object.entries(headers)) {
    const name = rawName.toLowerCase();
    if (RESPONSE_HEADERS.has(name) && typeof value === "string"
        && value.length <= 8192 && !/[\r\n\0]/.test(value)) {
      result[name] = value;
    }
  }
  return result;
}

function createBridge(configuration) {
  let active = 0;
  const server = http.createServer({ maxHeaderSize: MAX_HEADER }, (request, response) => {
    function reject(status, detail) {
      response.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      response.end(JSON.stringify({ detail }));
      request.resume();
    }
    if (!allowed(request.method, request.url, configuration.service)) {
      reject(404, "Route not found");
      return;
    }
    if (active >= MAX_ACTIVE) {
      reject(503, "Local bridge is busy. Try again shortly.");
      return;
    }
    const declared = request.headers["content-length"];
    if (declared !== undefined && (!/^\d+$/.test(declared) || Number(declared) > MAX_BODY)) {
      reject(413, "Upload is too large");
      return;
    }
    if (request.method === "GET" && (Number(declared || 0) !== 0 || request.headers["transfer-encoding"])) {
      reject(400, "GET requests cannot contain a body");
      return;
    }

    active += 1;
    let finished = false;
    let child = null;
    let requestSize = 0;
    let requestChunks = [];
    let responseChunks = [];
    let responseSize = 0;
    let headerBytes = Buffer.alloc(0);
    let upstream = null;

    function finish(status, headers, body) {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      active -= 1;
      if (child && child.exitCode === null && child.signalCode === null) child.kill();
      requestChunks = [];
      responseChunks = [];
      if (!response.destroyed) {
        response.writeHead(status, { ...headers, "Content-Length": body.length });
        response.end(body);
      }
    }
    function fail(status = 502, detail = "The local video service could not be reached.") {
      finish(status, { "Content-Type": "application/json", "Cache-Control": "no-store" },
        Buffer.from(JSON.stringify({ detail })));
      request.resume();
    }
    const timer = setTimeout(() => fail(502, "The local video request timed out."), TIMEOUT_MS);

    request.on("aborted", () => fail());
    request.on("error", () => fail());
    response.on("close", () => {
      if (!finished) fail();
    });
    request.on("data", chunk => {
      if (finished) return;
      requestSize += chunk.length;
      if (requestSize > MAX_BODY) {
        fail(413, "Upload is too large");
        return;
      }
      requestChunks.push(chunk);
    });
    request.on("end", () => {
      if (finished) return;
      if (declared !== undefined && Number(declared) !== requestSize) {
        fail(400, "Incomplete request body");
        return;
      }
      const headers = {};
      for (const name of ["content-type", "x-api-key"]) {
        if (typeof request.headers[name] === "string") headers[name] = request.headers[name];
      }
      const instruction = Buffer.from(JSON.stringify({
        method: request.method, path: request.url, headers, content_length: requestSize,
      }) + "\n");
      if (instruction.length > MAX_HEADER) {
        fail(400, "Request headers are too large");
        return;
      }
      try {
        child = spawn("C:\\Windows\\System32\\wsl.exe", [
          "-d", configuration.distro, "--cd", configuration.root, "--exec", "/usr/bin/env",
          "UV_CACHE_DIR=/tmp/datathon-uv-cache", "/home/ibrak/.local/bin/uv", "run", "--offline",
          "python", "-m", "scripts.loopback_request",
          "--service", configuration.service || "video",
        ], { stdio: ["pipe", "pipe", "ignore"], windowsHide: true });
      } catch {
        fail();
        return;
      }
      child.on("error", () => fail());
      child.stdin.on("error", () => fail());
      child.stdout.on("error", () => fail());
      child.stdout.on("data", chunk => {
        if (finished) return;
        if (upstream === null) {
          headerBytes = Buffer.concat([headerBytes, chunk]);
          const newline = headerBytes.indexOf(10);
          if (newline === -1) {
            if (headerBytes.length > MAX_HEADER) fail();
            return;
          }
          if (newline > MAX_HEADER) {
            fail();
            return;
          }
          try {
            const parsed = JSON.parse(headerBytes.subarray(0, newline).toString("utf8"));
            if (!Number.isInteger(parsed.status) || parsed.status < 200 || parsed.status > 599) {
              throw new Error("Invalid upstream status");
            }
            upstream = { status: parsed.status, headers: safeHeaders(parsed.headers) };
          } catch {
            fail();
            return;
          }
          chunk = headerBytes.subarray(newline + 1);
          headerBytes = Buffer.alloc(0);
        }
        responseSize += chunk.length;
        if (responseSize > (configuration.service === "review" ? MAX_BODY : MAX_RESULT)) {
          fail();
          return;
        }
        responseChunks.push(chunk);
      });
      child.on("close", code => {
        if (finished) return;
        if (code !== 0 || upstream === null) {
          fail();
          return;
        }
        finish(upstream.status, upstream.headers, Buffer.concat(responseChunks, responseSize));
      });
      (async () => {
        if (!child.stdin.write(instruction)) await once(child.stdin, "drain");
        for (const chunk of requestChunks) {
          if (finished) return;
          if (!child.stdin.write(chunk)) await once(child.stdin, "drain");
        }
        requestChunks = [];
        child.stdin.end();
      })().catch(() => fail());
    });
  });
  server.requestTimeout = TIMEOUT_MS;
  server.headersTimeout = 10_000;
  server.keepAliveTimeout = 5_000;
  return server;
}

if (require.main === module) {
  let configuration;
  try {
    configuration = options(process.argv.slice(2));
  } catch {
    process.stderr.write("Invalid local bridge configuration.\n");
    process.exit(1);
  }
  const server = createBridge(configuration);
  server.on("error", () => {
    process.stderr.write("The Windows localhost bridge could not start.\n");
    process.exitCode = 1;
  });
  server.listen(configuration.port, "127.0.0.1", () => {
    process.stdout.write(`Windows local UI: http://127.0.0.1:${configuration.port}/\n`);
  });
}

module.exports = { createBridge, options, allowed };
