import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { proxyBackend } from "../src/lib/backend.ts";

const originalFetch = globalThis.fetch;
const originalUrl = process.env.RAG_API_BASE_URL;
const originalToken = process.env.RAG_REGISTRY_TOKEN;

afterEach(() => {
  globalThis.fetch = originalFetch;
  for (const [key, value] of [["RAG_API_BASE_URL", originalUrl], ["RAG_REGISTRY_TOKEN", originalToken]]) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
});

function configure() {
  process.env.RAG_API_BASE_URL = "http://127.0.0.1:4318/";
  process.env.RAG_REGISTRY_TOKEN = "test-server-credential";
}

test("forwards filters and server credentials and disables caching", async () => {
  configure();
  globalThis.fetch = async (url, init) => {
    assert.equal(String(url), "http://127.0.0.1:4318/api/v1/traces?environment=staging&limit=1");
    assert.equal(init.headers.Authorization, "Bearer test-server-credential");
    assert.equal(init.cache, "no-store");
    assert.equal(init.redirect, "error");
    return Response.json({ items: [], next_cursor: null });
  };
  const response = await proxyBackend(new Request("http://localhost/api/traces?environment=staging&limit=1"), ["traces"]);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.deepEqual(await response.json(), { items: [], next_cursor: null });
});

test("registration preserves backend conflicts and validation messages", async () => {
  configure();
  const body = { service_name: "my-rag", framework: "custom", environments: ["staging"] };
  globalThis.fetch = async (url, init) => {
    assert.equal(String(url), "http://127.0.0.1:4318/api/v1/apps");
    assert.equal(init.method, "POST");
    assert.deepEqual(JSON.parse(init.body), body);
    return Response.json({ code: "service_name_exists", message: "Service name already registered" }, { status: 409 });
  };
  const response = await proxyBackend(new Request("http://localhost/api/apps", { method: "POST", body: JSON.stringify(body) }), ["apps"]);
  assert.equal(response.status, 409);
  assert.equal((await response.json()).code, "service_name_exists");
});

test("rejects unknown routes and invalid JSON before contacting backend", async () => {
  configure();
  globalThis.fetch = async () => { throw new Error("must not fetch"); };
  assert.equal((await proxyBackend(new Request("http://localhost/api/private"), ["private"])).status, 404);
  assert.equal((await proxyBackend(new Request("http://localhost/api/traces", { method: "POST" }), ["traces"])).status, 405);
  assert.equal((await proxyBackend(new Request("http://localhost/api/apps", { method: "POST", body: "{" }), ["apps"])).status, 400);
});

test("missing credentials and unavailable backend return actionable errors", async () => {
  delete process.env.RAG_REGISTRY_TOKEN;
  assert.equal((await proxyBackend(new Request("http://localhost/api/apps"), ["apps"])).status, 503);
  configure();
  globalThis.fetch = async () => { throw new Error("connection refused"); };
  const response = await proxyBackend(new Request("http://localhost/api/apps"), ["apps"]);
  assert.equal(response.status, 502);
  const body = await response.text();
  assert.match(body, /Cannot reach the Python backend/);
  assert.ok(!body.includes(process.env.RAG_REGISTRY_TOKEN));
});


test("registration preserves field errors for the form", async () => {
  configure();
  const error = {
    code: "invalid_registration",
    message: "Invalid app registration",
    field_errors: { service_name: "Use lowercase letters, for example my-rag-app." },
  };
  globalThis.fetch = async () => Response.json(error, { status: 422 });
  const response = await proxyBackend(new Request("http://localhost/api/apps", {
    method: "POST",
    body: JSON.stringify({ service_name: "My App", framework: "custom", environments: ["staging"] }),
  }), ["apps"]);
  assert.equal(response.status, 422);
  assert.deepEqual(await response.json(), error);
});


test("diagnosis requests reach only the scoped diagnosis endpoint", async () => {
  configure();
  globalThis.fetch = async (url, init) => {
    assert.equal(String(url), "http://127.0.0.1:4318/api/v1/traces/execution-id/diagnosis");
    assert.equal(init.method, "POST");
    assert.equal(init.headers.Authorization, "Bearer test-server-credential");
    assert.equal(init.body, undefined);
    return Response.json({ state: "queued" }, { status: 202 });
  };
  const response = await proxyBackend(new Request("http://localhost/api/traces/execution-id/diagnosis", {
    method: "POST",
  }), ["traces", "execution-id", "diagnosis"]);
  assert.equal(response.status, 202);
  assert.equal((await response.json()).state, "queued");
});
