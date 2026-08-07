import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the MAVIS upload-first workflow", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /MAVIS · Memory-Aware Visual Inference/);
  assert.match(html, /Upload a CCTV video/);
  assert.match(html, /Choose video/);
  assert.match(html, /Waiting for video/);
  assert.doesNotMatch(html, /src="\/demo-video\.mp4"/);
  assert.equal((html.match(/class="frame-placeholder"/g) ?? []).length, 16);
});

test("keeps unverified local analysis honest", async () => {
  const response = await render();
  const html = await response.text();

  assert.doesNotMatch(html, /Verified VLM analysis/);
  assert.doesNotMatch(html, /Local motion analysis · 3 observations/);
  assert.doesNotMatch(html, /Measured token efficiency/);
  assert.doesNotMatch(html, /Safe walkway violation detected/);
});
