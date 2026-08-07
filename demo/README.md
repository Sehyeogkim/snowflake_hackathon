# MAVIS interactive demo

This local-first web demo visualizes the MAVIS runtime on a real CCTV clip.
The browser extracts 16 observations and measures pixel motion locally. During
analysis, MAVIS retains one high-salience frame from each temporal third so the
visible evidence preserves context, transition, and outcome.

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:3000`, choose `public/demo-video.mp4`, and select
**Analyze with MAVIS**. The bundled clip is the held-out dataset sample recorded
in `public/demo-analysis.json`; only that filename and duration pair displays
the verified VLM safety classification. The result and token-efficiency card is
not rendered until analysis completes. Other uploads still receive real local
frame extraction and motion-based evidence selection, but the UI intentionally
withholds a VLM safety class because no browser credential is available.

## Validation

```bash
npm test
```

The test command builds the Cloudflare-compatible vinext app and verifies the
upload-first placeholder, all 16 evidence placeholders, and the honest fallback
for unverified videos.

The headline token reduction comes from the checked-in eight-clip provider
pilot: 71,254 dense-baseline tokens versus 10,063 optimized tokens, or 85.877%.
See `../results/provider-pilot-8.json` for the complete quality caveats.
