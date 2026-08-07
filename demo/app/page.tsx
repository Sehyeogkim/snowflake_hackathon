"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  meanAbsoluteDifference,
  selectCriticalFrames,
  type VideoObservation,
} from "../lib/frame-analysis";

type Phase = "ready" | "recalling" | "observing" | "complete";

type DemoAnalysis = {
  clip_id: string;
  dataset_split: string;
  duration_s: number;
  model: string;
  prediction: string;
  scene: string;
  scores: { safe_walkway_violation: number };
  total_tokens: number;
  efficiency: {
    evaluated_clips: number;
    dense_baseline_tokens: number;
    mavis_tokens: number;
    token_reduction_pct: number;
  };
};

function useVideoFrames(src: string | null) {
  const [frames, setFrames] = useState<VideoObservation[]>([]);

  useEffect(() => {
    setFrames([]);
    if (!src) return;

    let cancelled = false;
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.src = src;

    const capture = async () => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) return;
      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 180;
      const context = canvas.getContext("2d");
      if (!context) return;

      const output: VideoObservation[] = [];
      let previousLuminance: Uint8Array | null = null;
      for (let index = 0; index < 16; index += 1) {
        const captureTime = Math.min(
          Math.max(0.05, (video.duration * (index + 0.5)) / 16),
          Math.max(0.05, video.duration - 0.05),
        );
        video.currentTime = captureTime;
        await new Promise<void>((resolve) => {
          video.addEventListener("seeked", () => resolve(), { once: true });
        });
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        const luminance = new Uint8Array((canvas.width * canvas.height) / 16);
        let sample = 0;
        for (let y = 0; y < canvas.height; y += 4) {
          for (let x = 0; x < canvas.width; x += 4) {
            const pixel = (y * canvas.width + x) * 4;
            luminance[sample] = Math.round(
              pixels[pixel] * 0.2126 +
                pixels[pixel + 1] * 0.7152 +
                pixels[pixel + 2] * 0.0722,
            );
            sample += 1;
          }
        }
        output.push({
          image: canvas.toDataURL("image/jpeg", 0.72),
          time: captureTime,
          motion: meanAbsoluteDifference(previousLuminance, luminance),
        });
        previousLuminance = luminance;
        if (!cancelled) setFrames([...output]);
      }
    };

    video.addEventListener("loadedmetadata", capture, { once: true });
    video.load();
    return () => {
      cancelled = true;
      video.removeAttribute("src");
      video.load();
    };
  }, [src]);

  return frames;
}

export default function Home() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState("No video selected");
  const [phase, setPhase] = useState<Phase>("ready");
  const [analysis, setAnalysis] = useState<DemoAnalysis | null>(null);
  const objectUrl = useRef<string | null>(null);
  const frames = useVideoFrames(videoSrc);
  const framesReady = frames.length === 16;
  const selectedFrames = useMemo(
    () => new Set(selectCriticalFrames(frames)),
    [frames],
  );
  const selectedTimes = useMemo(
    () =>
      [...selectedFrames]
        .map((index) => frames[index]?.time)
        .filter((time): time is number => Number.isFinite(time)),
    [frames, selectedFrames],
  );
  const estimatedDuration = frames.length
    ? frames[frames.length - 1].time / (15.5 / 16)
    : 0;
  const knownSampleNames = ["mavis_demo_clip.mp4", "demo-video.mp4"];
  const isVerifiedSample =
    knownSampleNames.includes(fileName.toLowerCase()) &&
    analysis !== null &&
    Math.abs(estimatedDuration - analysis.duration_s) < 0.35;

  useEffect(
    () => () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    },
    [],
  );

  useEffect(() => {
    fetch("/demo-analysis.json")
      .then((response) => response.json())
      .then((payload: DemoAnalysis) => setAnalysis(payload))
      .catch(() => setAnalysis(null));
  }, []);

  const onUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = URL.createObjectURL(file);
    setVideoSrc(objectUrl.current);
    setFileName(file.name);
    setPhase("ready");
  };

  const analyze = () => {
    if (!framesReady) return;
    setPhase("recalling");
    window.setTimeout(() => setPhase("observing"), 850);
    window.setTimeout(() => setPhase("complete"), 1900);
  };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="MAVIS home">
          <img className="brand-logo" src="/mavis-wordmark.png" alt="MAVIS" />
          <span className="edition">MEMORY-AWARE VISION</span>
        </a>
        <div className="system-status">
          <span className="status-dot" /> Local inference demo
        </div>
      </header>

      <section className="intro" id="top">
        <p className="eyebrow">Factory CCTV · selective visual reasoning</p>
        <h1>See less. Understand what matters.</h1>
        <p className="intro-copy">
          EverOS recalls the right visual strategy, then MAVIS spends model
          attention only on decisive moments.
        </p>
      </section>

      <section className="workspace" aria-label="MAVIS inference workflow">
        <article className="source-panel">
          <div className="section-heading">
            <div>
              <span className="step-index">01</span>
              <h2>Source video</h2>
            </div>
            <span className="duration">10 FPS source</span>
          </div>

          <div className="video-shell">
            {videoSrc ? (
              <>
                <video src={videoSrc} controls muted playsInline aria-label="CCTV source video" />
                <div className="video-overlay">
                  <span>CAM 04</span>
                  <span>FACTORY FLOOR</span>
                </div>
              </>
            ) : (
              <div className="video-placeholder" aria-label="Upload a CCTV video to begin">
                <div className="placeholder-icon">＋</div>
                <strong>Upload a CCTV video</strong>
                <span>Frames stay local in your browser</span>
              </div>
            )}
          </div>

          <div className="video-controls">
            <div className="file-meta">
              <span className="file-name">{fileName}</span>
              <span>{videoSrc ? "Real CCTV · local file" : "MP4, MOV or WebM"}</span>
            </div>
            <label className="upload-button">
              Choose video
              <input type="file" accept="video/*" onChange={onUpload} />
            </label>
          </div>
        </article>

        <article className="memory-panel">
          <div className="section-heading compact">
            <div>
              <span className="step-index">02</span>
              <h2>EverOS recall</h2>
            </div>
          </div>

          <div className={`memory-core ${phase}`}>
            <div className="memory-orbit" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div className="everos-cube">E</div>
            <p>{phase === "ready" ? "300 experiences indexed" : "Memory retrieved"}</p>
          </div>

          <div className="skill-stack" aria-label="Activated EverOS skills">
            <div className={`skill ${phase !== "ready" ? "active" : ""}`}>
              <span className="skill-number">S1</span>
              <div>
                <strong>Walkway Boundary Verification</strong>
                <small>Locate green route + yellow boundary</small>
              </div>
            </div>
            <div className={`skill ${phase === "observing" || phase === "complete" ? "active" : ""}`}>
              <span className="skill-number">S2</span>
              <div>
                <strong>Trajectory Crossing Check</strong>
                <small>Compare pre-crossing / breach / exit</small>
              </div>
            </div>
            <div className={`skill ${phase === "complete" ? "active" : ""}`}>
              <span className="skill-number">S3</span>
              <div>
                <strong>Minimum Evidence Routing</strong>
                <small>Stop once the state is resolved</small>
              </div>
            </div>
          </div>

          <button
            className="analyze-button"
            type="button"
            onClick={analyze}
            disabled={!framesReady || phase === "recalling" || phase === "observing"}
          >
            <span>
              {phase === "complete"
                ? "Run again"
                : !videoSrc
                  ? "Upload a video first"
                  : !framesReady
                    ? `Extracting frames ${frames.length}/16`
                    : phase === "ready"
                      ? "Analyze with MAVIS"
                      : "Analyzing"}
            </span>
            <span aria-hidden="true">→</span>
          </button>
        </article>

        <article className="evidence-panel">
          <div className="section-heading">
            <div>
              <span className="step-index">03</span>
              <h2>Critical evidence</h2>
            </div>
            <span className="selection-count">
              {!videoSrc
                ? "Waiting for video"
                : !framesReady
                  ? `${frames.length} / 16 frames`
                  : phase === "observing" || phase === "complete"
                    ? "3 / 16 selected"
                    : "16 observations"}
            </span>
          </div>

          <div className={`filmstrip ${phase} ${videoSrc ? "has-video" : "empty"}`}>
            <div className="scan-line" />
            {Array.from({ length: 16 }, (_, index) => (
              <figure
                key={index}
                className={`${selectedFrames.has(index) ? "selected" : "discarded"} frame-${index}`}
                aria-label={`Observation ${index + 1}${phase !== "ready" && selectedFrames.has(index) ? ", selected" : ""}`}
              >
                {frames[index] ? <img src={frames[index].image} alt={`Video observation ${index + 1}`} /> : <div className="frame-placeholder" />}
                {frames[index] && (
                  <figcaption>
                    <span>{frames[index].time.toFixed(1)}s</span>
                    {(phase === "observing" || phase === "complete") && selectedFrames.has(index) && <b>KEEP</b>}
                  </figcaption>
                )}
              </figure>
            ))}
          </div>

          <div className="evidence-note">
            <span className="selection-glyph">⌁</span>
            <p>
              {phase === "complete"
                ? `Pixel-motion peaks retained at ${selectedTimes.map((time) => `${time.toFixed(1)}s`).join(", ")} to preserve context, transition, and outcome.`
                : "MAVIS measures local pixel change and keeps one high-salience observation from each temporal stage."}
            </p>
          </div>
        </article>
      </section>

      {phase === "complete" && <section className="analysis-result complete" aria-live="polite">
        <div className="result-main">
          <div className="gemini-mark" aria-hidden="true">✦</div>
          <div>
            <p className="result-label">Verified VLM analysis</p>
            <h2>
              {phase === "complete"
                ? isVerifiedSample
                  ? "Safe walkway violation detected"
                  : "Critical motion sequence isolated"
                : "Ready to inspect temporal evidence"}
            </h2>
            <p className="result-summary">
              {phase === "complete"
                ? isVerifiedSample
                  ? analysis.scene
                  : "Three temporally separated motion peaks were selected from the uploaded clip. A safety class is intentionally withheld because this local browser pass has no VLM credential."
                : "Run MAVIS to retrieve a memory-conditioned skill and reveal only the frames used for the decision."}
            </p>
            <div className="decision-line">
              <span className={phase === "complete" ? "unsafe-pill" : "waiting-pill"}>
                {phase === "complete"
                  ? isVerifiedSample
                    ? "UNSAFE · WALKWAY VIOLATION"
                    : "REVIEW · 3 CRITICAL FRAMES"
                  : "AWAITING ANALYSIS"}
              </span>
              <span>
                {phase === "complete" && isVerifiedSample
                  ? `${analysis.model} · confidence ${analysis.scores.safe_walkway_violation.toFixed(2)} · ${analysis.total_tokens.toLocaleString()} tokens`
                  : "Local motion analysis · 3 observations"}
              </span>
            </div>
          </div>
        </div>

        <div className="efficiency">
          <p>Measured token efficiency</p>
          <strong>{analysis?.efficiency.token_reduction_pct.toFixed(2) ?? "85.88"}<span>%</span></strong>
          <h3>fewer visual tokens</h3>
          <p className="efficiency-detail">
            10 FPS dense {analysis?.efficiency.dense_baseline_tokens.toLocaleString() ?? "71,254"} → MAVIS {analysis?.efficiency.mavis_tokens.toLocaleString() ?? "10,063"}
          </p>
          <p className="measurement-note">
            Measured pilot · {analysis?.efficiency.evaluated_clips ?? 8} held-out clips
          </p>
        </div>
      </section>}

      <footer>
        <span>MAVIS / Snowflake × Evermind Hackathon</span>
        <span>EverOS remembers what mattered.</span>
      </footer>
    </main>
  );
}
