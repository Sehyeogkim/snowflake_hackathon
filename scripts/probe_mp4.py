#!/usr/bin/env python3
"""Report duration / fps / frame count / resolution for MP4 clips.

Parses the MP4 box tree directly so it needs no ffmpeg or OpenCV — those are a
large download on a slow link, and all we want here is how many frames a clip
holds, which decides how many VLM calls a naive baseline would make.

  python3 scripts/probe_mp4.py data/raw/clips/test/*/*.mp4
"""
import struct
import sys
import glob
import os


def boxes(buf, start, end):
    """Yield (type, payload_start, payload_end) for each box in [start, end)."""
    off = start
    while off + 8 <= end:
        size, btype = struct.unpack(">I4s", buf[off:off + 8])
        head = 8
        if size == 1:  # 64-bit extended size
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            head = 16
        elif size == 0:
            size = end - off
        if size < head:
            break
        yield btype.decode("latin1"), off + head, off + size
        off += size


def find(buf, start, end, path):
    """Walk a slash-separated box path, returning (start, end) or None."""
    want, rest = path[0], path[1:]
    for btype, s, e in boxes(buf, start, end):
        if btype == want:
            return find(buf, s, e, rest) if rest else (s, e)
    return None


def video_track(buf, moov):
    """Return the trak whose handler is 'vide'."""
    for btype, s, e in boxes(buf, *moov):
        if btype != "trak":
            continue
        hdlr = find(buf, s, e, ["mdia", "hdlr"])
        if hdlr and buf[hdlr[0] + 8:hdlr[0] + 12] == b"vide":
            return s, e
    return None


def probe(path):
    with open(path, "rb") as f:
        buf = f.read()
    moov = find(buf, 0, len(buf), ["moov"])
    if not moov:
        raise RuntimeError("no moov box (not a valid MP4?)")
    trak = video_track(buf, moov)
    if not trak:
        raise RuntimeError("no video track")

    mdhd = find(buf, *trak, ["mdia", "mdhd"])
    version = buf[mdhd[0]]
    if version == 1:
        timescale, duration = struct.unpack(">IQ", buf[mdhd[0] + 20:mdhd[0] + 32])
    else:
        timescale, duration = struct.unpack(">II", buf[mdhd[0] + 12:mdhd[0] + 20])
    seconds = duration / timescale if timescale else 0

    stsz = find(buf, *trak, ["mdia", "minf", "stbl", "stsz"])
    frames = struct.unpack(">I", buf[stsz[0] + 8:stsz[0] + 12])[0]

    width = height = 0
    stsd = find(buf, *trak, ["mdia", "minf", "stbl", "stsd"])
    if stsd:
        for _btype, s, e in boxes(buf, stsd[0] + 8, stsd[1]):
            width, height = struct.unpack(">HH", buf[s + 24:s + 28])
            break

    return {
        "file": os.path.basename(path),
        "cls": os.path.basename(os.path.dirname(path)),
        "seconds": seconds,
        "frames": frames,
        "fps": frames / seconds if seconds else 0,
        "res": f"{width}x{height}",
        "mb": os.path.getsize(path) / 1048576,
    }


def main():
    paths = []
    for arg in sys.argv[1:]:
        paths += sorted(glob.glob(arg)) if any(c in arg for c in "*?[") else [arg]
    if not paths:
        print("usage: probe_mp4.py <clip.mp4> [...]", file=sys.stderr)
        sys.exit(1)

    rows = []
    print(f"{'class':38} {'file':14} {'sec':>6} {'frames':>7} {'fps':>5} {'res':>10} {'MB':>6}")
    for p in paths:
        try:
            r = probe(p)
        except Exception as exc:
            print(f"{os.path.basename(p):53} ERROR: {exc}")
            continue
        rows.append(r)
        print(f"{r['cls']:38} {r['file']:14} {r['seconds']:6.1f} {r['frames']:7d} "
              f"{r['fps']:5.1f} {r['res']:>10} {r['mb']:6.1f}")

    if rows:
        tf = sum(r["frames"] for r in rows)
        ts = sum(r["seconds"] for r in rows)
        print(f"\n{len(rows)} clips: {ts:.0f}s total, {tf} frames, "
              f"avg {ts / len(rows):.1f}s / {tf // len(rows)} frames per clip")


if __name__ == "__main__":
    main()
