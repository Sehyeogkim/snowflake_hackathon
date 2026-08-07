#!/usr/bin/env python3
"""Fetch individual clips from the Mendeley dataset zip without downloading all 9.3 GB.

Mendeley serves the dataset as one zip behind a redirect to a presigned S3 URL that
expires after ~300s. S3 honours Range requests, and a zip's central directory lives at
the end of the file — so we can pull the index (~120 KB), then range-fetch and inflate
just the members we want.

  python3 scripts/fetch_clips.py --index          # download + cache the file index
  python3 scripts/fetch_clips.py --list           # show per-class counts and sizes
  python3 scripts/fetch_clips.py --per-class 3    # fetch 3 smallest clips per class
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import zlib
from collections import defaultdict

DATASET_ID = "xjmtb22pff"
VERSION = 1
REDIRECT_URL = f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{VERSION}"
ZIP_TOTAL = 10002129420  # from Content-Range on the full object

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "data", "zip_index.json")
# Where `mavis data` / `mavis bench` look by default.
CLIPS_DIR = os.path.join(ROOT, "data", "clips")

# Classes 0-3 are the violations; 4-7 are their safe counterparts. Recall can
# only be measured on the violations, so they are fetched first — a partial
# download that is all safe clips cannot answer the question the benchmark asks.
HAZARD_PREFIXES = ("0_", "1_", "2_", "3_")


CHUNK = 512 * 1024
CURL_TIMEOUT = 280


def _fetch_exact(start, end, attempts):
    """GET exactly bytes [start, end], re-resolving the presigned URL each try."""
    want = end - start + 1
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            ["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", str(CURL_TIMEOUT),
             "-r", f"{start}-{end}", REDIRECT_URL],
            capture_output=True,
        )
        if proc.returncode == 0 and len(proc.stdout) == want:
            return proc.stdout
        print(f"    retry {attempt}/{attempts} "
              f"(rc={proc.returncode}, got {len(proc.stdout)}/{want} bytes)", flush=True)
    raise RuntimeError(f"failed to fetch bytes {start}-{end}")


def fetch_range(start, end, attempts=6):
    """GET bytes [start, end], split into chunks that fit the timeout.

    curl restarts a range from zero when --max-time expires, so on a slow link a
    request larger than (rate x timeout) can never finish: every attempt times
    out at the same place and retries from the beginning. At ~5 KB/s that ceiling
    is about 1.4 MB, which is smaller than most clips in this dataset.

    Chunking makes progress independent of bandwidth — each request is small
    enough to complete, and a failure costs one chunk rather than the whole file.
    It also keeps every request well inside the ~300s presigned-URL lifetime.
    """
    total = end - start + 1
    if total <= CHUNK:
        return _fetch_exact(start, end, attempts)

    parts, got, pos = [], 0, start
    while pos <= end:
        stop = min(pos + CHUNK - 1, end)
        parts.append(_fetch_exact(pos, stop, attempts))
        got += stop - pos + 1
        pos = stop + 1
        print(f"    {got / 1048576:.1f}/{total / 1048576:.1f} MB", flush=True)
    return b"".join(parts)


def build_index():
    """Download the zip tail and parse the central directory into a file list."""
    tail_len = 262144
    base = ZIP_TOTAL - tail_len
    print(f"fetching zip tail ({tail_len // 1024} KB) to read the central directory...")
    tail = fetch_range(base, ZIP_TOTAL - 1)

    j = tail.rfind(b"PK\x06\x06")  # zip64 end of central directory
    if j < 0:
        raise RuntimeError("zip64 EOCD not found in tail")
    count = struct.unpack("<Q", tail[j + 32:j + 40])[0]
    cd_offset = struct.unpack("<Q", tail[j + 48:j + 56])[0]

    off = cd_offset - base
    if off < 0:
        raise RuntimeError("central directory starts before the fetched tail")

    entries = []
    for _ in range(count):
        if tail[off:off + 4] != b"PK\x01\x02":
            raise RuntimeError(f"bad central directory signature at {off}")
        method, = struct.unpack("<H", tail[off + 10:off + 12])
        csize, usize = struct.unpack("<II", tail[off + 20:off + 28])
        nlen, elen, clen = struct.unpack("<HHH", tail[off + 28:off + 34])
        lho, = struct.unpack("<I", tail[off + 42:off + 46])
        name = tail[off + 46:off + 46 + nlen].decode("utf-8", "replace")
        extra = tail[off + 46 + nlen:off + 46 + nlen + elen]

        if 0xFFFFFFFF in (csize, usize, lho):
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[e:e + 4])
                body, b = extra[e + 4:e + 4 + hsz], 0
                if hid == 0x0001:
                    if usize == 0xFFFFFFFF:
                        usize, b = struct.unpack("<Q", body[b:b + 8])[0], b + 8
                    if csize == 0xFFFFFFFF:
                        csize, b = struct.unpack("<Q", body[b:b + 8])[0], b + 8
                    if lho == 0xFFFFFFFF:
                        lho, b = struct.unpack("<Q", body[b:b + 8])[0], b + 8
                e += 4 + hsz

        entries.append({"name": name, "csize": csize, "usize": usize,
                        "offset": lho, "method": method})
        off += 46 + nlen + elen + clen

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(entries, f, indent=1)
    print(f"indexed {len(entries)} files -> {INDEX_PATH}")
    return entries


def load_index():
    if not os.path.exists(INDEX_PATH):
        return build_index()
    with open(INDEX_PATH) as f:
        return json.load(f)


def split_class(entry):
    """'.../Dataset/test/3_carrying_overload_with_forklift/0_te10.mp4' -> ('test', '3_...')"""
    parts = entry["name"].split("/")
    return (parts[2], parts[3]) if len(parts) >= 5 else (None, None)


def show_list(entries):
    groups = defaultdict(lambda: [0, 0])
    for e in entries:
        s, c = split_class(e)
        if s:
            groups[(s, c)][0] += 1
            groups[(s, c)][1] += e["csize"]
    print(f"{'split/class':46} {'n':>5} {'MB':>9} {'avg MB':>9}")
    for (s, c) in sorted(groups):
        n, total = groups[(s, c)]
        print(f"{s + '/' + c:46} {n:5d} {total / 1048576:9.0f} {total / 1048576 / n:9.1f}")


def select(entries, splits, per_class=0, count=0):
    """Pick the smallest clips, ordered so an interrupted run is still usable.

    Bandwidth is the bottleneck and the link is unreliable, so the ordering
    matters as much as the selection. Smallest-first buys the most clips per
    byte. Within that, each pass walks hazard classes before their safe
    counterparts, and test before train — so stopping at any point leaves the
    most benchmarkable set available rather than whatever happened to be
    alphabetically first.
    """
    ranked = defaultdict(list)
    for e in entries:
        s, c = split_class(e)
        if s in splits:
            ranked[(s, c)].append(e)
    for key in ranked:
        ranked[key].sort(key=lambda x: x["csize"])

    def rank(key):
        split, cls = key
        return (
            0 if cls.startswith(HAZARD_PREFIXES) else 1,  # violations first
            list(splits).index(split),                    # then split order
            cls,
        )

    keys = sorted(ranked, key=rank)
    limit = per_class if per_class else len(entries)

    picked, depth = [], 0
    while depth < limit and (not count or len(picked) < count):
        advanced = False
        for key in keys:
            if count and len(picked) >= count:
                break
            if len(ranked[key]) > depth:
                picked.append((key[0], ranked[key][depth]))
                advanced = True
        if not advanced:
            break
        depth += 1
    return picked


def fetch_clip(entry, dest):
    """Range-fetch one zip member and inflate it to dest."""
    header = fetch_range(entry["offset"], entry["offset"] + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {entry['name']}")
    nlen, elen = struct.unpack("<HH", header[26:30])
    start = entry["offset"] + 30 + nlen + elen
    raw = fetch_range(start, start + entry["csize"] - 1)

    if entry["method"] == 0:
        data = raw
    elif entry["method"] == 8:
        data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
    else:
        raise RuntimeError(f"unsupported compression method {entry['method']}")

    if len(data) != entry["usize"]:
        raise RuntimeError(f"size mismatch: got {len(data)}, expected {entry['usize']}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true", help="(re)build the file index only")
    ap.add_argument("--list", action="store_true", help="show per-class counts and sizes")
    ap.add_argument("--per-class", type=int, default=0, help="clips to fetch per class")
    ap.add_argument("--count", type=int, default=0,
                    help="total clips to fetch, spread evenly over classes")
    ap.add_argument("--split", default="both", choices=["test", "train", "both"])
    args = ap.parse_args()
    # test first: it is the evaluation set, so it is what a partial download
    # most needs to cover.
    splits = ("test", "train") if args.split == "both" else (args.split,)

    if args.index:
        build_index()
        return

    entries = load_index()
    if args.list:
        show_list(entries)
        return
    if not args.per_class and not args.count:
        ap.error("pass --count N or --per-class N (or --list / --index)")

    picked = select(entries, splits, args.per_class, args.count)
    total = sum(e["csize"] for _s, e in picked)
    print(f"fetching {len(picked)} clips, {total / 1048576:.0f} MB compressed")
    print("order: violation classes first, then their safe counterparts\n")

    done = 0
    for i, (split, e) in enumerate(picked, 1):
        _, cls = split_class(e)
        name = os.path.basename(e["name"])
        dest = os.path.join(CLIPS_DIR, split, cls, name)
        tag = f"[{i}/{len(picked)}] {split}/{cls}/{name}"
        if os.path.exists(dest) and os.path.getsize(dest) == e["usize"]:
            print(f"{tag} — cached")
            done += 1
            continue
        print(f"{tag} ({e['csize'] / 1048576:.1f} MB)", flush=True)
        try:
            fetch_clip(e, dest)
            done += 1
        except Exception as exc:  # keep going; a missing clip is not fatal
            print(f"    FAILED: {exc}", file=sys.stderr, flush=True)

    print(f"\n{done}/{len(picked)} clips in {CLIPS_DIR}")


if __name__ == "__main__":
    main()
