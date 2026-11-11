"""
Software provenance for every run: git state of the repos in
configs/provenance_repos.json, the container images, and the libWCSimRoot in use.

Standard library only, so grid and shell scripts can call it by path:

    python3 .../src/ambe/provenance.py snapshot --tag ta_tank_12 --out <dir>
    python3 .../src/ambe/provenance.py snapshot --out-file <dir>/PROVENANCE.json --tarball x.tar.gz
    python3 .../src/ambe/provenance.py status [--brief]
    python3 .../src/ambe/provenance.py index-docs --out <dir>
    python3 .../src/ambe/provenance.py render-index --campaigns <dir>

Records, never blocks: a dirty tree is written down with `clean: false`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "provenance_repos.json"
MAX_UNTRACKED = 200


def load_registry(path=None) -> dict:
    with open(path or os.environ.get("AMBE_PROVENANCE_REGISTRY", REGISTRY)) as fh:
        return json.load(fh)


def md5_file(path) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git(repo, *args, timeout=120) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                         timeout=timeout, check=True)
    return out.stdout.decode(errors="replace")


def repo_state(path) -> dict:
    rec = {"path": str(path)}
    if not (Path(path) / ".git").exists():
        rec["error"] = "not a git repository"
        return rec
    try:
        rec["branch"] = _git(path, "rev-parse", "--abbrev-ref", "HEAD").strip()
        rec["head"] = _git(path, "rev-parse", "HEAD").strip()
        rec["describe"] = _git(path, "describe", "--dirty", "--always", "--tags").strip()
        status = _git(path, "status", "--porcelain").splitlines()
        untracked = [l[3:] for l in status if l.startswith("??")]
        rec["dirty_count"] = len(status)
        rec["modified"] = [l[3:] for l in status if not l.startswith("??")]
        rec["untracked"] = untracked[:MAX_UNTRACKED]
        rec["untracked_count"] = len(untracked)
        diff = subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--binary"],
                              capture_output=True, timeout=300, check=True).stdout
        rec["diff_md5"] = hashlib.md5(diff).hexdigest() if diff else None
        rec["clean"] = not status
    except (subprocess.SubprocessError, OSError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def image_state(path) -> dict:
    p = Path(path)
    rec = {"path": str(path)}
    try:
        rec["resolved"] = os.path.realpath(p)
        rec["mtime"] = _dt.datetime.fromtimestamp(Path(rec["resolved"]).stat().st_mtime,
                                                  _dt.timezone.utc).isoformat()
    except OSError as exc:
        rec["error"] = str(exc)
    return rec


def wcsim_lib_state(lib_dir) -> dict:
    def files(d):
        return {f: md5_file(Path(d) / f) for f in
                ("libWCSimRoot.so", "WCSimRootDict_rdict.pcm", "libWCSimRoot.rootmap")}

    rec = {"registry_dir": str(lib_dir), "registry_md5": files(lib_dir)}
    for d in filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":")):
        if (Path(d) / "libWCSimRoot.so").is_file():
            rec["ld_library_path_dir"] = d
            rec["ld_library_path_md5"] = files(d)
            break
    return rec


def snapshot(tag="", config=None, tarballs=(), argv=None, registry=None) -> dict:
    reg = load_registry(registry)
    repos = {name: repo_state(p) for name, p in reg["repos"].items()}
    rec = {
        "tag": tag,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "cwd": os.getcwd(),
        "argv": list(argv) if argv is not None else sys.argv,
        "clean": all(r.get("clean", False) for r in repos.values()),
        "repos": repos,
        "images": {name: image_state(p) for name, p in reg.get("images", {}).items()},
        "wcsim_lib": wcsim_lib_state(reg["wcsim_lib_dir"]),
    }
    if config:
        rec["config"] = {"path": str(Path(config).resolve()), "md5": md5_file(config),
                         "text": Path(config).read_text(errors="replace")}
    if tarballs:
        rec["tarballs"] = {str(Path(t).resolve()): md5_file(t) for t in tarballs}
    return rec


def write(rec: dict, out_dir=None, out_file=None) -> Path:
    if out_file is None:
        stamp = rec["utc"].replace("-", "").replace(":", "")
        name = "provenance_" + (re.sub(r"[^A-Za-z0-9._-]+", "_", rec["tag"]) + "_" if rec["tag"] else "") + stamp + ".json"
        out_file = Path(out_dir or ".") / name
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(rec, indent=1) + "\n")
    return out_file


def record_ambe_run(ctx, group: str, cmd: str, argv) -> Path | None:
    """`ambe` CLI hook: write the manifest into <run_dir>/configs/; warn instead of failing."""
    try:
        rec = snapshot(tag=f"ambe_{group}_{cmd}", config=ctx.config_path, argv=["ambe", *argv])
        path = write(rec, out_dir=ctx.run_dir / "configs")
        if not rec["clean"]:
            dirty = [n for n, r in rec["repos"].items() if not r.get("clean", False)]
            print(f"[provenance] {path}  (dirty: {', '.join(dirty)})", file=sys.stderr, flush=True)
        return path
    except Exception as exc:
        print(f"[provenance] WARNING: no manifest written: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None


def status_lines(brief=False, registry=None) -> list[str]:
    reg = load_registry(registry)
    lines = []
    for name, p in reg["repos"].items():
        r = repo_state(p)
        if "error" in r:
            lines.append(f"{name:22s} ERROR {r['error']}")
            continue
        state = "clean" if r["clean"] else f"dirty {len(r['modified'])}M/{r['untracked_count']}??"
        line = f"{name:22s} {r['branch']}@{r['head'][:7]}  {state}"
        if not brief and not r["clean"]:
            line += "  " + " ".join(r["modified"][:6]) + (" ..." if len(r["modified"]) > 6 else "")
        lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# Task-document inventory
# --------------------------------------------------------------------------- #
_LINK = re.compile(r"(?:\]\(|`|\s)((?:/exp|/pnfs|\.{0,2}/?[\w.-]+/)[\w./-]*\.md)\b")
_MEMORY_LINK = re.compile(r"\[\[([\w-]+)\]\]")


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()[:160]
    return fallback


def index_docs(registry=None) -> list[dict]:
    reg = load_registry(registry)
    skip = set(reg.get("doc_skip", []))
    docs, seen = [], set()
    for root in reg.get("doc_roots", []):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith("."))
            for fn in sorted(filenames):
                if not fn.endswith(".md"):
                    continue
                p = Path(dirpath) / fn
                real = os.path.realpath(p)
                if real in seen or p.is_symlink():
                    continue
                seen.add(real)
                try:
                    st = p.stat()
                    text = p.read_text(errors="replace")
                except OSError:
                    continue
                docs.append({
                    "path": str(p),
                    "mtime": _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
                    "size": st.st_size,
                    "md5": hashlib.md5(text.encode()).hexdigest(),
                    "title": _title(text, fn),
                    "links": sorted(set(_LINK.findall(text)))[:50],
                    "memory_links": sorted(set(_MEMORY_LINK.findall(text))),
                })
    by_md5 = {}
    for d in docs:
        by_md5.setdefault(d["md5"], []).append(d["path"])
    for d in docs:
        d["duplicates"] = [p for p in by_md5[d["md5"]] if p != d["path"]]
    return docs


def render_index(campaigns_dir) -> Path:
    """INDEX.md from docs_inventory.json + index.json (classification; `approved` gates the status)."""
    cdir = Path(campaigns_dir)
    inv = {d["path"]: d for d in json.loads((cdir / "docs_inventory.json").read_text())}
    entries = {e["path"]: e for e in json.loads((cdir / "index.json").read_text())}
    short = lambda p: p.replace("/exp/annie/app/users/dajana/", "APP/").replace("/exp/annie/data/users/dajana/", "DATA/")
    order = {"current": 0, "archive": 1, "superseded": 2, "retired": 3}
    lines = ["# Task-document index", "",
             f"Generated {_dt.date.today()} by `provenance.py render-index`. APP = /exp/annie/app/users/dajana, "
             "DATA = /exp/annie/data/users/dajana.", "",
             "Status: **current** = follow this one; **archive** = finished, nothing replaces it; "
             "**superseded** = follow the → document instead; **retired** = merged into the → document's "
             "`History and lessons` and removed (copy in `_archive/`). `(proposed)` = not yet approved by Dajana.", ""]
    by_topic = {}
    for e in entries.values():
        by_topic.setdefault(e.get("topic", "unsorted"), []).append(e)
    for topic in sorted(by_topic):
        lines += [f"## {topic}", "", "| status | type | date | document | summary |", "|---|---|---|---|---|"]
        for e in sorted(by_topic[topic], key=lambda e: (order.get(e["status"], 9), e["path"])):
            st = e["status"] + ("" if e.get("approved") else " (proposed)")
            if e.get("superseded_by"):
                st += f" → `{short(e['superseded_by'])}`"
            date = inv.get(e["path"], {}).get("mtime", e.get("date", ""))
            lines.append(f"| {st} | {e.get('type', '')} | {date} | `{short(e['path'])}` | {e.get('summary', '')} |")
        lines.append("")
    new = sorted(set(inv) - set(entries))
    if new:
        lines += ["## Unclassified (new since the last classification)", ""]
        lines += [f"- `{short(p)}` ({inv[p]['mtime']}) {inv[p]['title']}" for p in new]
        lines.append("")
    out = cdir / "INDEX.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="provenance")
    ap.add_argument("--registry", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--tag", default="")
    s.add_argument("--out", default=None, help="directory; file name is generated")
    s.add_argument("--out-file", default=None)
    s.add_argument("--config", default=None)
    s.add_argument("--tarball", action="append", default=[])
    st = sub.add_parser("status")
    st.add_argument("--brief", action="store_true")
    ix = sub.add_parser("index-docs")
    ix.add_argument("--out", required=True, help="writes docs_inventory.json here")
    ri = sub.add_parser("render-index")
    ri.add_argument("--campaigns", required=True, help="dir holding docs_inventory.json + index.json")
    a = ap.parse_args(argv)

    if a.cmd == "snapshot":
        rec = snapshot(a.tag, a.config, a.tarball, argv=sys.argv, registry=a.registry)
        print(write(rec, a.out, a.out_file))
        if not rec["clean"]:
            print("[provenance] dirty repos: " + ", ".join(
                n for n, r in rec["repos"].items() if not r.get("clean", False)), file=sys.stderr)
    elif a.cmd == "status":
        print("\n".join(status_lines(a.brief, a.registry)))
    elif a.cmd == "index-docs":
        docs = index_docs(a.registry)
        out = Path(a.out) / "docs_inventory.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(docs, indent=1) + "\n")
        print(f"{out}  ({len(docs)} documents)")
    elif a.cmd == "render-index":
        print(render_index(a.campaigns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
