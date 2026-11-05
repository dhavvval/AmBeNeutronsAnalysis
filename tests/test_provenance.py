import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from ambe import provenance


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    run("add", "a.txt")
    run("commit", "-qm", "init")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return repo, head


def _registry(tmp_path, repo):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"repos": {"r": str(repo)}, "images": {},
                               "wcsim_lib_dir": str(tmp_path),
                               "doc_roots": [str(tmp_path)], "doc_skip": [".git"]}))
    return reg


def test_clean_then_dirty(tmp_path):
    repo, head = _repo(tmp_path)
    reg = _registry(tmp_path, repo)

    rec = provenance.snapshot(registry=reg, argv=[])
    r = rec["repos"]["r"]
    assert r["head"] == head and r["clean"] and rec["clean"]
    assert r["diff_md5"] is None and r["dirty_count"] == 0

    (repo / "a.txt").write_text("two\n")
    (repo / "new.txt").write_text("x\n")
    rec = provenance.snapshot(registry=reg, argv=[])
    r = rec["repos"]["r"]
    assert not rec["clean"] and r["describe"].endswith("-dirty")
    assert r["modified"] == ["a.txt"] and r["untracked"] == ["new.txt"]
    first = r["diff_md5"]
    assert first

    (repo / "a.txt").write_text("three\n")
    assert provenance.snapshot(registry=reg, argv=[])["repos"]["r"]["diff_md5"] != first


def test_write_and_ambe_hook(tmp_path, monkeypatch):
    repo, _ = _repo(tmp_path)
    monkeypatch.setenv("AMBE_PROVENANCE_REGISTRY", str(_registry(tmp_path, repo)))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("run_name: x\n")
    ctx = SimpleNamespace(config_path=cfg, run_dir=tmp_path / "out")

    path = provenance.record_ambe_run(ctx, "plots", "basic", ["plots", "basic", "--config", str(cfg)])
    rec = json.loads(Path(path).read_text())
    assert path.parent == tmp_path / "out" / "configs"
    assert rec["tag"] == "ambe_plots_basic" and rec["config"]["text"] == "run_name: x\n"


def test_index_docs_finds_duplicates(tmp_path):
    repo, _ = _repo(tmp_path)
    reg = _registry(tmp_path, repo)
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "d1" / "R.md").write_text("# Report\nsee [[some-note]]\n")
    (tmp_path / "d2" / "R.md").write_text("# Report\nsee [[some-note]]\n")
    docs = {Path(d["path"]).parent.name: d for d in provenance.index_docs(reg)}
    assert docs["d1"]["title"] == "Report" and docs["d1"]["memory_links"] == ["some-note"]
    assert docs["d1"]["duplicates"] == [str(tmp_path / "d2" / "R.md")]


def test_render_index_marks_proposed_and_unclassified(tmp_path):
    a, b, c = (str(tmp_path / n) for n in ("old.md", "new.md", "fresh.md"))
    inv = [{"path": p, "mtime": "2026-10-01", "title": "t"} for p in (a, b, c)]
    (tmp_path / "docs_inventory.json").write_text(json.dumps(inv))
    (tmp_path / "index.json").write_text(json.dumps([
        {"path": a, "topic": "x", "type": "report/results", "status": "superseded", "superseded_by": b,
         "summary": "old", "approved": False},
        {"path": b, "topic": "x", "type": "report/results", "status": "current", "summary": "new", "approved": True},
    ]))
    text = provenance.render_index(tmp_path).read_text()
    assert "superseded (proposed) →" in text and "| current |" in text
    assert "Unclassified" in text and "fresh.md" in text
