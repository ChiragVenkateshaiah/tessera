"""The generator is deterministic and its committed output is current."""

from __future__ import annotations

import filecmp
import importlib.util
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "data" / "expertise" / "generate.py"
COMMITTED_PEOPLE = REPO_ROOT / "data" / "expertise" / "people"


def _load_generator():
    spec = importlib.util.spec_from_file_location("_expertise_generate", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dir_contents_equal(a: Path, b: Path) -> bool:
    names_a = sorted(p.name for p in a.glob("*.yaml"))
    names_b = sorted(p.name for p in b.glob("*.yaml"))
    if names_a != names_b:
        return False
    match, mismatch, errors = filecmp.cmpfiles(a, b, names_a, shallow=False)
    return not mismatch and not errors


def test_generator_output_matches_committed_dataset(tmp_path, monkeypatch):
    """Re-running the generator reproduces the committed files byte for
    byte — so the committed dataset and the script never drift, and eval
    labels built against it stay valid.
    """
    module = _load_generator()
    staging = tmp_path / "people"
    staging.mkdir()
    monkeypatch.setattr(module, "OUT_DIR", staging)

    module.main()

    assert _dir_contents_equal(staging, COMMITTED_PEOPLE), (
        "data/expertise/people/ is stale — re-run `python data/expertise/generate.py`"
    )


def test_generator_is_deterministic_run_to_run(tmp_path):
    module = _load_generator()
    first, second = tmp_path / "a", tmp_path / "b"

    for target in (first, second):
        target.mkdir()
        module.OUT_DIR = target
        module.main()

    assert _dir_contents_equal(first, second)


def test_generated_dataset_size_and_pyramid(tmp_path):
    module = _load_generator()
    staging = tmp_path / "people"
    staging.mkdir()
    module.OUT_DIR = staging
    module.main()

    import yaml

    people = []
    for f in staging.glob("*.yaml"):
        people.extend(yaml.safe_load(f.read_text(encoding="utf-8")))

    assert len(people) == module.TOTAL_PEOPLE == 600
    titles = [p["title"] for p in people]
    # junior-heavy pyramid
    assert titles.count("Analyst") > titles.count("Partner") * 4
    assert titles.count("Partner") >= 5

    # every practice file present
    assert {f.stem for f in staging.glob("*.yaml")} == set(module.PRACTICES)
