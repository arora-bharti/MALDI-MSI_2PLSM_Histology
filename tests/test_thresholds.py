"""
Guard against a silent StarDist threshold regression.

StarDist reads its detection thresholds from ``model.thresholds``, NOT from
``model.config``. Assigning ``model.config.prob_thresh`` therefore does nothing —
no error, no warning — and the model quietly runs at its built-in 0.6925 instead of
the documented 0.4. On the sample H&E image that is 573 detected nuclei instead of
956, i.e. roughly 40% of nuclei missed.

Source-level checks on purpose: reproducing this for real needs TensorFlow, StarDist
and a downloaded model, none of which are available in CI, and a test that is always
skipped is exactly how this defect survived in the first place.
"""
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _strip_comments(source):
    """Drop comments so a comment explaining the bug isn't mistaken for the bug."""
    return "\n".join(
        code for line in source.splitlines()
        if (code := line.split("#", 1)[0]).strip()
    )


def _source(rel_path):
    path = REPO_ROOT / rel_path
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        text = "\n".join("".join(c["source"]) for c in nb["cells"]
                         if c["cell_type"] == "code")
    else:
        text = path.read_text()
    return _strip_comments(text)


CALLERS = ["scripts/segment_nuclei.py",
           "notebooks/Nuclei_segmentation.ipynb",
           "app/streamlit_app.py"]


@pytest.mark.parametrize("rel_path", CALLERS)
def test_thresholds_not_assigned_to_config(rel_path):
    src = _source(rel_path)
    assert "model.config.prob_thresh" not in src, (
        f"{rel_path}: model.config.prob_thresh is silently ignored by StarDist — "
        "pass prob_thresh=... to the predict call instead"
    )
    assert "model.config.nms_thresh" not in src


@pytest.mark.parametrize("rel_path", CALLERS)
def test_every_predict_call_sets_thresholds(rel_path):
    """An unwired predict call silently falls back to StarDist's own threshold."""
    src = _source(rel_path)
    calls = src.count(".predict_instances")
    wired = src.count("prob_thresh=")
    assert calls > 0, f"{rel_path}: no predict call found"
    assert wired >= calls, (
        f"{rel_path}: {calls} predict call(s) but only {wired} pass prob_thresh"
    )


def test_notebook_uses_documented_default():
    """The notebook must use the 0.4 documented in README.md, not StarDist's 0.6925."""
    src = _source("notebooks/Nuclei_segmentation.ipynb")
    assert "prob_thresh = 0.4" in src
    assert "nms_thresh = 0.3" in src
