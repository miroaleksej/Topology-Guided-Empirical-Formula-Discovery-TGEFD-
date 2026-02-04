from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def _find_artifacts(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("v1/sha256-*/artifact"))


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.title("TGEFD Artifact Dashboard")

    base_dir = st.text_input("Artifacts base dir", value="./runs")
    base_path = Path(base_dir)
    if not base_path.exists():
        st.warning("Base dir does not exist yet.")
        return

    artifacts = _find_artifacts(base_path)
    if not artifacts:
        st.info("No artifacts found.")
        return

    labels = [p.parent.name for p in artifacts]
    selected = st.selectbox("Artifact", labels)
    artifact_dir = artifacts[labels.index(selected)]

    manifest_path = artifact_dir / "manifest.json"
    metrics_path = artifact_dir / "metrics.json"

    if manifest_path.exists():
        st.subheader("Manifest")
        st.json(_load_json(manifest_path))

    if metrics_path.exists():
        st.subheader("Metrics")
        metrics = _load_json(metrics_path)
        st.json(metrics)

        if "noise_report" in metrics:
            st.subheader("Noise Report")
            st.dataframe(metrics["noise_report"])


if __name__ == "__main__":
    main()
