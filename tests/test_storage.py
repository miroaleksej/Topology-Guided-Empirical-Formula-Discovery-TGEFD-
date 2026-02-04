from pathlib import Path

import pytest

from tgefd.storage import GCSArtifactStore, S3ArtifactStore, _retry_with_backoff


def _make_artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("a", encoding="utf-8")
    (artifact_dir / "b.txt").write_text("b", encoding="utf-8")
    return artifact_dir


def test_retry_with_backoff_eventual_success():
    state = {"calls": 0}

    def _fn():
        state["calls"] += 1
        if state["calls"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result = _retry_with_backoff(
        "test action",
        _fn,
        max_attempts=3,
        backoff_sec=0.0,
        max_backoff_sec=0.0,
    )
    assert result == "ok"
    assert state["calls"] == 3


def test_s3_store_rolls_back_partial_failure(tmp_path: Path):
    artifact_dir = _make_artifact_dir(tmp_path)

    class _Client:
        def __init__(self) -> None:
            self.uploaded: list[str] = []
            self.deleted: list[str] = []

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            if key.endswith("b.txt"):
                raise RuntimeError("upload failed")
            self.uploaded.append(key)

        def delete_object(self, Bucket: str, Key: str) -> None:
            self.deleted.append(Key)

    client = _Client()
    store = S3ArtifactStore.__new__(S3ArtifactStore)
    store.bucket = "bucket"
    store.prefix = "prefix"
    store.client = client
    store.upload_timeout_sec = 30.0
    store.retry_max_attempts = 1
    store.retry_backoff_sec = 0.0
    store.retry_max_backoff_sec = 0.0

    with pytest.raises(RuntimeError, match="rollback attempted"):
        store.store(artifact_dir, "sha256-abc")

    assert len(client.uploaded) == 1
    assert len(client.deleted) == 1
    assert client.deleted[0].endswith("a.txt")


def test_s3_store_retries_transient_failure(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "one.txt").write_text("1", encoding="utf-8")

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")

        def delete_object(self, Bucket: str, Key: str) -> None:
            return None

    client = _Client()
    store = S3ArtifactStore.__new__(S3ArtifactStore)
    store.bucket = "bucket"
    store.prefix = "prefix"
    store.client = client
    store.upload_timeout_sec = 30.0
    store.retry_max_attempts = 2
    store.retry_backoff_sec = 0.0
    store.retry_max_backoff_sec = 0.0

    uri = store.store(artifact_dir, "sha256-abc")
    assert uri == "s3://bucket/prefix/sha256-abc/artifact"
    assert client.calls == 2


def test_gcs_store_rolls_back_partial_failure(tmp_path: Path):
    artifact_dir = _make_artifact_dir(tmp_path)

    class _Blob:
        def __init__(self, key: str, uploaded: list[str], deleted: list[str]) -> None:
            self.key = key
            self._uploaded = uploaded
            self._deleted = deleted

        def upload_from_filename(self, filename: str) -> None:
            if self.key.endswith("b.txt"):
                raise RuntimeError("upload failed")
            self._uploaded.append(self.key)

        def delete(self) -> None:
            self._deleted.append(self.key)

    class _Bucket:
        def __init__(self) -> None:
            self.uploaded: list[str] = []
            self.deleted: list[str] = []

        def blob(self, key: str) -> _Blob:
            return _Blob(key, self.uploaded, self.deleted)

    bucket = _Bucket()
    store = GCSArtifactStore.__new__(GCSArtifactStore)
    store.bucket = "bucket"
    store.prefix = "prefix"
    store.bucket_obj = bucket
    store.upload_timeout_sec = 30.0
    store.retry_max_attempts = 1
    store.retry_backoff_sec = 0.0
    store.retry_max_backoff_sec = 0.0

    with pytest.raises(RuntimeError, match="rollback attempted"):
        store.store(artifact_dir, "sha256-abc")

    assert len(bucket.uploaded) == 1
    assert len(bucket.deleted) == 1
    assert bucket.deleted[0].endswith("a.txt")
