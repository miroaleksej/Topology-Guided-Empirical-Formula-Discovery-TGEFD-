from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Literal


@dataclass(frozen=True)
class ArtifactStoreConfig:
    backend: Literal["local", "s3", "gcs"] = "local"
    base_dir: str = "./runs"
    bucket: str | None = None
    prefix: str = "tgefd/runs"
    endpoint_url: str | None = None
    region: str | None = None
    upload_timeout_sec: float = 30.0
    retry_max_attempts: int = 3
    retry_backoff_sec: float = 0.2
    retry_max_backoff_sec: float = 2.0


class ArtifactStore:
    def uri_for(self, artifact_id: str) -> str:
        raise NotImplementedError

    def store(self, artifact_dir: Path, artifact_id: str) -> str:
        raise NotImplementedError

    def exists(self, artifact_id: str) -> bool:
        raise NotImplementedError

    def load_json(self, artifact_id: str, filename: str) -> dict:
        raise NotImplementedError


def _retry_with_backoff(
    action_name: str,
    fn,
    *,
    max_attempts: int,
    backoff_sec: float,
    max_backoff_sec: float,
):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if backoff_sec < 0:
        raise ValueError("backoff_sec must be non-negative")
    if max_backoff_sec < 0:
        raise ValueError("max_backoff_sec must be non-negative")

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{action_name} failed after {attempt} attempts: {exc}"
                ) from exc
            delay = min(max_backoff_sec, backoff_sec * (2 ** (attempt - 1)))
            if delay > 0:
                time.sleep(delay)


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def uri_for(self, artifact_id: str) -> str:
        return str(Path(self.base_dir) / "v1" / artifact_id / "artifact")

    def store(self, artifact_dir: Path, artifact_id: str) -> str:
        return self.uri_for(artifact_id)

    def exists(self, artifact_id: str) -> bool:
        manifest = Path(self.base_dir) / "v1" / artifact_id / "artifact" / "manifest.json"
        return manifest.exists()

    def load_json(self, artifact_id: str, filename: str) -> dict:
        path = Path(self.base_dir) / "v1" / artifact_id / "artifact" / filename
        return json.loads(path.read_text(encoding="utf-8"))


class S3ArtifactStore(ArtifactStore):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        endpoint_url: str | None,
        region: str | None,
        *,
        upload_timeout_sec: float,
        retry_max_attempts: int,
        retry_backoff_sec: float,
        retry_max_backoff_sec: float,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.region = region
        self.upload_timeout_sec = upload_timeout_sec
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_sec = retry_backoff_sec
        self.retry_max_backoff_sec = retry_max_backoff_sec

        try:
            import boto3
        except Exception as exc:  # pragma: no cover - runtime path
            raise ImportError("boto3 is required for S3 artifact storage.") from exc

        session = boto3.session.Session(region_name=region)
        self.client = session.client("s3", endpoint_url=endpoint_url)

    def uri_for(self, artifact_id: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{artifact_id}/artifact"

    def store(self, artifact_dir: Path, artifact_id: str) -> str:
        base_prefix = f"{self.prefix}/{artifact_id}/artifact"
        uploaded_keys: list[str] = []
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(artifact_dir)
            key = f"{base_prefix}/{rel.as_posix()}"
            started = time.perf_counter()

            def _upload():
                self.client.upload_file(str(path), self.bucket, key)
                elapsed = time.perf_counter() - started
                if elapsed > self.upload_timeout_sec:
                    raise TimeoutError(
                        f"upload timeout for {rel.as_posix()} ({elapsed:.2f}s > {self.upload_timeout_sec:.2f}s)"
                    )

            try:
                _retry_with_backoff(
                    f"s3 upload {rel.as_posix()}",
                    _upload,
                    max_attempts=self.retry_max_attempts,
                    backoff_sec=self.retry_backoff_sec,
                    max_backoff_sec=self.retry_max_backoff_sec,
                )
            except Exception as exc:
                self._rollback_uploaded(uploaded_keys)
                raise RuntimeError(
                    f"s3 artifact upload failed for {artifact_id}; rollback attempted: {exc}"
                ) from exc
            uploaded_keys.append(key)
        return self.uri_for(artifact_id)

    def exists(self, artifact_id: str) -> bool:
        key = f"{self.prefix}/{artifact_id}/artifact/manifest.json"

        def _head():
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True

        try:
            return _retry_with_backoff(
                f"s3 head {key}",
                _head,
                max_attempts=self.retry_max_attempts,
                backoff_sec=self.retry_backoff_sec,
                max_backoff_sec=self.retry_max_backoff_sec,
            )
        except Exception:
            return False

    def load_json(self, artifact_id: str, filename: str) -> dict:
        key = f"{self.prefix}/{artifact_id}/artifact/{filename}"

        def _get():
            return self.client.get_object(Bucket=self.bucket, Key=key)

        response = _retry_with_backoff(
            f"s3 get {key}",
            _get,
            max_attempts=self.retry_max_attempts,
            backoff_sec=self.retry_backoff_sec,
            max_backoff_sec=self.retry_max_backoff_sec,
        )
        payload = response["Body"].read()
        return json.loads(payload.decode("utf-8"))

    def _rollback_uploaded(self, uploaded_keys: list[str]) -> None:
        for key in reversed(uploaded_keys):
            try:
                _retry_with_backoff(
                    f"s3 rollback delete {key}",
                    lambda key=key: self.client.delete_object(Bucket=self.bucket, Key=key),
                    max_attempts=self.retry_max_attempts,
                    backoff_sec=self.retry_backoff_sec,
                    max_backoff_sec=self.retry_max_backoff_sec,
                )
            except Exception:
                # Best effort rollback for partial upload failures.
                continue


class GCSArtifactStore(ArtifactStore):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        *,
        upload_timeout_sec: float,
        retry_max_attempts: int,
        retry_backoff_sec: float,
        retry_max_backoff_sec: float,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.upload_timeout_sec = upload_timeout_sec
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_sec = retry_backoff_sec
        self.retry_max_backoff_sec = retry_max_backoff_sec
        try:
            from google.cloud import storage  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime path
            raise ImportError("google-cloud-storage is required for GCS artifact storage.") from exc

        self.client = storage.Client()
        self.bucket_obj = self.client.bucket(bucket)

    def uri_for(self, artifact_id: str) -> str:
        return f"gs://{self.bucket}/{self.prefix}/{artifact_id}/artifact"

    def store(self, artifact_dir: Path, artifact_id: str) -> str:
        base_prefix = f"{self.prefix}/{artifact_id}/artifact"
        uploaded_blobs: list[tuple[str, object]] = []
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(artifact_dir)
            blob = self.bucket_obj.blob(f"{base_prefix}/{rel.as_posix()}")
            started = time.perf_counter()

            def _upload():
                blob.upload_from_filename(str(path))
                elapsed = time.perf_counter() - started
                if elapsed > self.upload_timeout_sec:
                    raise TimeoutError(
                        f"upload timeout for {rel.as_posix()} ({elapsed:.2f}s > {self.upload_timeout_sec:.2f}s)"
                    )

            try:
                _retry_with_backoff(
                    f"gcs upload {rel.as_posix()}",
                    _upload,
                    max_attempts=self.retry_max_attempts,
                    backoff_sec=self.retry_backoff_sec,
                    max_backoff_sec=self.retry_max_backoff_sec,
                )
            except Exception as exc:
                self._rollback_uploaded(uploaded_blobs)
                raise RuntimeError(
                    f"gcs artifact upload failed for {artifact_id}; rollback attempted: {exc}"
                ) from exc
            uploaded_blobs.append((f"{base_prefix}/{rel.as_posix()}", blob))
        return self.uri_for(artifact_id)

    def exists(self, artifact_id: str) -> bool:
        key = f"{self.prefix}/{artifact_id}/artifact/manifest.json"
        blob = self.bucket_obj.blob(key)
        try:
            return bool(
                _retry_with_backoff(
                    f"gcs head {key}",
                    blob.exists,
                    max_attempts=self.retry_max_attempts,
                    backoff_sec=self.retry_backoff_sec,
                    max_backoff_sec=self.retry_max_backoff_sec,
                )
            )
        except Exception:
            return False

    def load_json(self, artifact_id: str, filename: str) -> dict:
        key = f"{self.prefix}/{artifact_id}/artifact/{filename}"
        blob = self.bucket_obj.blob(key)

        def _get():
            return blob.download_as_bytes()

        payload = _retry_with_backoff(
            f"gcs get {key}",
            _get,
            max_attempts=self.retry_max_attempts,
            backoff_sec=self.retry_backoff_sec,
            max_backoff_sec=self.retry_max_backoff_sec,
        )
        return json.loads(payload.decode("utf-8"))

    def _rollback_uploaded(self, uploaded_blobs: list[tuple[str, object]]) -> None:
        for key, blob in reversed(uploaded_blobs):
            try:
                _retry_with_backoff(
                    f"gcs rollback delete {key}",
                    blob.delete,
                    max_attempts=self.retry_max_attempts,
                    backoff_sec=self.retry_backoff_sec,
                    max_backoff_sec=self.retry_max_backoff_sec,
                )
            except Exception:
                # Best effort rollback for partial upload failures.
                continue


def build_store(config: ArtifactStoreConfig) -> ArtifactStore:
    if config.backend == "local":
        return LocalArtifactStore(config.base_dir)
    if config.backend == "s3":
        if not config.bucket:
            raise ValueError("artifact store bucket is required for s3 backend")
        return S3ArtifactStore(
            config.bucket,
            config.prefix,
            config.endpoint_url,
            config.region,
            upload_timeout_sec=config.upload_timeout_sec,
            retry_max_attempts=config.retry_max_attempts,
            retry_backoff_sec=config.retry_backoff_sec,
            retry_max_backoff_sec=config.retry_max_backoff_sec,
        )
    if config.backend == "gcs":
        if not config.bucket:
            raise ValueError("artifact store bucket is required for gcs backend")
        return GCSArtifactStore(
            config.bucket,
            config.prefix,
            upload_timeout_sec=config.upload_timeout_sec,
            retry_max_attempts=config.retry_max_attempts,
            retry_backoff_sec=config.retry_backoff_sec,
            retry_max_backoff_sec=config.retry_max_backoff_sec,
        )
    raise ValueError("artifact store backend must be local, s3, or gcs")
