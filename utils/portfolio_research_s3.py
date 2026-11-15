"""S3 upload for portfolio research artifacts (external_k8s)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def s3_env_complete() -> bool:
    return bool(
        _env("PORTFOLIO_RESEARCH_S3_ENDPOINT")
        and _env("PORTFOLIO_RESEARCH_S3_BUCKET")
        and _env("PORTFOLIO_RESEARCH_S3_ACCESS_KEY")
        and _env("PORTFOLIO_RESEARCH_S3_SECRET_KEY")
    )


def should_use_s3_backend() -> bool:
    override = _env("PORTFOLIO_RESEARCH_ARTIFACT_BACKEND").lower()
    if override == "local":
        return False
    if override == "s3":
        if not s3_env_complete():
            raise RuntimeError("PORTFOLIO_RESEARCH_ARTIFACT_BACKEND=s3 but S3 env is incomplete")
        return True
    runtime = _env("PORTFOLIO_RESEARCH_RUNTIME") or "local_docker"
    if runtime != "external_k8s":
        return False
    if not s3_env_complete():
        raise RuntimeError("PORTFOLIO_RESEARCH_RUNTIME=external_k8s requires complete S3 configuration")
    return True


def _s3_client():
    import boto3
    from botocore.config import Config

    endpoint = _env("PORTFOLIO_RESEARCH_S3_ENDPOINT")
    region = _env("PORTFOLIO_RESEARCH_S3_REGION") or "us-east-1"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=_env("PORTFOLIO_RESEARCH_S3_ACCESS_KEY"),
        aws_secret_access_key=_env("PORTFOLIO_RESEARCH_S3_SECRET_KEY"),
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )


def checkpoint_s3_key(job_id: str) -> str:
    return f"portfolio-research/{job_id}/checkpoint/completed_jobs.json"


class S3CheckpointStore:
    """Sweep checkpoint index kept in S3 instead of on the worker's disk.

    Research workers run as a Deployment, so a routine image rollout replaces
    the pod mid-sweep and takes any emptyDir with it. Keeping the index in S3 is
    what lets the next worker resume instead of recomputing every combination.

    A checkpoint is an optimization, so read and write failures degrade to
    "no resume" rather than killing the sweep — but they are logged loudly,
    because silent degradation is how the emptyDir problem went unnoticed.
    """

    def __init__(self, job_id: str):
        self.job_id = str(job_id)
        self.bucket = _env("PORTFOLIO_RESEARCH_S3_BUCKET")
        self.key = checkpoint_s3_key(self.job_id)

    def load(self) -> Dict[str, Any]:
        try:
            client = _s3_client()
            response = client.get_object(Bucket=self.bucket, Key=self.key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except Exception as exc:
            if _is_missing_key(exc):
                return {}
            logger.warning("could not read sweep checkpoint s3://%s/%s: %s", self.bucket, self.key, exc)
            return {}

    def save(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        try:
            _s3_client().put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=body,
                ContentType="application/json",
            )
        except Exception as exc:
            logger.warning("could not write sweep checkpoint s3://%s/%s: %s", self.bucket, self.key, exc)

    def describe(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def _is_missing_key(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str((response.get("Error") or {}).get("Code") or "")
    return code in {"NoSuchKey", "404", "NotFound"}


def artifact_s3_prefix(job_id: str, result_id: str) -> str:
    id8 = result_id[:8]
    return f"portfolio-research/{job_id}/{id8}/"


def upload_artifact_to_s3(
    *,
    job_id: str,
    result_id: str,
    html_path: Path,
    json_path: Path,
    combo_dir: Optional[Path],
) -> Dict[str, Any]:
    """Upload report + combo JSONs; returns ``storage`` + ``s3`` fields for Mongo artifact."""
    bucket = _env("PORTFOLIO_RESEARCH_S3_BUCKET")
    prefix = artifact_s3_prefix(job_id, result_id)
    combos_prefix = f"{prefix}combos/"
    html_key = f"{prefix}report.html"
    json_key = f"{prefix}report.json"

    client = _s3_client()
    uploaded_keys = []
    try:
        client.upload_file(str(html_path), bucket, html_key)
        uploaded_keys.append(html_key)
        client.upload_file(str(json_path), bucket, json_key)
        uploaded_keys.append(json_key)
        if combo_dir is not None and combo_dir.is_dir():
            for combo_file in sorted(combo_dir.glob("*.json")):
                combo_key = f"{combos_prefix}{combo_file.name}"
                client.upload_file(str(combo_file), bucket, combo_key)
                uploaded_keys.append(combo_key)
    except Exception:
        for key in reversed(uploaded_keys):
            try:
                client.delete_object(Bucket=bucket, Key=key)
            except Exception as cleanup_exc:
                logger.warning("failed to clean partial S3 artifact s3://%s/%s: %s", bucket, key, cleanup_exc)
        raise

    return {
        "storage": "s3",
        "s3": {
            "bucket": bucket,
            "prefix": prefix,
            "html_key": html_key,
            "json_key": json_key,
            "combos_prefix": combos_prefix,
        },
    }
