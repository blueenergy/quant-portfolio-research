import io
import json
from unittest.mock import MagicMock, patch

from utils import portfolio_research_s3 as s3mod


def _s3_env(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_BUCKET", "aipoc")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_SECRET_KEY", "sk")


def test_should_use_s3_backend_local_docker(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_RESEARCH_ARTIFACT_BACKEND", raising=False)
    monkeypatch.setenv("PORTFOLIO_RESEARCH_RUNTIME", "local_docker")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_BUCKET", "aipoc")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_SECRET_KEY", "sk")
    assert s3mod.should_use_s3_backend() is False


def test_should_use_s3_backend_external_k8s(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_RESEARCH_ARTIFACT_BACKEND", raising=False)
    monkeypatch.setenv("PORTFOLIO_RESEARCH_RUNTIME", "external_k8s")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_BUCKET", "aipoc")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_SECRET_KEY", "sk")
    assert s3mod.should_use_s3_backend() is True


def test_external_k8s_rejects_incomplete_s3_env(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_RESEARCH_ARTIFACT_BACKEND", raising=False)
    monkeypatch.setenv("PORTFOLIO_RESEARCH_RUNTIME", "external_k8s")
    for key in (
        "PORTFOLIO_RESEARCH_S3_ENDPOINT",
        "PORTFOLIO_RESEARCH_S3_BUCKET",
        "PORTFOLIO_RESEARCH_S3_ACCESS_KEY",
        "PORTFOLIO_RESEARCH_S3_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    try:
        s3mod.should_use_s3_backend()
    except RuntimeError as exc:
        assert "requires complete S3 configuration" in str(exc)
    else:
        raise AssertionError("external_k8s must reject incomplete S3 configuration")


def test_upload_artifact_to_s3_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_BUCKET", "aipoc")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_SECRET_KEY", "sk")

    html_path = tmp_path / "a.html"
    json_path = tmp_path / "a.json"
    combo_dir = tmp_path / "job_combos"
    combo_dir.mkdir()
    html_path.write_text("<html></html>", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")
    (combo_dir / "combo-a.json").write_text("{}", encoding="utf-8")

    client = MagicMock()
    with patch.object(s3mod, "_s3_client", return_value=client):
        meta = s3mod.upload_artifact_to_s3(
            job_id="job-1",
            result_id="abcdef12-0000-0000-0000-000000000000",
            html_path=html_path,
            json_path=json_path,
            combo_dir=combo_dir,
        )

    assert meta["storage"] == "s3"
    assert meta["s3"]["bucket"] == "aipoc"
    assert meta["s3"]["prefix"] == "portfolio-research/job-1/abcdef12/"
    assert meta["s3"]["html_key"].endswith("report.html")
    assert client.upload_file.call_count == 3


def test_upload_failure_cleans_already_uploaded_objects(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_BUCKET", "aipoc")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("PORTFOLIO_RESEARCH_S3_SECRET_KEY", "sk")

    html_path = tmp_path / "a.html"
    json_path = tmp_path / "a.json"
    html_path.write_text("<html></html>", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")

    client = MagicMock()
    client.upload_file.side_effect = [None, RuntimeError("upload failed")]
    with patch.object(s3mod, "_s3_client", return_value=client):
        try:
            s3mod.upload_artifact_to_s3(
                job_id="job-1",
                result_id="abcdef12-0000-0000-0000-000000000000",
                html_path=html_path,
                json_path=json_path,
                combo_dir=None,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("upload failure must propagate")

    client.delete_object.assert_called_once_with(
        Bucket="aipoc",
        Key="portfolio-research/job-1/abcdef12/report.html",
    )


def test_checkpoint_key_is_scoped_to_the_job(monkeypatch):
    _s3_env(monkeypatch)

    assert s3mod.checkpoint_s3_key("job-1") == "portfolio-research/job-1/checkpoint/completed_jobs.json"
    assert s3mod.S3CheckpointStore("job-1").describe() == (
        "s3://aipoc/portfolio-research/job-1/checkpoint/completed_jobs.json"
    )


def test_checkpoint_store_round_trips_the_completed_index(monkeypatch):
    _s3_env(monkeypatch)
    client = MagicMock()

    with patch.object(s3mod, "_s3_client", return_value=client):
        store = s3mod.S3CheckpointStore("job-1")
        store.save({"hash-a": [{"top_n": 10}]})

        kwargs = client.put_object.call_args.kwargs
        client.get_object.return_value = {"Body": io.BytesIO(kwargs["Body"])}
        assert store.load() == {"hash-a": [{"top_n": 10}]}

    assert kwargs["Bucket"] == "aipoc"
    assert kwargs["Key"] == "portfolio-research/job-1/checkpoint/completed_jobs.json"
    assert json.loads(kwargs["Body"].decode("utf-8")) == {"hash-a": [{"top_n": 10}]}


def test_checkpoint_load_treats_a_missing_object_as_a_fresh_start(monkeypatch):
    _s3_env(monkeypatch)
    client = MagicMock()
    client.get_object.side_effect = _client_error("NoSuchKey")

    with patch.object(s3mod, "_s3_client", return_value=client):
        assert s3mod.S3CheckpointStore("job-1").load() == {}


def test_checkpoint_failures_degrade_instead_of_killing_the_sweep(monkeypatch, caplog):
    _s3_env(monkeypatch)
    client = MagicMock()
    client.get_object.side_effect = _client_error("InternalError")
    client.put_object.side_effect = RuntimeError("network down")

    with patch.object(s3mod, "_s3_client", return_value=client):
        store = s3mod.S3CheckpointStore("job-1")
        with caplog.at_level("WARNING"):
            assert store.load() == {}
            store.save({"hash-a": []})

    # A checkpoint is an optimization, but a broken one must not be silent.
    messages = [record.getMessage() for record in caplog.records]
    assert any("could not read sweep checkpoint" in message for message in messages)
    assert any("could not write sweep checkpoint" in message for message in messages)


def _client_error(code: str) -> Exception:
    exc = Exception(code)
    exc.response = {"Error": {"Code": code}}  # type: ignore[attr-defined]
    return exc
