from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from extension.taiyi import TaiyiTracker


class FakeMonitor:
    def get_output(self):
        return {"layer": {"Metric": [{"step": 1, "value": 2.0}]}}


class FailingLoggedArtifact:
    name = "taiyi-output-run123:v0"

    def wait(self):
        raise ValueError("Invalid Client ID digest")


class FakeRun:
    id = "run123"

    def __init__(self):
        self.logged_artifacts = []

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)
        return FailingLoggedArtifact()


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.Artifact = FakeArtifact



def test_taiyi_finish_keeps_local_output_when_wandb_artifact_wait_fails(tmp_path):
    tracker = TaiyiTracker(
        SimpleNamespace(taiyi=False),
        monitor_config={"layer": ["Metric"]},
        wandb=FakeWandb(),
        output_dir=str(tmp_path),
    )
    tracker.monitor = FakeMonitor()

    info = tracker.finish()

    assert info["taiyi_output"] is True
    assert info["taiyi_artifact"] is None
    assert "Invalid Client ID digest" in info["taiyi_artifact_error"]
    assert info["taiyi_output_path"] == str(tmp_path / "taiyi_output.pt")
    assert (tmp_path / "taiyi_output.pt").exists()
    assert len(tracker.wandb.run.logged_artifacts) == 1
