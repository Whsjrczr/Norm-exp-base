import argparse
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest

torch = pytest.importorskip("torch")
import extension.checkpoint as checkpoint


def test_checkpoint_save_switch_defaults_on_and_can_be_disabled():
    parser = argparse.ArgumentParser()
    checkpoint.add_arguments(parser)

    assert parser.parse_args([]).save_checkpoint is True
    assert parser.parse_args(["--no-save-checkpoint"]).save_checkpoint is False
    assert parser.parse_args(["--save-checkpoint"]).save_checkpoint is True


def test_no_save_checkpoint_skips_checkpoint_but_not_model():
    model = torch.nn.Linear(2, 1)
    cfg = SimpleNamespace(save_checkpoint=False)
    save_dir = checkpoint.os.path.abspath(
        checkpoint.os.path.join("results", "_test_checkpoint_" + uuid4().hex)
    )
    checkpoint.os.makedirs(save_dir, exist_ok=False)
    try:
        saver = checkpoint.Checkpoint(model, cfg=cfg, save_dir=save_dir, save_to_disk=True)

        saver.save_checkpoint()
        saver.save_model("best.pth")

        assert not checkpoint.os.path.exists(checkpoint.os.path.join(save_dir, "checkpoint.pth"))
        assert checkpoint.os.path.exists(checkpoint.os.path.join(save_dir, "best.pth"))
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
