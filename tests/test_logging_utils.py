import json
from pathlib import Path

from src.logging_utils import RunLogger, create_run_manifest


def test_run_logger_configures_once():
    logger = RunLogger(name="test_logger").configure()
    assert logger.name == "test_logger"


def test_create_run_manifest_stores_metadata(tmp_path):
    path = create_run_manifest(tmp_path, {"a": 1}, extras={"note": "demo"})
    data = json.loads(Path(path).read_text())
    assert data["config"]["a"] == 1
    assert data["metadata"]["note"] == "demo"
