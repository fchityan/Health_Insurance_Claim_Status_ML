"""Configure entirely synthetic test source paths without real dataset filenames."""
import pytest


@pytest.fixture(autouse=True)
def configure_synthetic_data_paths(monkeypatch, tmp_path):
    for key, basename in {
        "ENCOUNTERS_DATA_PATH": "private_a.csv",
        "CLAIMS_DATA_PATH": "private_b.csv",
        "ORGANIZATIONS_DATA_PATH": "private_c.csv",
        "PATIENTS_DATA_PATH": "private_d.csv",
        "PROVIDERS_DATA_PATH": "private_e.csv",
    }.items():
        monkeypatch.setenv(key, str(tmp_path / basename))
