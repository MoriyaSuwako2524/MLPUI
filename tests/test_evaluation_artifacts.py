import json

import numpy as np
import pytest

from mlpui.data import NpyDataset
from mlpui.evaluation_artifacts import save_evaluation_artifacts


def dense_data(path):
    path.mkdir()
    np.save(path / "z.npy", np.array([6, 1, 1], dtype=np.int64))
    np.save(path / "pos.npy", np.zeros((2, 3, 3)))
    np.save(path / "charges.npy", np.array([[.1, -.2, .1], [.2, -.4, .2]]))
    np.save(path / "frame_index.npy", np.array([17, 29], dtype=np.int64))
    for name, values in {
        "is_train": [True, False], "is_validation": [False, True],
        "is_test": [False, False], "is_guard": [False, False],
    }.items():
        np.save(path / f"{name}.npy", values)
    return NpyDataset(path)


def test_dense_charge_artifacts_preserve_order_and_sidecars(tmp_path):
    data = dense_data(tmp_path / "data")
    observations = {"charges": [
        (np.array([.1, -.2, .1]), np.array([.11, -.21, .10])),
        (np.array([.2, -.4, .2]), np.array([.19, -.39, .20])),
    ]}
    result = save_evaluation_artifacts(observations, data, tmp_path / "artifacts",
                                       metadata={"model_family": "newtonnet"})
    assert result["frame_index_origin"] == "dataset_sidecar"
    predicted = np.load(tmp_path / "artifacts/charges_pred.npy", allow_pickle=False)
    reference = np.load(tmp_path / "artifacts/charges_ref.npy", allow_pickle=False)
    assert predicted.shape == reference.shape == (2, 3)
    np.testing.assert_array_equal(np.load(tmp_path / "artifacts/frame_index.npy"), [17, 29])
    np.testing.assert_array_equal(np.load(tmp_path / "artifacts/source_row_index.npy"), [0, 1])
    manifest = json.loads((tmp_path / "artifacts/metadata.json").read_text(encoding="utf-8"))
    assert manifest["properties"]["charges"]["unit"] == "e"
    assert manifest["files"]["charges_ref.npy"]["role"] == "dataset_reference_label"
    assert manifest["provenance"]["model_family"] == "newtonnet"


def test_ragged_artifacts_use_offsets_without_pickle(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    np.save(path / "z.npy", np.array([6, 1, 8, 1, 1], dtype=np.int64))
    np.save(path / "pos.npy", np.zeros((5, 3)))
    np.save(path / "offsets.npy", np.array([0, 2, 5], dtype=np.int64))
    np.save(path / "charges.npy", np.zeros(5))
    data = NpyDataset(path)
    observations = {"charges": [(np.zeros(2), np.ones(2)),
                                  (np.zeros(3), np.ones(3) * 2)]}
    result = save_evaluation_artifacts(observations, data, tmp_path / "artifacts")
    assert result["properties"]["charges"]["layout"] == "flat_atoms"
    np.testing.assert_array_equal(np.load(tmp_path / "artifacts/structure_offsets.npy"), [0, 2, 5])
    prediction = np.load(tmp_path / "artifacts/charges_pred.npy", allow_pickle=False)
    assert prediction.shape == (5,) and prediction.dtype.kind == "f"
    assert np.load(tmp_path / "artifacts/frame_index.npy").tolist() == [0, 1]


def test_invalid_sidecars_do_not_publish_partial_directory(tmp_path):
    data = dense_data(tmp_path / "data")
    (tmp_path / "data/is_guard.npy").unlink()
    observations = {"charges": [(np.zeros(3), np.zeros(3))] * 2}
    with pytest.raises(ValueError, match="incomplete"):
        save_evaluation_artifacts(observations, data, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()
    assert not list(tmp_path.glob(".artifacts.*.tmp"))
