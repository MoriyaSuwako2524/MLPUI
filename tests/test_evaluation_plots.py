import numpy as np
import pytest

from mlpui.evaluation_plots import save_parity_plots
from mlpui.training import TrainingStopped


def test_constant_and_single_point_plots(tmp_path):
    pairs = {"energy": [np.array([[0., 0.]])], "charges": [np.ones((5, 2))]}
    metrics = {k: dict(mae=0., rmse=0., count=sum(len(c) for c in v)) for k, v in pairs.items()}
    result = save_parity_plots(pairs, metrics, tmp_path)
    for key in pairs:
        assert (tmp_path / result[key]["png"]).read_bytes().startswith(b"\x89PNG")
        svg = (tmp_path / result[key]["svg"]).read_text(encoding="utf-8")
        assert "RMSE = 0" in svg
        assert "Reference (converted dataset units)" in svg
    with pytest.raises(TrainingStopped):
        save_parity_plots(pairs, metrics, tmp_path, should_stop=lambda: True)
