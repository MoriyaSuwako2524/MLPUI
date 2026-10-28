"""Batched execution must reproduce per-structure gradient accumulation exactly.

Also covers the validation_interval / eval_batch_size / batched options.
"""
import numpy as np
import pytest
import torch

from mlpui.backends import get_backend
from mlpui.web.config import presets
from mlpui.data import NpyDataset
from mlpui.training import Trainer, TrainingConfig
from test_external_models_integration import torchmd_args

QM_COUNTS = (3, 4, 2, 5)
MM_COUNTS = (2, 0, 3, 1)  # includes an empty MM region
QMMM_TARGETS = ("energy", "forces", "mm_forces", "mm_potential", "mm_potential_gradient")


def mace_config(**kwargs):
    settings = presets()["mace"]
    settings.update(atomic_numbers=[1, 8], atomic_energies={"1": -.1, "8": -.2},
                    num_channels=4, max_ell=1, max_L=1, num_interactions=2,
                    correlation=2, radial_MLP=[8], readout_channels=4, charge_hidden_channels=4)
    settings.update(kwargs)
    return settings


def qmmm_config(family):
    config = get_backend(family).default_config()
    config.update(embedding_dimension=8, num_layers=1, num_rbf=4, precision=64)
    if family == "tensornet_qmmm":
        config.update(ext_num_rbf=4)
    return config


CASES = {
    "newtonnet": (lambda: dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1,
                               output_properties=["energy", "gradient_force"]), ("energy", "forces")),
    "newtonnet-charges": (lambda: dict(cutoff=3., n_features=8, n_basis=4, n_interactions=1,
                                       output_properties=["energy", "gradient_force"]), ("energy", "forces", "charges")),
    "torchmdnet": (lambda: torchmd_args("tensornet"), ("energy", "forces")),
    "torchmdnet-charges": (lambda: torchmd_args("tensornet"), ("energy", "forces", "charges")),
    "mace": (lambda: mace_config(), ("energy", "forces")),
    "mace-charges": (lambda: mace_config(), ("energy", "forces", "charges")),
    "tensornet_qmmm": (lambda: qmmm_config("tensornet_qmmm"), QMMM_TARGETS),
    "tensornet-ext": (lambda: qmmm_config("tensornet-ext"), QMMM_TARGETS),
}


def write_ragged(path, qmmm, seed=0):
    """Ragged QM (and optionally ragged MM) dataset with differing sizes."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    z, pos = [], []
    for n in QM_COUNTS:
        z.extend([8] + [1] * (n - 1))
        # Spread atoms ~1 A apart so every structure has neighbors within the cutoffs.
        pos.append(np.cumsum(rng.normal(scale=.55, size=(n, 3)) + [.6, 0, 0], axis=0))
    pos = np.concatenate(pos)
    atoms = len(z)
    np.save(path / "z.npy", np.array(z))
    np.save(path / "pos.npy", pos)
    np.save(path / "offsets.npy", np.concatenate(([0], np.cumsum(QM_COUNTS))))
    np.save(path / "energy.npy", rng.normal(size=len(QM_COUNTS)))
    np.save(path / "forces.npy", rng.normal(scale=.3, size=(atoms, 3)))
    np.save(path / "charges.npy", rng.normal(scale=.2, size=atoms))
    np.save(path / "charge.npy", np.zeros(len(QM_COUNTS)))
    if qmmm:
        starts = np.concatenate(([0], np.cumsum(QM_COUNTS)))
        mm_pos = np.concatenate([pos[starts[i]] + rng.uniform(2.5, 4., size=(m, 3)) * rng.choice([-1, 1], size=(m, 3))
                                 for i, m in enumerate(MM_COUNTS)])
        total = len(mm_pos)
        np.save(path / "mm_pos.npy", mm_pos)
        np.save(path / "mm_charges.npy", rng.normal(scale=.4, size=total))
        np.save(path / "mm_offsets.npy", np.concatenate(([0], np.cumsum(MM_COUNTS))))
        np.save(path / "mm_forces.npy", rng.normal(scale=.1, size=(total, 3)))
        np.save(path / "mm_potential.npy", rng.normal(scale=.1, size=total))
        np.save(path / "mm_potential_gradient.npy", rng.normal(scale=.1, size=(total, 3)))
        for name in ("charges",):
            (path / f"{name}.npy").unlink()
    return path


def make_trainer(case, **options):
    model_config, targets = CASES[case]
    family = case.split("-charges")[0]
    torch.manual_seed(3)
    config = TrainingConfig(dtype=torch.float64, loss_weights={t: 1. + .5 * i for i, t in enumerate(targets)},
                            seed=5, **options)
    return Trainer(family, model_config(), config)


def dataset(tmp_path, case):
    return NpyDataset(write_ragged(tmp_path / "data", qmmm=case in ("tensornet_qmmm", "tensornet-ext")))


def grads(trainer):
    return {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in trainer.model.named_parameters()}


@pytest.mark.parametrize("case", list(CASES))
def test_batched_loss_and_gradient_match_per_structure(tmp_path, case):
    trainer = make_trainer(case)
    data = dataset(tmp_path, case)
    samples = [data[i] for i in range(len(data))]
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    reference_parts = []
    for sample in samples:
        loss, parts = trainer._loss(sample)
        (loss / len(samples)).backward()
        reference_parts.append({k: v.item() for k, v in parts.items()})
    reference = grads(trainer)
    trainer.optimizer.zero_grad(set_to_none=True)
    batched = trainer._batched_errors(samples)
    assert batched is not None, "backend should batch these samples"
    results = [trainer._losses(errors) for errors in batched]
    (sum(total for total, _ in results) / len(samples)).backward()
    for (_, parts), expected in zip(results, reference_parts):
        for key, value in expected.items():
            assert parts[key].item() == pytest.approx(value, rel=1e-10, abs=1e-12)
    actual = grads(trainer)
    assert any(v.abs().sum() > 0 for v in reference.values())
    for name, value in reference.items():
        torch.testing.assert_close(actual[name], value, rtol=1e-8, atol=1e-11, msg=name)


@pytest.mark.parametrize("case", ["newtonnet", "torchmdnet", "mace", "tensornet_qmmm"])
def test_batched_fit_and_evaluate_equal_unbatched(tmp_path, case):
    data = dataset(tmp_path, case)
    runs = {}
    for batched in (False, True):
        trainer = make_trainer(case, epochs=2, batch_size=3, batched=batched, eval_batch_size=2)
        history = trainer.fit(data, data)
        runs[batched] = (history, {k: v.detach().clone() for k, v in trainer.model.state_dict().items()},
                         trainer.evaluate(data)["metrics"])
    (h0, s0, m0), (h1, s1, m1) = runs[False], runs[True]
    for r0, r1 in zip(h0, h1):
        for split in ("train", "validation"):
            for key, value in r0[split].items():
                assert r1[split][key] == pytest.approx(value, rel=1e-7, abs=1e-10)
    for key, value in s0.items():
        torch.testing.assert_close(s1[key], value, rtol=1e-7, atol=1e-10)
    for key, value in m0.items():
        assert m1[key]["count"] == value["count"]
        assert m1[key]["mae"] == pytest.approx(value["mae"], rel=1e-7, abs=1e-10)


def test_evaluation_artifact_order_is_preserved(tmp_path):
    data = dataset(tmp_path, "newtonnet")
    collected = {}
    for batched in (False, True):
        trainer = make_trainer("newtonnet", batched=batched, eval_batch_size=3)
        seen = []
        trainer.model.eval()
        for start in range(0, len(data), 3):
            samples = [data[i] for i in range(start, min(start + 3, len(data)))]
            with torch.enable_grad():
                trainer._chunk_errors(samples, on_prediction=lambda k, ref, pred: seen.append((k, ref.shape, ref.copy())))
        collected[batched] = seen
    assert [(k, s) for k, s, _ in collected[True]] == [(k, s) for k, s, _ in collected[False]]
    for (_, _, a), (_, _, b) in zip(collected[True], collected[False]):
        np.testing.assert_array_equal(a, b)


def test_validation_interval_and_patience_counts_checks(tmp_path):
    data = dataset(tmp_path, "newtonnet")
    trainer = make_trainer("newtonnet", epochs=5, batch_size=2, validation_interval=2)
    history = trainer.fit(data, data, test_data=data)
    assert [("validation" in row) for row in history] == [False, True, False, True, True]
    assert all("test" in row for row in history)  # test cadence unchanged (test_interval=1)
    # Early stopping only advances on validation epochs.
    trainer = make_trainer("newtonnet", epochs=10, batch_size=4, validation_interval=3,
                           early_stopping=True, early_stopping_patience=2)
    history = trainer.fit(data, data, output_dir=tmp_path / "run")
    checked = [row["epoch"] for row in history if "validation" in row]
    assert checked == [e for e in range(1, len(history) + 1) if e % 3 == 0 or e == 10]
    assert trainer.early_stopping["best_epoch"] in checked
    assert (tmp_path / "run" / "best.pt").is_file()


@pytest.mark.parametrize("kwargs", [dict(validation_interval=0), dict(validation_interval=1.5),
                                    dict(eval_batch_size=0), dict(eval_batch_size=True), dict(batched="yes")])
def test_invalid_batch_and_validation_options(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_unbatchable_samples_fall_back(tmp_path):
    """TorchMD-Net periodic structures are run one by one (single box per call)."""
    data = dataset(tmp_path, "torchmdnet")
    trainer = make_trainer("torchmdnet")
    samples = [dict(data[i], cell=np.eye(3) * 12., pbc=np.ones(3, dtype=bool)) for i in range(2)]
    assert trainer.backend.prepare_batch(samples, "cpu", torch.float64, model=trainer.model) is None
    assert trainer._batched_errors(samples) is None
    assert len(trainer._chunk_errors(samples)) == 2


def test_qmmm_batch_rejects_mixed_baseline_metadata(tmp_path):
    data = dataset(tmp_path, "tensornet_qmmm")
    trainer = make_trainer("tensornet_qmmm")
    samples = [dict(data[0], baseline_metadata={"mode": "a"}), dict(data[1], baseline_metadata={"mode": "b"})]
    with pytest.raises(ValueError, match="baseline_metadata"):
        trainer.backend.prepare_batch(samples, "cpu", torch.float64, model=trainer.model)
