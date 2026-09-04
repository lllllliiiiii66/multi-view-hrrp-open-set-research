from __future__ import annotations

import ast
import hashlib
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hrrp_osr.models.official_cssr_1d import (
    MATCHED_LINEAR_CONTROL_1D,
    OFFICIAL_CSSR_REFERENCE_COMMIT,
    OFFICIAL_SEMANTICS_PCSSR_1D,
    MatchedLinearHead1D,
    OfficialPCSSRHead1D,
    official_pcssr_loss,
)

from .official_cssr_scores import (
    OFFICIAL_SCORE_RULES,
    build_official_score_templates,
    fit_score_normalization,
    official_g_p_pro,
    official_pcssr_pair_scores,
    raw_official_scores,
    standardize_and_integrate,
)


EXPECTED_FORMAL_CUDA_DEVICE_NAME = "NVIDIA GeForce RTX 4090"
FORMAL_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ORACLE_SEED = 20260904
ORACLE_DETERMINISTIC_REPEATS = 2
ORACLE_TOLERANCES = MappingProxyType(
    {
        "float32": MappingProxyType({"rtol": 1.0e-5, "atol": 1.0e-6}),
        "float64": MappingProxyType({"rtol": 1.0e-9, "atol": 1.0e-11}),
    }
)


OFFICIAL_CSSR_REFERENCE_HASHES = MappingProxyType(
    {
        "methods/cssr.py": (
            "0d23558c6a3cc4bf068036502a8ab43ee6278aecd91d96741f7375a142d9c5a3"
        ),
        "methods/cssr_ft.py": (
            "31244f194d91f6cab0bdf34eb14a0ed3b58f25b6c49a44042bb96baa9977fb16"
        ),
        "configs/basic.json": (
            "672375c6838004ae604509ba57098c7fefd17b6ac0f38e7c955fc8c09ba3192a"
        ),
        "configs/pcssr.json": (
            "353b0768cc6ee60ac76c110a22da8bdb5c15179260d4abeb2f43fee422d24c6b"
        ),
        "configs/pcssr/cifar10.json": (
            "ce5c7187cab1d8a7387526e459dc21c257f407e15e2304a91f618a8d8d34b0ab"
        ),
        "configs/pcssr/imagenet.json": (
            "170b8b7f86a2bde8fd409feaa96edfbfbd4226cc7ed9d1a564db8ca8a783b505"
        ),
    }
)


class OfficialCSSROracleError(RuntimeError):
    """The fixed official checkout or a differential check is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_official_checkout(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise OfficialCSSROracleError(f"official CSSR root is not a directory: {root}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OfficialCSSROracleError(
            f"cannot resolve official CSSR git commit at {root}"
        ) from exc
    commit = completed.stdout.strip()
    if commit != OFFICIAL_CSSR_REFERENCE_COMMIT:
        raise OfficialCSSROracleError(
            "official CSSR commit mismatch: "
            f"expected {OFFICIAL_CSSR_REFERENCE_COMMIT}, observed {commit}"
        )

    observed: dict[str, str] = {}
    for relative, expected in OFFICIAL_CSSR_REFERENCE_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise OfficialCSSROracleError(f"official CSSR file is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise OfficialCSSROracleError(
                f"official CSSR hash mismatch for {relative}: "
                f"expected {expected}, observed {actual}"
            )
        observed[relative] = actual
    return observed


def _load_selected_official_definitions(cssr_path: Path) -> dict[str, Any]:
    """Execute selected definitions directly from the hash-verified source.

    The complete official module imports dataset and vision dependencies that
    are irrelevant to this oracle.  Extracting the exact AST definitions keeps
    the comparison independent of those packages without vendoring or editing
    the official source.
    """

    selected = {
        "LinearClassifier",
        "sim_conv_layer",
        "AutoEncoder",
        "CSSRClassifier",
        "G_p_pro",
        "CSSRModel",
        "CSSRCriterion",
    }
    source = cssr_path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(cssr_path))
    body = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected
    ]
    found = {node.name for node in body}
    if found != selected:
        missing = sorted(selected - found)
        extra = sorted(found - selected)
        raise OfficialCSSROracleError(
            f"official AST selection changed; missing={missing}, extra={extra}"
        )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "_hash_verified_official_cssr_oracle",
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
    }
    exec(compile(module, str(cssr_path), "exec"), namespace)
    return namespace


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> float:
    if actual.shape != expected.shape:
        raise OfficialCSSROracleError(
            f"{name} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise OfficialCSSROracleError(f"{name} contains NaN or Inf")
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise OfficialCSSROracleError(f"official differential mismatch: {name}") from exc
    if actual.numel() == 0:
        return 0.0
    return float((actual.detach() - expected.detach()).abs().max().cpu().item())


def _official_probability(logits_2d: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits_2d, dim=1).mean(dim=(2, 3))


def _official_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[1]).to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return -(one_hot * torch.log(probabilities)).sum(dim=1).mean()


def _copy_pcssr_1d_to_official_2d(
    candidate: OfficialPCSSRHead1D,
    official: nn.Module,
) -> None:
    with torch.no_grad():
        for candidate_ae, official_ae in zip(
            candidate.class_autoencoders,
            official.class_aes,
            strict=True,
        ):
            official_ae.latent_conv[0].weight.copy_(
                candidate_ae.encoder[0].weight.unsqueeze(2)
            )
            official_ae.latent_deconv.weight.copy_(
                candidate_ae.decoder.weight.unsqueeze(2)
            )


def _fresh_official_score_model(
    cssr_model_type: type[nn.Module],
    *,
    num_classes: int,
) -> nn.Module:
    model = cssr_model_type.__new__(cssr_model_type)
    nn.Module.__init__(model)
    model.num_classes = int(num_classes)
    model.avg_feature = [[0, 0] for _ in range(num_classes)]
    model.powers = [8]
    model.avg_gram = [
        [[0, 0] for _ in range(num_classes)] for _ in model.powers
    ]
    model.enable_gram = True
    return model


def _dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        values = ORACLE_TOLERANCES["float32"]
        return float(values["rtol"]), float(values["atol"])
    if dtype == torch.float64:
        values = ORACLE_TOLERANCES["float64"]
        return float(values["rtol"]), float(values["atol"])
    raise OfficialCSSROracleError(f"unsupported oracle dtype: {dtype}")


def _backend_flag(container: Any, name: str) -> bool | None:
    return bool(getattr(container, name)) if hasattr(container, name) else None


def _set_backend_flag(container: Any, name: str, value: bool | None) -> None:
    if value is not None and hasattr(container, name):
        setattr(container, name, value)


def _validate_oracle_device(device: torch.device) -> dict[str, Any]:
    if device.type not in {"cpu", "cuda"}:
        raise OfficialCSSROracleError(
            f"official CSSR oracle supports only CPU or CUDA, observed {device.type}"
        )
    if device.type == "cpu":
        return {
            "device": str(device),
            "device_type": "cpu",
            "cuda_device_name": None,
            "expected_cuda_device_name": None,
            "formal_cuda_device_match": None,
            "cublas_workspace_config": None,
        }
    if not torch.cuda.is_available():
        raise OfficialCSSROracleError("CUDA oracle requested but CUDA is unavailable")
    observed_name = str(torch.cuda.get_device_name(device))
    if observed_name != EXPECTED_FORMAL_CUDA_DEVICE_NAME:
        raise OfficialCSSROracleError(
            "formal CUDA oracle GPU mismatch: "
            f"expected {EXPECTED_FORMAL_CUDA_DEVICE_NAME!r}, observed {observed_name!r}"
        )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace != FORMAL_CUBLAS_WORKSPACE_CONFIG:
        raise OfficialCSSROracleError(
            "formal CUDA oracle requires "
            f"CUBLAS_WORKSPACE_CONFIG={FORMAL_CUBLAS_WORKSPACE_CONFIG!r}, "
            f"observed {workspace!r}"
        )
    return {
        "device": str(device),
        "device_type": "cuda",
        "cuda_device_name": observed_name,
        "expected_cuda_device_name": EXPECTED_FORMAL_CUDA_DEVICE_NAME,
        "formal_cuda_device_match": True,
        "cublas_workspace_config": workspace,
    }


@contextmanager
def _deterministic_oracle_runtime(
    device: torch.device,
) -> Iterator[dict[str, Any]]:
    """Run the oracle under the frozen deterministic numerical contract.

    Backend switches are restored afterwards so a diagnostic CPU call does not
    silently mutate the caller's training runtime.  The yielded record captures
    the settings that were active while every differential operation ran.
    """

    device_record = _validate_oracle_device(device)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only_getter = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )
    old_warn_only = bool(warn_only_getter())
    old_cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    old_matmul_tf32 = _backend_flag(torch.backends.cuda.matmul, "allow_tf32")
    old_cudnn_tf32 = _backend_flag(torch.backends.cudnn, "allow_tf32")
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        _set_backend_flag(torch.backends.cuda.matmul, "allow_tf32", False)
        _set_backend_flag(torch.backends.cudnn, "allow_tf32", False)
        runtime = {
            **device_record,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": bool(warn_only_getter()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": _backend_flag(
                torch.backends.cuda.matmul,
                "allow_tf32",
            ),
            "cudnn_allow_tf32": _backend_flag(torch.backends.cudnn, "allow_tf32"),
        }
        if (
            runtime["deterministic_algorithms"] is not True
            or runtime["deterministic_algorithms_warn_only"] is not False
            or runtime["cudnn_benchmark"] is not False
            or runtime["cuda_matmul_allow_tf32"] not in {False, None}
            or runtime["cudnn_allow_tf32"] not in {False, None}
        ):
            raise OfficialCSSROracleError(
                "failed to establish the deterministic oracle runtime"
            )
        yield runtime
    finally:
        torch.use_deterministic_algorithms(
            old_deterministic,
            warn_only=old_warn_only,
        )
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        _set_backend_flag(torch.backends.cuda.matmul, "allow_tf32", old_matmul_tf32)
        _set_backend_flag(torch.backends.cudnn, "allow_tf32", old_cudnn_tf32)


def _assert_equal_indices(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if actual.shape != expected.shape:
        raise OfficialCSSROracleError(
            f"{name} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != torch.long or expected.dtype != torch.long:
        raise OfficialCSSROracleError(f"{name} must use torch.long indices")
    if not bool(torch.equal(actual, expected)):
        raise OfficialCSSROracleError(f"official differential mismatch: {name}")


def _audit_clip_boundaries(
    namespace: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Exercise both official clip boundaries, including exact-boundary values."""

    num_classes = 2
    channels = 4
    length = 3
    gamma = 0.1
    requested_magnitudes = torch.tensor(
        [0.0, 99.0, 100.0, 101.0, 137.5],
        device=device,
        dtype=dtype,
    )
    features = (
        requested_magnitudes[:, None, None] / (gamma * channels)
    ).expand(-1, channels, length).contiguous()

    candidate = OfficialPCSSRHead1D(
        num_classes=num_classes,
        input_channels=channels,
        latent_channels=2,
        gamma=gamma,
        clip_length=100.0,
    ).to(device=device, dtype=dtype)
    official_pcssr = namespace["CSSRClassifier"](
        channels,
        num_classes,
        {
            "ae_hidden": [],
            "ae_latent": 2,
            "error_measure": "L1",
            "model": "pcssr",
            "gamma": gamma,
        },
    ).to(device=device, dtype=dtype)
    official_rcssr = namespace["CSSRClassifier"](
        channels,
        num_classes,
        {
            "ae_hidden": [],
            "ae_latent": 2,
            "error_measure": "L1",
            "model": "rcssr",
            "gamma": gamma,
        },
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        for module in (candidate, official_pcssr, official_rcssr):
            for parameter in module.parameters():
                parameter.zero_()
        candidate_logits = candidate(features).logits
        official_negative = official_pcssr(features.unsqueeze(2)).squeeze(2)
        official_positive = official_rcssr(features.unsqueeze(2)).squeeze(2)

    preclip = requested_magnitudes[:, None, None].expand(
        -1,
        num_classes,
        length,
    )
    expected_negative = torch.clamp(-preclip, min=-100.0, max=100.0)
    expected_positive = torch.clamp(preclip, min=-100.0, max=100.0)
    differences = {
        "clip_lower_candidate_vs_official": _assert_close(
            "pCSSR lower clip candidate versus official",
            candidate_logits,
            official_negative,
            rtol=rtol,
            atol=atol,
        ),
        "clip_lower_official_vs_literal": _assert_close(
            "pCSSR lower clip official versus literal",
            official_negative,
            expected_negative,
            rtol=rtol,
            atol=atol,
        ),
        "clip_upper_official_vs_literal": _assert_close(
            "RCSSR upper clip official versus literal",
            official_positive,
            expected_positive,
            rtol=rtol,
            atol=atol,
        ),
    }
    lower_observed = official_negative[:, 0, 0]
    upper_observed = official_positive[:, 0, 0]
    expected_negative_vector = torch.clamp(
        -requested_magnitudes,
        min=-100.0,
        max=100.0,
    )
    expected_positive_vector = torch.clamp(
        requested_magnitudes,
        min=-100.0,
        max=100.0,
    )
    _assert_close(
        "pCSSR lower clip boundary vector",
        lower_observed,
        expected_negative_vector,
        rtol=rtol,
        atol=atol,
    )
    _assert_close(
        "RCSSR upper clip boundary vector",
        upper_observed,
        expected_positive_vector,
        rtol=rtol,
        atol=atol,
    )
    return differences, {
        "passed": True,
        "bounds": [-100.0, 100.0],
        "requested_preclip_magnitudes": [0.0, 99.0, 100.0, 101.0, 137.5],
        "lower_interior_exercised": True,
        "lower_exact_boundary_exercised": True,
        "lower_saturation_exercised": True,
        "upper_interior_exercised": True,
        "upper_exact_boundary_exercised": True,
        "upper_saturation_exercised": True,
        "upper_boundary_reference_only": (
            "official RCSSR positive-sign path; the candidate implements pCSSR only"
        ),
    }


def _audit_dtype(
    namespace: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    rtol, atol = _dtype_tolerances(dtype)
    num_classes = 3
    channels = 4
    latent_channels = 2
    length = 7
    gamma = 0.1
    cpu_generator = torch.Generator(device="cpu").manual_seed(ORACLE_SEED)

    base_features = torch.randn(
        9,
        channels,
        length,
        generator=cpu_generator,
        dtype=torch.float64,
    )
    base_features = base_features * 0.45 + 0.15
    features = base_features.to(device=device, dtype=dtype)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device)

    candidate = OfficialPCSSRHead1D(
        num_classes=num_classes,
        input_channels=channels,
        latent_channels=latent_channels,
        gamma=gamma,
    ).to(device=device, dtype=dtype)
    official = namespace["CSSRClassifier"](
        channels,
        num_classes,
        {
            "ae_hidden": [],
            "ae_latent": latent_channels,
            "error_measure": "L1",
            "model": "pcssr",
            "gamma": gamma,
        },
    ).to(device=device, dtype=dtype)
    _copy_pcssr_1d_to_official_2d(candidate, official)

    candidate_features = features.detach().clone().requires_grad_(True)
    official_features = features.detach().clone().unsqueeze(2).requires_grad_(True)
    candidate_output = candidate(candidate_features)
    official_logits_2d = official(official_features)
    official_logits = official_logits_2d.squeeze(2)
    official_criterion = namespace["CSSRCriterion"]("softmax_avg", False).to(
        device=device,
        dtype=dtype,
    )
    official_probabilities = official_criterion(official_logits_2d, prob=True)
    official_reconstructions: list[torch.Tensor] = []
    official_latents: list[torch.Tensor] = []
    official_errors: list[torch.Tensor] = []
    for official_ae in official.class_aes:
        reconstruction, latent = official_ae(official_features)
        official_reconstructions.append(reconstruction.squeeze(2))
        official_latents.append(latent.squeeze(2))
        official_errors.append(
            torch.norm(reconstruction - official_features, p=1, dim=1).squeeze(1)
        )
    official_reconstruction_tensor = torch.stack(official_reconstructions, dim=1)
    official_latent_tensor = torch.stack(official_latents, dim=1)
    official_error_tensor = torch.stack(official_errors, dim=1)

    differences, clip_coverage = _audit_clip_boundaries(
        namespace,
        device=device,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )
    differences["criterion_probability_literal"] = _assert_close(
        "official criterion probability",
        official_probabilities,
        _official_probability(official_logits_2d),
        rtol=rtol,
        atol=atol,
    )
    differences["pcssr_reconstructions"] = _assert_close(
        "pcssr reconstructions",
        candidate_output.reconstructions,
        official_reconstruction_tensor,
        rtol=rtol,
        atol=atol,
    )
    differences["pcssr_latents"] = _assert_close(
        "pcssr latents",
        candidate_output.latents,
        official_latent_tensor,
        rtol=rtol,
        atol=atol,
    )
    differences["pcssr_reconstruction_errors"] = _assert_close(
        "pcssr reconstruction errors",
        candidate_output.reconstruction_errors,
        official_error_tensor,
        rtol=rtol,
        atol=atol,
    )
    differences["pcssr_logits"] = _assert_close(
        "pcssr logits",
        candidate_output.logits,
        official_logits,
        rtol=rtol,
        atol=atol,
    )
    differences["pcssr_probabilities"] = _assert_close(
        "pcssr probabilities",
        candidate_output.probabilities,
        official_probabilities,
        rtol=rtol,
        atol=atol,
    )
    candidate_loss = official_pcssr_loss(candidate_output.probabilities, targets)
    reference_loss = _official_loss(official_probabilities, targets)
    differences["pcssr_loss"] = _assert_close(
        "pcssr loss",
        candidate_loss,
        reference_loss,
        rtol=rtol,
        atol=atol,
    )
    candidate_loss.backward()
    reference_loss.backward()
    differences["pcssr_input_gradient"] = _assert_close(
        "pcssr input gradient",
        candidate_features.grad,
        official_features.grad.squeeze(2),
        rtol=rtol,
        atol=atol,
    )
    for index, (candidate_ae, official_ae) in enumerate(
        zip(candidate.class_autoencoders, official.class_aes, strict=True)
    ):
        differences[f"pcssr_encoder_gradient_class_{index}"] = _assert_close(
            f"pcssr encoder gradient class {index}",
            candidate_ae.encoder[0].weight.grad,
            official_ae.latent_conv[0].weight.grad.squeeze(2),
            rtol=rtol,
            atol=atol,
        )
        differences[f"pcssr_decoder_gradient_class_{index}"] = _assert_close(
            f"pcssr decoder gradient class {index}",
            candidate_ae.decoder.weight.grad,
            official_ae.latent_deconv.weight.grad.squeeze(2),
            rtol=rtol,
            atol=atol,
        )

    matched_linear = MatchedLinearHead1D(
        num_classes=num_classes,
        input_channels=channels,
        gamma=gamma,
    ).to(device=device, dtype=dtype)
    official_linear = namespace["LinearClassifier"](
        channels,
        num_classes,
        {"gamma": gamma},
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        official_linear.cls.weight.copy_(matched_linear.classifier.weight.unsqueeze(2))
    candidate_linear_features = features.detach().clone().requires_grad_(True)
    official_linear_features = features.detach().clone().unsqueeze(2).requires_grad_(True)
    matched_output = matched_linear(candidate_linear_features)
    official_linear_logits_2d = official_linear(official_linear_features)
    official_linear_probabilities = official_criterion(
        official_linear_logits_2d,
        prob=True,
    )
    differences["linear_logits"] = _assert_close(
        "matched linear logits",
        matched_output.logits,
        official_linear_logits_2d.squeeze(2),
        rtol=rtol,
        atol=atol,
    )
    differences["linear_probabilities"] = _assert_close(
        "matched linear probabilities",
        matched_output.probabilities,
        official_linear_probabilities,
        rtol=rtol,
        atol=atol,
    )
    candidate_linear_loss = official_pcssr_loss(matched_output.probabilities, targets)
    official_linear_loss = _official_loss(official_linear_probabilities, targets)
    differences["linear_loss"] = _assert_close(
        "matched linear loss",
        candidate_linear_loss,
        official_linear_loss,
        rtol=rtol,
        atol=atol,
    )
    candidate_linear_loss.backward()
    official_linear_loss.backward()
    differences["linear_input_gradient"] = _assert_close(
        "matched linear input gradient",
        candidate_linear_features.grad,
        official_linear_features.grad.squeeze(2),
        rtol=rtol,
        atol=atol,
    )
    differences["linear_weight_gradient"] = _assert_close(
        "matched linear weight gradient",
        matched_linear.classifier.weight.grad,
        official_linear.cls.weight.grad.squeeze(2),
        rtol=rtol,
        atol=atol,
    )

    template_predictions = targets
    candidate_templates = build_official_score_templates(
        features.detach(),
        template_predictions,
        num_classes=num_classes,
        power=8,
    )
    official_score_model = _fresh_official_score_model(
        namespace["CSSRModel"],
        num_classes=num_classes,
    )
    official_score_model.cal_feature_prototype(
        features.detach().unsqueeze(2),
        template_predictions.detach().cpu().numpy(),
    )
    official_first_order, official_gram_list = (
        official_score_model.obtain_usable_feature_prototype()
    )
    official_first_order_1d = official_first_order[:, 0, :, 0, 0]
    official_gram = official_gram_list[0]
    differences["first_order_template"] = _assert_close(
        "first-order template",
        candidate_templates.first_order,
        official_first_order_1d,
        rtol=rtol,
        atol=atol,
    )
    differences["gram_template"] = _assert_close(
        "Gram template",
        candidate_templates.gram,
        official_gram,
        rtol=rtol,
        atol=atol,
    )
    direct_official_gram = namespace["G_p_pro"](features.detach().unsqueeze(2), 8)
    differences["g_p_pro"] = _assert_close(
        "G_p_pro",
        official_g_p_pro(features.detach(), power=8),
        direct_official_gram,
        rtol=rtol,
        atol=atol,
    )

    test_features = (features.detach()[:6] * 1.17 - 0.09).contiguous()
    test_logits = candidate(test_features).logits.detach()
    predicted_class = torch.tensor([0, 1, 2, 2, 1, 0], device=device)
    candidate_scores = raw_official_scores(
        test_features,
        test_logits,
        predicted_class,
        candidate_templates,
    )
    numpy_logits = test_logits.detach().cpu().numpy()
    numpy_features = test_features.detach().cpu().numpy()
    numpy_predicted = predicted_class.detach().cpu().numpy()
    selected_logits = numpy_logits[np.arange(len(numpy_predicted)), numpy_predicted]
    activation = np.abs(numpy_features).mean(axis=1)
    reference_s1 = (selected_logits / activation / activation).reshape(
        len(numpy_predicted), -1
    ).mean(axis=1)
    reference_s2_map = official_score_model.get_feature_prototype_deviation(
        test_features.unsqueeze(2),
        numpy_predicted,
    )
    reference_s3_map = official_score_model.get_feature_gram_deviation(
        test_features.unsqueeze(2),
        numpy_predicted,
    )
    reference_score_values = np.stack(
        (
            reference_s1,
            reference_s2_map.reshape(len(numpy_predicted), -1).mean(axis=1),
            reference_s3_map.reshape(len(numpy_predicted), -1).mean(axis=1),
        ),
        axis=1,
    )
    reference_score_tensor = torch.as_tensor(
        reference_score_values,
        device=device,
        dtype=dtype,
    )
    differences["raw_s1"] = _assert_close(
        "raw S1",
        candidate_scores.s1,
        reference_score_tensor[:, 0],
        rtol=rtol,
        atol=atol,
    )
    differences["raw_s2"] = _assert_close(
        "raw S2",
        candidate_scores.s2,
        reference_score_tensor[:, 1],
        rtol=rtol,
        atol=atol,
    )
    differences["raw_s3"] = _assert_close(
        "raw S3",
        candidate_scores.s3,
        reference_score_tensor[:, 2],
        rtol=rtol,
        atol=atol,
    )
    abs_s2 = (
        test_features.abs()
        * candidate_templates.first_order.index_select(0, predicted_class).unsqueeze(-1)
    ).mean(dim=(1, 2))
    signed_abs_delta = float(
        (candidate_scores.s2.detach() - abs_s2.detach()).abs().max().cpu().item()
    )
    if signed_abs_delta <= (10.0 * atol):
        raise OfficialCSSROracleError(
            "S2 fixture does not distinguish signed test semantics from abs test semantics"
        )

    normalization_values = torch.cat(
        (
            candidate_scores.values,
            candidate_scores.values * 1.09 + torch.tensor(
                [0.03, -0.02, 0.07],
                device=device,
                dtype=dtype,
            ),
        ),
        dim=0,
    )
    normalization = fit_score_normalization(normalization_values)
    candidate_standardized = standardize_and_integrate(candidate_scores, normalization)
    reference_normalization_values = np.concatenate(
        (
            reference_score_values.astype(np.float64, copy=False),
            reference_score_values.astype(np.float64, copy=False) * 1.09
            + np.asarray([0.03, -0.02, 0.07], dtype=np.float64),
        ),
        axis=0,
    )
    reference_mean = reference_normalization_values.mean(axis=0)
    reference_std = reference_normalization_values.std(axis=0, ddof=0)
    score_numpy = reference_score_values.astype(np.float64, copy=False)
    reference_standardized_numpy = (score_numpy - reference_mean) / (
        reference_std + 1.0e-8
    )
    reference_standardized = torch.as_tensor(
        reference_standardized_numpy,
        device=device,
        dtype=torch.float64,
    )
    differences["normalization_mean"] = _assert_close(
        "normalization mean",
        normalization.mean,
        torch.as_tensor(reference_mean, device=device, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )
    differences["normalization_std"] = _assert_close(
        "normalization population std",
        normalization.std,
        torch.as_tensor(reference_std, device=device, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )
    differences["standardized_scores"] = _assert_close(
        "standardized S1/S2/S3",
        candidate_standardized.standardized,
        reference_standardized,
        rtol=rtol,
        atol=atol,
    )
    differences["integrated_score"] = _assert_close(
        "integrated S1+S2+S3",
        candidate_standardized.integrated,
        reference_standardized.sum(dim=1),
        rtol=rtol,
        atol=atol,
    )

    pair_features = test_features[:4].reshape(2, 2, channels, length)
    pair_logits = test_logits[:4].reshape(2, 2, num_classes, length)
    pair_probabilities = torch.softmax(pair_logits, dim=2).mean(dim=-1)
    pair_scores = official_pcssr_pair_scores(
        pair_features,
        pair_logits,
        pair_probabilities,
        candidate_templates,
        normalization,
    )
    flat_official_pair_features = pair_features.reshape(
        4,
        channels,
        length,
    ).unsqueeze(2)
    with torch.no_grad():
        flat_official_pair_logits_2d = official(flat_official_pair_features)
        flat_official_pair_probabilities = official_criterion(
            flat_official_pair_logits_2d,
            prob=True,
        )
    official_pair_logits = flat_official_pair_logits_2d.squeeze(2).reshape(
        2,
        2,
        num_classes,
        length,
    )
    official_view_probabilities = flat_official_pair_probabilities.reshape(
        2,
        2,
        num_classes,
    )
    differences["pair_view_logits"] = _assert_close(
        "pair view logits",
        pair_logits,
        official_pair_logits,
        rtol=rtol,
        atol=atol,
    )
    differences["pair_view_probabilities"] = _assert_close(
        "pair view probabilities",
        pair_probabilities,
        official_view_probabilities,
        rtol=rtol,
        atol=atol,
    )
    official_view_probabilities_numpy = (
        official_view_probabilities.detach().cpu().numpy()
    )
    reference_pair_probabilities_numpy = official_view_probabilities_numpy.mean(axis=1)
    reference_pair_probabilities = torch.as_tensor(
        reference_pair_probabilities_numpy,
        device=device,
        dtype=dtype,
    )
    differences["pair_probabilities"] = _assert_close(
        "two-view pair probability",
        pair_scores.pair_probabilities,
        reference_pair_probabilities,
        rtol=rtol,
        atol=atol,
    )
    reference_pair_prediction = torch.as_tensor(
        np.argmax(reference_pair_probabilities_numpy, axis=1),
        device=device,
        dtype=torch.long,
    )
    _assert_equal_indices(
        "two-view pair argmax",
        pair_scores.predicted_class,
        reference_pair_prediction,
    )
    repeated_pair_prediction = reference_pair_prediction[:, None].expand(-1, 2).reshape(-1)
    flat_pair_features = pair_features.reshape(4, channels, length)
    flat_pair_logits = pair_logits.reshape(4, num_classes, length)
    flat_numpy_features = flat_pair_features.detach().cpu().numpy()
    flat_numpy_logits = flat_pair_logits.detach().cpu().numpy()
    flat_numpy_prediction = repeated_pair_prediction.detach().cpu().numpy()
    flat_selected_logits = flat_numpy_logits[
        np.arange(len(flat_numpy_prediction)), flat_numpy_prediction
    ]
    flat_activation = np.abs(flat_numpy_features).mean(axis=1)
    flat_s1 = (flat_selected_logits / flat_activation / flat_activation).reshape(
        4, -1
    ).mean(axis=1)
    flat_s2 = official_score_model.get_feature_prototype_deviation(
        flat_pair_features.unsqueeze(2), flat_numpy_prediction
    ).reshape(4, -1).mean(axis=1)
    flat_s3 = official_score_model.get_feature_gram_deviation(
        flat_pair_features.unsqueeze(2), flat_numpy_prediction
    ).reshape(4, -1).mean(axis=1)
    flat_reference = np.stack((flat_s1, flat_s2, flat_s3), axis=1).astype(
        np.float64,
        copy=False,
    )
    reference_per_view_raw = torch.as_tensor(
        flat_reference.reshape(2, 2, 3),
        device=device,
        dtype=dtype,
    )
    differences["pair_per_view_raw_scores"] = _assert_close(
        "two-view per-view raw scores",
        pair_scores.per_view_raw,
        reference_per_view_raw,
        rtol=rtol,
        atol=atol,
    )
    reference_per_view_standardized_numpy = (
        (flat_reference - reference_mean) / (reference_std + 1.0e-8)
    ).reshape(2, 2, 3)
    reference_per_view_standardized = torch.as_tensor(
        reference_per_view_standardized_numpy,
        device=device,
        dtype=torch.float64,
    )
    differences["pair_per_view_standardized_scores"] = _assert_close(
        "two-view per-view standardized scores",
        pair_scores.per_view_standardized,
        reference_per_view_standardized,
        rtol=rtol,
        atol=atol,
    )
    reference_pair_components_numpy = reference_per_view_standardized_numpy.mean(
        axis=1
    )
    reference_pair_components = torch.as_tensor(
        reference_pair_components_numpy,
        device=device,
        dtype=torch.float64,
    )
    differences["pair_standardized_components"] = _assert_close(
        "two-view standardized score components",
        pair_scores.pair_standardized_components,
        reference_pair_components,
        rtol=rtol,
        atol=atol,
    )
    differences["pair_full_integration"] = _assert_close(
        "two-view full score integration",
        pair_scores.full_knownness,
        reference_pair_components.sum(dim=1),
        rtol=rtol,
        atol=atol,
    )

    if tuple(pair_scores.knownness_by_rule) != OFFICIAL_SCORE_RULES:
        raise OfficialCSSROracleError(
            "official pair knownness rule order or membership changed"
        )
    if tuple(pair_scores.unknown_scores_by_rule) != OFFICIAL_SCORE_RULES:
        raise OfficialCSSROracleError(
            "official pair unknown-score rule order or membership changed"
        )
    reference_max_pair_probability = reference_pair_probabilities.max(dim=1).values
    reference_knownness = {
        "S1": reference_pair_components[:, 0],
        "S2": reference_pair_components[:, 1],
        "S3": reference_pair_components[:, 2],
        "S1+S2": reference_pair_components[:, 0] + reference_pair_components[:, 1],
        "S1+S3": reference_pair_components[:, 0] + reference_pair_components[:, 2],
        "S2+S3": reference_pair_components[:, 1] + reference_pair_components[:, 2],
        "full": reference_pair_components.sum(dim=1),
        "pcssr_max_pair_probability": reference_max_pair_probability,
    }
    for rule in OFFICIAL_SCORE_RULES:
        differences[f"pair_knownness::{rule}"] = _assert_close(
            f"two-view knownness rule {rule}",
            pair_scores.knownness_by_rule[rule],
            reference_knownness[rule],
            rtol=rtol,
            atol=atol,
        )
        differences[f"pair_unknown_score::{rule}"] = _assert_close(
            f"two-view unknown-score rule {rule}",
            pair_scores.unknown_scores_by_rule[rule],
            -reference_knownness[rule],
            rtol=rtol,
            atol=atol,
        )

    return {
        "passed": True,
        "dtype": str(dtype).removeprefix("torch."),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_differences": differences,
        "s2_signed_vs_abs_max_delta": signed_abs_delta,
        "clip_boundary_checks": clip_coverage,
        "pair_checks": {
            "passed": True,
            "probability": "passed",
            "argmax": "passed",
            "knownness_rules": list(OFFICIAL_SCORE_RULES),
            "unknown_score_direction": "negative_knownness",
            "independent_reference": (
                "hash-verified official head outputs plus independent NumPy "
                "pair aggregation and score-rule construction"
            ),
        },
    }


def audit_official_cssr_oracle(
    official_root: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Hard precondition for official-semantics CSSR HRRP experiments.

    This verifies the exact official checkout identity, executes selected AST
    definitions from its fixed ``methods/cssr.py``, and differentially checks
    float32 and float64 forward, loss, gradients, S1/S2/S3, population-score
    normalization, and the symmetric two-view integration used by this pilot.
    A mismatch raises :class:`OfficialCSSROracleError`; only a fully passing
    audit returns a record suitable for a smoke/pilot precondition artifact.
    """

    root = Path(official_root).expanduser().resolve()
    resolved_device = torch.device(device)
    hashes = _verify_official_checkout(root)
    namespace = _load_selected_official_definitions(root / "methods/cssr.py")
    dtype_records: dict[str, Any] = {}
    devices = [] if resolved_device.type == "cpu" else [resolved_device.index or 0]
    with _deterministic_oracle_runtime(resolved_device) as runtime_record:
        with torch.random.fork_rng(devices=devices):
            for dtype in (torch.float32, torch.float64):
                repeated: list[dict[str, Any]] = []
                for _ in range(ORACLE_DETERMINISTIC_REPEATS):
                    torch.manual_seed(ORACLE_SEED)
                    if resolved_device.type == "cuda":
                        torch.cuda.manual_seed_all(ORACLE_SEED)
                    repeated.append(
                        _audit_dtype(
                            namespace,
                            device=resolved_device,
                            dtype=dtype,
                        )
                    )
                if any(record != repeated[0] for record in repeated[1:]):
                    raise OfficialCSSROracleError(
                        f"{repeated[0]['dtype']} oracle is not deterministic across "
                        f"{ORACLE_DETERMINISTIC_REPEATS} identical repeats"
                    )
                dtype_records[repeated[0]["dtype"]] = {
                    **repeated[0],
                    "deterministic_repeat": {
                        "passed": True,
                        "repeats": ORACLE_DETERMINISTIC_REPEATS,
                        "record_equality": "exact",
                    },
                }
    oracle_contract = {
        "seed": ORACLE_SEED,
        "required_dtypes": ["float32", "float64"],
        "tolerances": {
            name: dict(values) for name, values in ORACLE_TOLERANCES.items()
        },
        "clip_bounds": [-100.0, 100.0],
        "pair_score_rules": list(OFFICIAL_SCORE_RULES),
        "pair_probability_rule": "arithmetic_mean_of_two_view_probabilities",
        "pair_prediction_rule": "argmax_pair_probability",
        "unknown_score_direction": "negative_knownness",
        "deterministic_repeats": ORACLE_DETERMINISTIC_REPEATS,
    }
    return {
        "passed": True,
        "status": "passed",
        "official_root": str(root),
        "official_commit": OFFICIAL_CSSR_REFERENCE_COMMIT,
        "file_sha256": hashes,
        "verified_file_sha256": hashes,
        "source_execution": "selected_ast_definitions_from_hash_verified_methods/cssr.py",
        "method_ids": {
            "official": OFFICIAL_SEMANTICS_PCSSR_1D,
            "matched_linear_control": MATCHED_LINEAR_CONTROL_1D,
        },
        "device": str(resolved_device),
        "cuda_device_name": runtime_record["cuda_device_name"],
        "runtime_contract": runtime_record,
        "oracle_contract": oracle_contract,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "float32": "passed",
        "float64": "passed",
        "dtype_checks": dtype_records,
    }
