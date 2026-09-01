from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import genpareto

from hrrp_osr.data.errors import DataValidationError


@dataclass(frozen=True)
class GPDTail:
    threshold: float
    shape: float
    scale: float
    tail_count: int
    population_count: int
    side: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class JDSRGPDModel:
    matching_tails: tuple[GPDTail, ...]
    nonmatching_tails: tuple[GPDTail, ...]
    rho: float
    nonmatching_weight: float
    fit_set_ids_by_class: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matching_tails": [tail.to_dict() for tail in self.matching_tails],
            "nonmatching_tails": [tail.to_dict() for tail in self.nonmatching_tails],
            "rho": self.rho,
            "nonmatching_weight": self.nonmatching_weight,
            "fit_set_ids_by_class": [list(values) for values in self.fit_set_ids_by_class],
        }


def l2_normalize_profiles(profiles: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(profiles, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= epsilon):
        raise DataValidationError("JDSR profile has invalid L2 norm")
    return np.asarray(values / norms, dtype=np.float32)


def jdsr_reconstruction_errors(
    profiles: np.ndarray,
    dictionary: np.ndarray,
    *,
    sparsity: int,
    device: torch.device,
    batch_size: int = 32,
    excluded_atom_indices: np.ndarray | None = None,
    ridge: float = 1.0e-6,
) -> np.ndarray:
    """Greedy JDSR core: shared class blocks, view-specific atoms within each block."""
    observations = l2_normalize_profiles(profiles)
    atoms = l2_normalize_profiles(dictionary)
    if observations.ndim != 3:
        raise DataValidationError("JDSR observations must have shape [sets, views, bins]")
    if atoms.ndim != 3 or atoms.shape[-1] != observations.shape[-1]:
        raise DataValidationError("JDSR dictionary must have shape [classes, atoms, bins]")
    class_count, atoms_per_class, _ = atoms.shape
    if not 1 <= sparsity <= class_count:
        raise DataValidationError("JDSR sparsity must select between one and C class blocks")
    exclusions = None if excluded_atom_indices is None else np.asarray(excluded_atom_indices, dtype=int)
    if exclusions is not None and exclusions.shape != observations.shape[:2]:
        raise DataValidationError("JDSR excluded atom indices must have shape [sets, views]")
    dictionary_tensor = torch.from_numpy(atoms).to(device)
    all_errors: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        targets = torch.from_numpy(observations[start:stop]).to(device)
        batch, views, _ = targets.shape
        residual = targets.clone()
        selected_classes = torch.full((batch, sparsity), -1, dtype=torch.long, device=device)
        selected_atoms = torch.full((batch, views, sparsity), -1, dtype=torch.long, device=device)
        coefficients = torch.zeros((batch, views, sparsity), dtype=targets.dtype, device=device)
        for iteration in range(sparsity):
            correlations = torch.einsum("bvl,cal->bvca", residual, dictionary_tensor).abs()
            if exclusions is not None:
                batch_exclusions = torch.from_numpy(exclusions[start:stop]).to(device)
                exclusion_classes = torch.div(batch_exclusions, atoms_per_class, rounding_mode="floor")
                exclusion_atoms = batch_exclusions % atoms_per_class
                batch_indices = torch.arange(batch, device=device)[:, None].expand(batch, views)
                view_indices = torch.arange(views, device=device)[None, :].expand(batch, views)
                correlations[batch_indices, view_indices, exclusion_classes, exclusion_atoms] = -torch.inf
            best_values, best_atoms = correlations.max(dim=-1)
            group_scores = torch.sqrt(torch.sum(best_values.square(), dim=1))
            if iteration:
                group_scores.scatter_(1, selected_classes[:, :iteration], -torch.inf)
            chosen_class = group_scores.argmax(dim=1)
            chosen_atom = best_atoms.gather(
                2, chosen_class[:, None, None].expand(batch, views, 1)
            ).squeeze(2)
            selected_classes[:, iteration] = chosen_class
            selected_atoms[:, :, iteration] = chosen_atom
            support_vectors = []
            for support_index in range(iteration + 1):
                support_class = selected_classes[:, support_index]
                support_atom = selected_atoms[:, :, support_index]
                support_vectors.append(dictionary_tensor[support_class[:, None], support_atom])
            support = torch.stack(support_vectors, dim=-1)
            gram = torch.einsum("bvlk,bvlm->bvkm", support, support)
            identity = torch.eye(iteration + 1, device=device, dtype=targets.dtype)
            gram = gram + ridge * identity
            rhs = torch.einsum("bvlk,bvl->bvk", support, targets)
            solved = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
            coefficients[:, :, : iteration + 1] = solved
            residual = targets - torch.einsum("bvlk,bvk->bvl", support, solved)
        class_errors = []
        for class_index in range(class_count):
            reconstruction = torch.zeros_like(targets)
            for support_index in range(sparsity):
                active = (selected_classes[:, support_index] == class_index).to(targets.dtype)
                support_atom = selected_atoms[:, :, support_index]
                support_vector = dictionary_tensor[selected_classes[:, support_index, None], support_atom]
                reconstruction = reconstruction + (
                    support_vector
                    * coefficients[:, :, support_index, None]
                    * active[:, None, None]
                )
            squared_error = torch.sum((targets - reconstruction).square(), dim=(1, 2)) / views
            class_errors.append(squared_error)
        errors = torch.stack(class_errors, dim=1)
        if not torch.isfinite(errors).all():
            raise DataValidationError("JDSR solver produced non-finite reconstruction errors")
        all_errors.append(errors.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(all_errors, axis=0)


def _fit_right_tail(values: np.ndarray, tail_count: int, side: str) -> GPDTail:
    population = np.asarray(values, dtype=np.float64)
    population = population[np.isfinite(population)]
    if population.size < 5:
        raise DataValidationError("GPD requires at least five finite values per class")
    count = min(max(3, int(tail_count)), int(population.size) - 1)
    ordered = np.sort(population)
    threshold = float(ordered[-count - 1])
    exceedances = np.maximum(ordered[-count:] - threshold, np.finfo(np.float64).eps)
    if float(np.ptp(exceedances)) <= np.finfo(np.float64).eps * max(1.0, float(np.max(exceedances))):
        shape, scale = 0.0, float(np.mean(exceedances))
    else:
        shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        raise DataValidationError("GPD fit produced invalid parameters")
    return GPDTail(
        threshold=threshold, shape=float(shape), scale=float(scale),
        tail_count=count, population_count=int(population.size), side=side,
    )


def fit_dual_tail_gpd(
    reconstruction_errors: np.ndarray,
    true_labels: np.ndarray,
    set_ids: Sequence[str],
    *,
    class_count: int,
    rho: float,
    nonmatching_weight: float,
) -> JDSRGPDModel:
    errors = np.asarray(reconstruction_errors, dtype=np.float64)
    labels = np.asarray(true_labels, dtype=int)
    if errors.ndim != 2 or errors.shape[1] != class_count or len(errors) != len(labels) or len(errors) != len(set_ids):
        raise DataValidationError("dual-tail GPD fit arrays have inconsistent shapes")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise DataValidationError("dual-tail GPD fit received unknown labels")
    if not 0.0 < rho < 1.0 or not 0.0 <= nonmatching_weight <= 1.0:
        raise DataValidationError("invalid dual-tail GPD rho or weight")
    matching_tails = []
    nonmatching_tails = []
    fit_ids = []
    for class_index in range(class_count):
        selected = np.flatnonzero(labels == class_index)
        if selected.size < 5:
            raise DataValidationError(f"GPD class {class_index} has too few fit sets")
        class_errors = errors[selected]
        matching = class_errors[:, class_index]
        nonmatching = class_errors.sum(axis=1) - matching
        matching_count = int(np.ceil(selected.size * (1.0 - rho)))
        nonmatching_count = int(np.ceil(selected.size * rho))
        matching_tails.append(_fit_right_tail(matching, matching_count, "matching_right"))
        nonmatching_tails.append(_fit_right_tail(-nonmatching, nonmatching_count, "nonmatching_left"))
        fit_ids.append(tuple(str(set_ids[index]) for index in selected))
    return JDSRGPDModel(
        matching_tails=tuple(matching_tails),
        nonmatching_tails=tuple(nonmatching_tails),
        rho=float(rho),
        nonmatching_weight=float(nonmatching_weight),
        fit_set_ids_by_class=tuple(fit_ids),
    )


def _tail_outlier_probability(value: float, tail: GPDTail) -> float:
    exceedance = max(0.0, float(value) - tail.threshold)
    probability = float(genpareto.cdf(exceedance, tail.shape, loc=0.0, scale=tail.scale))
    return float(np.clip(probability, 0.0, 1.0))


def dual_tail_unknown_scores(
    reconstruction_errors: np.ndarray,
    model: JDSRGPDModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    errors = np.asarray(reconstruction_errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != len(model.matching_tails):
        raise DataValidationError("dual-tail inference error shape mismatch")
    candidates = errors.argmin(axis=1)
    matching_scores = np.empty(len(errors), dtype=np.float64)
    nonmatching_scores = np.empty(len(errors), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        matching = float(errors[index, candidate])
        nonmatching = float(errors[index].sum() - matching)
        matching_scores[index] = _tail_outlier_probability(matching, model.matching_tails[candidate])
        nonmatching_scores[index] = _tail_outlier_probability(-nonmatching, model.nonmatching_tails[candidate])
    combined = matching_scores + model.nonmatching_weight * nonmatching_scores
    if not np.all(np.isfinite(combined)):
        raise DataValidationError("dual-tail GPD produced non-finite scores")
    return candidates, combined, matching_scores, nonmatching_scores
