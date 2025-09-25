from typing import Any, Dict

import numpy as np
from art.attacks.evasion import (
    FastGradientMethod,
    MomentumIterativeMethod,
    ProjectedGradientDescent,
    SquareAttack,
)


def make_attack(
    classifier,
    attack_name: str,
    eps: float,
    attack_params: Dict[str, Any] | None = None,
):
    if attack_params is None:
        attack_params = {}

    batch_size = int(attack_params.get("batch_size", 128))
    norm = attack_params.get("norm", np.inf)
    eps = float(eps)

    name = attack_name.upper()
    if name == "FGSM":
        eps_step = float(attack_params.get("eps_step", eps))
        return FastGradientMethod(
            classifier, norm=norm, eps=eps, eps_step=eps_step, batch_size=batch_size
        )

    eps_step = attack_params.get("eps_step", None)
    if eps_step is None:
        eps_step_ratio = attack_params.get(
            "eps_step_ratio", 0.25 if name == "PGD" else 0.10
        )
        eps_step = eps * float(eps_step_ratio)
    eps_step = float(max(eps_step, 1e-6))
    max_iter = int(attack_params.get("max_iter", 40 if name == "PGD" else 10))

    if name == "PGD":
        return ProjectedGradientDescent(
            classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=False,
            batch_size=batch_size,
        )
    if name == "MIM":
        decay = float(attack_params.get("decay", 1.0))
        return MomentumIterativeMethod(
            classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=False,
            batch_size=batch_size,
        )
    if name in {"SQUARE", "SQUAREATTACK"}:
        max_iter = int(attack_params.get("max_iter", 1000))
        p_init = float(attack_params.get("p_init", 0.8))
        nb_restarts = int(attack_params.get("nb_restarts", 1))
        return SquareAttack(
            classifier,
            norm=norm,
            eps=eps,
            max_iter=max_iter,
            p_init=p_init,
            nb_restarts=nb_restarts,
            batch_size=batch_size,
        )

    raise ValueError(
        f"Unsupported attack_name '{attack_name}'. Choose from FGSM, PGD, MIM, SQUARE."
    )
