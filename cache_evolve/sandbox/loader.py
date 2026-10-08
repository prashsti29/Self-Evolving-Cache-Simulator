from __future__ import annotations

from typing import Type

from cache_evolve.sim.policy import Policy

from .validator import PolicyValidationError, validate_policy_source


def load_policy_class(source: str) -> Type[Policy]:
    validate_policy_source(source)
    namespace: dict = {"Policy": Policy}
    try:
        exec(compile(source, "<policy>", "exec"), namespace, namespace)
    except Exception as exc:
        raise PolicyValidationError(f"execution failed: {exc}") from exc

    candidates = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, Policy) and obj is not Policy
    ]
    if not candidates:
        raise PolicyValidationError("no Policy subclass found")
    if len(candidates) > 1:
        raise PolicyValidationError("multiple Policy subclasses found")
    return candidates[0]
