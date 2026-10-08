from __future__ import annotations

import ast

FORBIDDEN_NAMES = frozenset(
    {"__import__", "eval", "exec", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals", "input"}
)
FORBIDDEN_MODULES = frozenset({"os", "sys", "subprocess", "socket", "pathlib", "importlib", "builtins"})
ALLOWED_MODULES = frozenset({"cache_evolve.sim.policy", "collections"})


class PolicyValidationError(ValueError):
    pass


class _PolicyValidator(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in ALLOWED_MODULES:
                continue
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES:
                raise PolicyValidationError(f"import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module not in ALLOWED_MODULES:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULES or node.module.startswith("cache_evolve."):
                raise PolicyValidationError(f"import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            raise PolicyValidationError(f"name not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr.startswith("__")
            and node.attr.endswith("__")
            and node.attr not in {"__init__"}
        ):
            raise PolicyValidationError(f"dunder attribute access not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            raise PolicyValidationError(f"call not allowed: {node.func.id}")
        self.generic_visit(node)


def validate_policy_source(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PolicyValidationError(f"syntax error: {exc}") from exc
    _PolicyValidator().visit(tree)
