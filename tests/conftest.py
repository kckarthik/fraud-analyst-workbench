"""
Test bootstrap.

The pipeline scripts are run as standalone entry points from their own
directories (`cd enrichment && python pipeline.py`), so they import siblings
flatly: `from features import ...`. That means two different modules are both
named `features` — enrichment/features.py and model/features.py — and a plain
sys.path insert would make which one you get depend on ordering.

So: the one unambiguous package (backend/agent) goes on sys.path normally, and
anything with a colliding basename is loaded explicitly by file path under a
distinct module name.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# backend/agent is a real package with no name collisions.
sys.path.insert(0, str(ROOT / "backend"))


def _load(module_name: str, relative_path: str):
    """Import a module from an explicit path under an unambiguous name."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Registered at import time so test modules can `import enrichment_features`.
_load("enrichment_features", "enrichment/features.py")
