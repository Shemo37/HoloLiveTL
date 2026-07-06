"""
Test setup: the test environment (Linux CI) has no GPU, audio devices, or
heavyweight ML packages, so torch and soundcard are stubbed when absent.
Tests only exercise pure-Python/numpy logic that doesn't touch them.
"""
import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_torch():
    # A plain-module stub with a real Tensor class: scipy's array-api
    # detection probes sys.modules["torch"].Tensor with issubclass(), which
    # blows up on a MagicMock.
    torch = types.ModuleType("torch")

    class Tensor:
        pass

    torch.Tensor = Tensor
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.set_num_threads = lambda n: None
    torch.jit = types.SimpleNamespace(load=MagicMock())
    return torch


try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = _stub_torch()

try:
    import soundcard  # noqa: F401
except (ImportError, OSError, AssertionError):
    # soundcard raises at import time when no audio subsystem exists
    sys.modules["soundcard"] = MagicMock()
