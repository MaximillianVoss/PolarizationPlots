# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
import importlib
import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def cpu_worker_count() -> int:
    return max(1, os.cpu_count() or 1)


@lru_cache(maxsize=1)
def _cupy_module() -> Any | None:
    if importlib.util.find_spec("cupy") is None:
        return None
    try:
        import cupy as cp  # type: ignore[import-not-found]

        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return None
        # Touch the active device once so broken CUDA runtimes fail here.
        cp.asarray([0.0], dtype=float)
        cp.cuda.get_current_stream().synchronize()
        return cp
    except Exception as ex:  # pragma: no cover - depends on local CUDA runtime
        logger.info("CUDA backend is unavailable: %s", ex)
        return None


def cuda_is_available() -> bool:
    return _cupy_module() is not None


def cupy_if_available(*, min_element_count: int = 0, element_count: int = 0) -> Any | None:
    if int(element_count) < int(min_element_count):
        return None
    return _cupy_module()


__all__ = [
    "cpu_worker_count",
    "cuda_is_available",
    "cupy_if_available",
]
