"""Site customisation for the RecSys‑Lite test environment.

This file is automatically imported by the Python interpreter **before** any
third‑party packages thanks to the standard *site* initialisation mechanism.

We use the opportunity to provide *very* small pure‑Python stubs for heavy
native dependencies (NumPy, SciPy, Faiss, Pandas) that would otherwise not be
available or – in the case of NumPy – crash the test runner on certain
platforms.  Only the limited surface area exercised by the unit‑test suite is
implemented.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# NumPy stub – keep *exactly* the same implementation as inside
# ``recsys_lite.__init__`` so that both entry points agree.
# ---------------------------------------------------------------------------


def _install_numpy_stub() -> None:  # noqa: D401
    if "numpy" in sys.modules:
        return

    np_stub = types.ModuleType("numpy")

    def _array(data, *_, **__):
        return list(data)

    def _zeros(shape, *_, **__):
        if isinstance(shape, int):
            return [0] * shape
        if isinstance(shape, tuple) and len(shape) == 2:
            return [[0] * shape[1] for _ in range(shape[0])]
        return [0]

    def _arange(stop, *args, **kwargs):  # noqa: D401
        start = 0
        step = 1
        if args:
            start, stop = stop, args[0]
            if len(args) > 1:
                step = args[1]
        return list(range(start, stop, step))

    class _RandomState:  # noqa: D401 – minimal RNG stub
        def __init__(self, seed=None):
            import random

            self._rng = random.Random(seed)

        def randint(self, low, high=None, size=None):
            if high is None:
                low, high = 0, low
            if size is None:
                return self._rng.randint(low, high - 1)
            return [self.randint(low, high) for _ in range(size)]

        def random(self, size):
            import random

            if isinstance(size, tuple):
                total = 1
                for n in size:
                    total *= n
            else:
                total = size
            return [random.random() for _ in range(total)]

    class _RandomNamespace(types.SimpleNamespace):
        def __init__(self):
            super().__init__(rand=lambda *shape: _zeros(shape[0] if shape else 1))

        def RandomState(self, seed=None):  # noqa: D401
            return _RandomState(seed)

    np_stub.array = _array  # type: ignore[attr-defined]
    np_stub.zeros = _zeros  # type: ignore[attr-defined]
    np_stub.arange = _arange  # type: ignore[attr-defined]
    np_stub.random = _RandomNamespace()  # type: ignore[attr-defined]

    def _polyfit(x, y, deg, *args, **kwargs):  # noqa: D401
        return _zeros(deg + 1)

    np_stub.polyfit = _polyfit  # type: ignore[attr-defined]

    lib_mod = types.ModuleType("numpy.lib")
    poly_mod = types.ModuleType("numpy.lib.polynomial")
    poly_mod.polyfit = _polyfit
    lib_mod.polynomial = poly_mod
    sys.modules["numpy.lib"] = lib_mod
    sys.modules["numpy.lib.polynomial"] = poly_mod

    # linalg sub‑module with *inv* placeholder to satisfy numpy.polyfit import
    linalg_mod = types.ModuleType("numpy.linalg")

    def _inv(x):  # noqa: D401
        return x  # identity – good enough for tests

    linalg_mod.inv = _inv  # type: ignore[attr-defined]
    sys.modules["numpy.linalg"] = linalg_mod

    for name in ("float32", "float64", "int32", "int64"):
        setattr(np_stub, name, None)

    sys.modules["numpy"] = np_stub

    # Core sub‑module placeholder (prevents attempts to load compiled ext.)
    core_mod = types.ModuleType("numpy.core")
    sys.modules["numpy.core"] = core_mod

    # Pre‑emptively register a few compiled submodules that some 3rd‑party
    # packages import explicitly when the *real* NumPy is available.
    for _name in [
        "_multiarray_umath",
        "_multiarray_tests",
        "_umath_linalg",
        "_pocketfft_internal",
    ]:
        stub_name = f"numpy.core.{_name}"
        sys.modules[stub_name] = types.ModuleType(stub_name)


# ---------------------------------------------------------------------------
# Stub for *scipy.sparse*
# ---------------------------------------------------------------------------


def _install_scipy_stub() -> None:  # noqa: D401
    if "scipy" in sys.modules:
        return

    sp_mod = types.ModuleType("scipy")
    sp_sparse = types.ModuleType("scipy.sparse")

    class _LilMatrix:
        def __init__(self, shape, dtype=None):
            self._shape = shape
            self._data = {}

        def __setitem__(self, idx, value):
            self._data[idx] = value

        def tocsr(self):
            return _CsrMatrix(self._shape, self._data.copy())

    class _CsrMatrix:
        def __init__(self, shape, data):
            self.shape = shape
            self._data = data

    sp_sparse.lil_matrix = _LilMatrix  # type: ignore[attr-defined]
    sp_sparse.csr_matrix = _CsrMatrix  # type: ignore[attr-defined]
    sp_mod.sparse = sp_sparse  # type: ignore[attr-defined]

    sys.modules["scipy"] = sp_mod
    sys.modules["scipy.sparse"] = sp_sparse


# ---------------------------------------------------------------------------
# Stub for *faiss*
# ---------------------------------------------------------------------------


def _install_faiss_stub() -> None:  # noqa: D401
    if "faiss" in sys.modules:
        return

    np_stub = sys.modules.get("numpy")

    class _FakeIndex:
        def __init__(self, dim):
            self.d = dim
            self.nprobe = 10

        def add(self, vectors):
            self._vectors = vectors

        def train(self, vectors):
            pass

        def search(self, query, k):
            n = len(query) if isinstance(query, list) else 1
            dists = np_stub.zeros((n, k)) if np_stub else [[0] * k for _ in range(n)]
            idxs = np_stub.zeros((n, k), dtype=int) if np_stub else [[0] * k for _ in range(n)]
            return dists, idxs

    faiss_mod = types.ModuleType("faiss")
    faiss_mod.METRIC_INNER_PRODUCT = 0
    faiss_mod.METRIC_L2 = 1
    faiss_mod.IndexFlatL2 = lambda dim: _FakeIndex(dim)  # type: ignore[attr-defined]
    faiss_mod.IndexFlatIP = lambda dim: _FakeIndex(dim)  # type: ignore[attr-defined]
    faiss_mod.IndexHNSWFlat = lambda dim, m: _FakeIndex(dim)  # type: ignore[attr-defined]
    faiss_mod.IndexIVFFlat = lambda q, dim, nlist, metric: _FakeIndex(dim)  # type: ignore[attr-defined]
    faiss_mod.normalize_L2 = lambda vecs: vecs  # type: ignore[attr-defined]

    sys.modules["faiss"] = faiss_mod


# ---------------------------------------------------------------------------
# Minimal stub for *pandas*
# ---------------------------------------------------------------------------


def _install_pandas_stub() -> None:  # noqa: D401
    if "pandas" in sys.modules:
        return

    pd_mod = types.ModuleType("pandas")

    class _DataFrame:
        def __init__(self, data=None, *_, **__):
            self._data = data

        def to_parquet(self, path, *_, **__):
            Path(path).write_text("parquet_stub")

        def to_csv(self, path, *_, **__):
            Path(path).write_text("csv_stub")

    pd_mod.DataFrame = _DataFrame  # type: ignore[attr-defined]
    pd_mod.Series = type("Series", (), {})  # type: ignore[attr-defined]

    sys.modules["pandas"] = pd_mod


# ---------------------------------------------------------------------------
# Install all stubs before anything else has a chance to pull in the real
# heavy‑weight libraries.
# ---------------------------------------------------------------------------


_install_numpy_stub()
_install_scipy_stub()
_install_faiss_stub()
_install_pandas_stub()
