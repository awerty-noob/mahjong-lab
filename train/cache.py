"""Compilation caches and fixed-page scratch sizing."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from functools import wraps
from threading import RLock
from typing import NamedTuple

import jax

# --- from jit_cache.py ---


class CacheInfo(NamedTuple):
    size: int
    limit: int
    hits: int
    misses: int
    evictions: int


class JitLRUCache:
    """Bounded key -> owned callable cache, evicting BEFORE calling the factory.

    ``factory(key)`` should return a unique JIT callable owned by this cache. An
    evicted callable's ``clear_cache`` is invoked, then the reference is removed.
    There is deliberately no ``jax.clear_caches``. A factory failure leaves the
    remaining cache valid, without inserting a failed entry. Counters include
    failed misses and survive explicit ``clear_cache`` for operational telemetry.
    """

    def __init__(self, max_entries, factory):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.max_entries = max_entries
        self.factory = factory
        self._entries = OrderedDict()
        self._hits = self._misses = self._evictions = 0
        self._lock = RLock()

    @staticmethod
    def _release(value):
        clear = getattr(value, "clear_cache", None)
        if clear is not None:
            clear()

    def get(self, key):
        # Fail early on invalid/unhashable keys, without evicting valid entries.
        hash(key)
        with self._lock:
            if key in self._entries:
                self._hits += 1
                value = self._entries.pop(key)
                self._entries[key] = value
                return value
            self._misses += 1
            if len(self._entries) >= self.max_entries:
                _, old = self._entries.popitem(last=False)
                self._release(old)
                self._evictions += 1
                del old  # Do not retain the old executable during new compilation.
            value = self.factory(key)
            self._entries[key] = value
            return value

    def clear_cache(self):
        with self._lock:
            # Pop before releasing, avoiding references to every entry in a list.
            while self._entries:
                _, value = self._entries.popitem(last=False)
                self._release(value)
                del value

    def _cache_size(self):
        with self._lock:
            return len(self._entries)

    def cache_info(self):
        with self._lock:
            return CacheInfo(len(self._entries), self.max_entries, self._hits, self._misses, self._evictions)


def _abstract_tree(value):
    """Shape/dtype/weak-type only: no device_get/np.asarray or captured values."""
    leaves, tree = jax.tree_util.tree_flatten(value)
    abstract = []
    for leaf in leaves:
        # get_aval moved out of jax.core in current JAX; typeof is its public
        # replacement. Older supported JAX releases expose only core.get_aval.
        aval = jax.typeof(leaf) if hasattr(jax, "typeof") else jax.core.get_aval(leaf)
        if not hasattr(aval, "shape") or not hasattr(aval, "dtype"):
            raise TypeError("bounded_jit requires shaped array/scalar dynamic arguments")
        abstract.append((tuple(aval.shape), str(aval.dtype), bool(aval.weak_type)))
    return tree, tuple(abstract)


def bounded_jit(fun=None, *, max_entries=8, static_argnames=()):
    """Like a small subset of ``jax.jit``, with an explicit specialization LRU.

    Static parameters must be named, declared parameters and hashable, as for
    JAX's static args. Dynamic keys include the PyTree schema, shapes, dtypes and
    weak types, never array contents. Bound defaults/call styles are normalized.
    Each entry has a UNIQUE inner function identity: clearing an evicted entry
    cannot clear a surviving entry for another shape of the original function.

    ``clear_cache``, ``_cache_size``, ``cache_info`` and ``lower`` are exposed.
    Host caching also works when invoked while tracing an outer jit/grad; no
    tracer is captured in an entry closure. Limits do not cover the outer JIT.
    """
    if fun is None:
        return lambda fn: bounded_jit(fn, max_entries=max_entries, static_argnames=static_argnames)
    signature = inspect.signature(fun)
    names = (static_argnames,) if isinstance(static_argnames, str) else tuple(static_argnames)
    if len(set(names)) != len(names) or any(name not in signature.parameters for name in names):
        raise ValueError("static_argnames must name unique declared parameters")
    static_names = frozenset(names)

    def factory(_key):
        @wraps(fun)
        def owned(*args, **kwargs):
            return fun(*args, **kwargs)

        # Do not capture bound args/the cache key: they may contain live tracers.
        return jax.jit(owned, static_argnames=names)

    cache = JitLRUCache(max_entries, factory)

    def select(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = []
        for name, value in bound.arguments.items():
            if name in static_names:
                try:
                    hash(value)
                except TypeError as error:
                    raise TypeError(f"static argument {name!r} must be hashable") from error
                # Include type to distinguish False from 0 in host specializations.
                key.append((name, "static", type(value), value))
            else:
                key.append((name, "dynamic", _abstract_tree(value)))
        return cache.get(tuple(key)), bound.args, bound.kwargs

    @wraps(fun)
    def wrapped(*args, **kwargs):
        compiled, args, kwargs = select(args, kwargs)
        return compiled(*args, **kwargs)

    def lower(*args, **kwargs):
        compiled, args, kwargs = select(args, kwargs)
        return compiled.lower(*args, **kwargs)

    wrapped.clear_cache = cache.clear_cache
    wrapped._cache_size = cache._cache_size
    wrapped.cache_info = cache.cache_info
    wrapped.lower = lower
    return wrapped


# --- from vrpo_page_memory.py ---


def fixed_page_scratch_bytes(page_rows, raw_row_bytes, num_games, leaf_count):
    """Bound feature compaction copies, numerical arrays and per-game carries.

    The bound is independent of actual lengths, selected/owned row counts and
    page count. Three full raw copies conservatively dominate public-only Actor
    and combined public/private Critic compaction/gather scratch. Numeric reserve
    covers Q prediction, laws/log-prob/reference, all-seat labels and row indices.
    """
    for name, value in (
        ("page_rows", page_rows),
        ("raw_row_bytes", raw_row_bytes),
        ("num_games", num_games),
        ("leaf_count", leaf_count),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    feature_copy = page_rows * raw_row_bytes + leaf_count * 256 + 1024
    numeric_per_row = 3 * 87 * 4 + 4 * 87 * 4 + 4 * 3 * 4 + 128
    # Stable row grouping/permutation plus per-game parallel reverse recurrence.
    # Reserve for promoted float64 as well as the production float32 trace:
    # 8 int32[P] arrays and (128 + 48 * itemsize)[G] bytes. Backend sorting /
    # compiler workspace is explicitly outside this logical live-array ledger.
    parallel_trace = 32 * page_rows + 512 * num_games
    return 3 * feature_copy + page_rows * numeric_per_row + num_games * 64 + parallel_trace + 8192
