"""Pure bounded-cache primitives and payload-size helpers.

This module deliberately has no RAPTOR or server state.  The server re-exports these exact
objects so existing callers and tests retain their class/function identity and monkeypatch
surface while cache implementation details stay isolated from routing logic.
"""
import copy
import sys
import threading
from collections import OrderedDict


class BoundedLRU:
    """Thread-safe bounded LRU with optional copy and caller-supplied byte policies."""

    def __init__(self, maxsize, copy_mode=None, lock=None, *, maxbytes=None, weight_fn=None):
        self.maxsize = int(maxsize)
        self._copy_mode = copy_mode
        self.maxbytes = None if maxbytes is None else max(0, int(maxbytes))
        self._weight_fn = weight_fn
        self._od = OrderedDict()
        self._weights = {}
        self._bytes = 0
        self.lock = lock if lock is not None else threading.RLock()

    def _out(self, value):
        if self._copy_mode == "deep":
            return copy.deepcopy(value)
        if self._copy_mode == "shallow":
            return dict(value)
        return value

    def get(self, key, default=None):
        with self.lock:
            if key in self._od:
                self._od.move_to_end(key)
                return self._out(self._od[key])
        return default

    def put(self, key, value):
        with self.lock:
            stored = copy.deepcopy(value) if self._copy_mode == "deep" else value
            weight = (max(0, int(self._weight_fn(stored)))
                      if self._weight_fn is not None else 0)
            if key in self._od:
                self._bytes -= self._weights.pop(key, 0)
            self._od[key] = stored
            self._weights[key] = weight
            self._bytes += weight
            self._od.move_to_end(key)
            while (len(self._od) > self.maxsize
                   or (self.maxbytes is not None and self._bytes > self.maxbytes)):
                old_key, _old_value = self._od.popitem(last=False)
                self._bytes -= self._weights.pop(old_key, 0)

    def pop(self, key, default=None):
        with self.lock:
            if key not in self._od:
                return default
            self._bytes -= self._weights.pop(key, 0)
            return self._od.pop(key)

    def clear(self):
        with self.lock:
            self._od.clear()
            self._weights.clear()
            self._bytes = 0

    def __contains__(self, key):
        with self.lock:
            return key in self._od

    def __len__(self):
        with self.lock:
            return len(self._od)

    @property
    def nbytes(self):
        """Current caller-supplied payload weight (bookkeeping objects excluded)."""
        with self.lock:
            return self._bytes


class BoundedCellCache(dict):
    """Insertion-ordered bounded mapping for lazy per-cell response fragments."""

    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = max(1, int(maxsize))

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.maxsize:
            del self[next(iter(self))]
        super().__setitem__(key, value)


def _owned_payload_nbytes(root, *, borrowed_root_ids=frozenset()):
    """Conservatively estimate bytes retained below an owned cache root.

    This is shadow telemetry rather than an eviction weight.  Identity de-duplication keeps
    shared aliases from being charged twice, while explicit borrowed roots exclude process-static
    objects owned outside the cache.  NumPy arrays/views, JSON-like containers, ordinary
    ``__dict__`` objects, and slotted payload objects are handled without importing NumPy.
    """
    borrowed = frozenset(int(ident) for ident in borrowed_root_ids)
    seen = set()

    def visit(item):
        ident = id(item)
        if ident in borrowed or ident in seen:
            return 0
        seen.add(ident)
        try:
            shallow = max(0, int(sys.getsizeof(item)))
        except (TypeError, ValueError, OverflowError):
            shallow = 0

        module = getattr(item.__class__, "__module__", "")
        nbytes = None
        if module.startswith("numpy"):
            try:
                nbytes = getattr(item, "nbytes", None)
            except (AttributeError, TypeError, ValueError):
                nbytes = None
        if nbytes is not None:
            try:
                payload = max(0, int(nbytes))
            except (TypeError, ValueError, OverflowError):
                payload = 0
            base = getattr(item, "base", None)
            if base is not None:
                return shallow + visit(base)
            return max(shallow, payload + 128)

        if isinstance(item, memoryview):
            return shallow + max(0, int(getattr(item, "nbytes", 0)))
        if isinstance(item, dict):
            try:
                pairs = list(item.items())
            except RuntimeError:
                return shallow
            return shallow + sum(visit(key) + visit(value) for key, value in pairs)
        if isinstance(item, (list, tuple, set, frozenset)):
            try:
                values = list(item)
            except RuntimeError:
                return shallow
            return shallow + sum(visit(value) for value in values)
        if isinstance(item, (str, bytes, bytearray, bool, int, float, complex, type(None))):
            return shallow

        try:
            attrs = getattr(item, "__dict__", None)
        except (AttributeError, TypeError, ValueError):
            attrs = None
        if attrs is not None:
            return shallow + visit(attrs)

        total = shallow
        for cls in getattr(item.__class__, "__mro__", ()):
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in ("__dict__", "__weakref__"):
                    continue
                try:
                    value = getattr(item, name)
                except (AttributeError, TypeError):
                    continue
                total += visit(value)
        return total

    return max(0, int(visit(root)))


def _array_tuple_weight(value):
    """Payload bytes for tuples/lists/dicts made primarily of numpy arrays."""
    seen = set()

    def visit(item):
        ident = id(item)
        if ident in seen:
            return 0
        seen.add(ident)
        nbytes = getattr(item, "nbytes", None)
        if nbytes is not None and item.__class__.__module__.startswith("numpy"):
            return int(nbytes) + 128
        if isinstance(item, dict):
            return 64 + sum(visit(k) + visit(v) for k, v in item.items())
        if isinstance(item, (list, tuple)):
            return 64 + sum(visit(v) for v in item)
        return 64

    return visit(value)


def _walkpath_tree_weight(tree):
    """Owned predecessor/distance payload only; the referenced WalkGraph is process-global."""
    return 1024 + sum(int(getattr(getattr(tree, name, None), "nbytes", 0))
                      for name in ("_snodes", "_dist", "_pred"))
