from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

from pkg.base import Base, EagerBase

if TYPE_CHECKING:
    from pkg.cycle_a import Cycled


class ReadBeforeAssign:
    """The shape of the scheduler crash: a step reads a field that a later step sets."""

    def __init__(self):
        self.init_throttle()
        self.init_mode()

    def init_throttle(self):
        self.throttle = self.mode == "null"

    def init_mode(self):
        self.mode = "null"


class Ordered:
    def __init__(self):
        self.init_mode()
        self.init_throttle()

    def init_throttle(self):
        self.throttle = self.mode == "null"

    def init_mode(self):
        self.mode = "null"


class LazyField:
    def __init__(self):
        self.ready = True

    def run(self):
        self.count = 1


class AsyncStep:
    def __init__(self):
        self.task = self.serve()
        self.later = 1

    async def serve(self):
        return self.later


class GeneratorStep:
    def __init__(self):
        self.items = self.walk()
        self.later = 1

    def walk(self):
        yield self.later


class AnnotationOnly:
    def __init__(self):
        self.value: int
        self.other = 1


class HasattrGuard:
    def __init__(self):
        if hasattr(self, "late") and self.late:
            pass
        self.late = 1


class GetterReads:
    def __init__(self):
        self.first = self.derived
        self.base_value = 1

    @property
    def derived(self):
        return self.base_value


class SetterWrites:
    def __init__(self):
        self.value = 1
        self.seen = self._value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v


class Mutual:
    def __init__(self):
        self.ping(3)
        self.done = True

    def ping(self, n):
        if n:
            self.pong(n - 1)

    def pong(self, n):
        self.ping(n)


class ComputedSetattr:
    def __init__(self, source):
        for name in ("a", "b"):
            setattr(self, name, getattr(source, name))
        self.c = self.a

    def reset(self):
        self.a = 0


class Derived(Base):
    """A base constructor calls an override that reads a field set after super().__init__()."""

    def __init__(self):
        super().__init__()
        self.lock = object()

    def reset(self):
        self.held = self.lock


class ExternalBase(EagerBase):
    def __init__(self):
        super().__init__()
        self.size = self.weight

    def load(self):
        self.weight = 2


class CooperativeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = self.base_mapper

    def rebuild(self):
        self.base_mapper = None


class Buffered(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", None)
        self.size = self.cache

    def reset(self):
        self.cache = None


class Middle(Base):
    def __init__(self):
        super().__init__()
        self.copy = self.later

    def load(self):
        self.later = 0


class SkipsMiddle(Middle):
    """super(Middle, self) runs Base.__init__, so Middle's read of `later` never runs."""

    def __init__(self):
        super(Middle, self).__init__()
        self.later = 1
        self.copy = None


class NestedSelf:
    def __init__(self):
        self.ready = True

    def build(self):
        def fset(self, value):
            self._value = value

        class Servicer:
            def __init__(self):
                self.collector = None

        return fset, Servicer


class DevBase:
    def __init__(self):
        self.base_ready = True


class ReadsDevice(DevBase):
    def __init__(self):
        super().__init__()
        self.where = self.device

    def move(self):
        self.device = "cuda"


class SetsDevice(DevBase):
    def __init__(self):
        self.device = "cpu"
        super().__init__()


class Diamond(ReadsDevice, SetsDevice):
    """C3 runs SetsDevice between ReadsDevice and DevBase; depth-first order would skip it."""


def _fill(store, size):
    store.size = size
    store.table = [0] * size


class HelperFills:
    def __init__(self):
        _fill(self, 4)
        self.total = self.size + len(self.table)

    def grow(self):
        self.size += 1


class SwapsClass:
    def __init__(self):
        self.ready = True

    def demote(self):
        self.__class__ = Ordered


class NoSplatMixin:
    def __init__(self):
        super().__init__()
        self.head = self.embed

    def load(self):
        self.embed = 1


class CommentedParams:
    def __init__(  # pylint: disable-all
        self,
        size,
    ):
        self.first = self.size

    def grow(self):
        self.size = 1


class Resettable:
    def __init__(self):
        self.ready = True

    def clear(self):
        self.free = []


class OverridesClear(Resettable):
    """Resettable.clear never runs here, so its fields are not this class's."""

    def clear(self):
        pass


class ResizableBase:
    def __init__(self):
        self.size = 0

    def resize(self):
        self.pages = 1


class SkipsBaseResize(ResizableBase):
    def resize(self):
        self.size = 2


class ChainsPastSkip(SkipsBaseResize):
    """super().resize() reaches SkipsBaseResize.resize, which never calls ResizableBase.resize."""

    def resize(self):
        super().resize()


class ExplicitOther:
    def __init__(self):
        Resettable.__init__(self)
        self.copy = self.ready


class AlwaysRaises:
    def __init__(self):
        self.build()
        self.after = self.late

    def build(self):
        raise NotImplementedError

    def load(self):
        self.late = 1


class NamedSetter:
    def __init__(self):
        self._set_field("_status", 0)
        self.copy = self._status

    def _set_field(self, name, value):
        if value == getattr(self, name, None):
            return
        setattr(self, name, value)

    def pause(self):
        self._paused = True
        self._status = 1


class RaisesOnlyUnderCondition:
    def __init__(self, fast):
        if fast:
            self.build()
        self.after = self.late

    def build(self):
        raise NotImplementedError

    def load(self):
        self.late = 1


@dataclass
class Record(Resettable):
    size: int
    name: str = ""

    def __post_init__(self):
        self.total = self.size


class UsesCycle:
    def __init__(self, other: "Cycled"):
        self.other = other
