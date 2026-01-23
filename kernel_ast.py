from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class EngineKind(Enum):
    ALU = "alu"
    VALU = "valu"
    LOAD = "load"
    STORE = "store"
    FLOW = "flow"
    DEBUG = "debug"


@dataclass(eq=False)
class ASTNode:
    engine: EngineKind
    op: str
    operands: tuple
    note: str = ""
    order: int = 0
    deps: set["ASTNode"] = field(default_factory=set)
    users: set["ASTNode"] = field(default_factory=set)

    def add_dep(self, other: "ASTNode") -> None:
        if other is self:
            return
        if other not in self.deps:
            self.deps.add(other)
            other.users.add(self)

    def __hash__(self) -> int:
        return id(self)


def scratch_reads(engine: EngineKind, op: str, operands: tuple) -> list[int]:
    if engine == EngineKind.ALU:
        _, _, a1, a2 = operands
        return [a1, a2]
    if engine == EngineKind.VALU:
        if op == "multiply_add":
            _, _, a, b, c = operands
            return [a, b, c]
        if op == "vbroadcast":
            _, _, src = operands
            return [src]
        if len(operands) >= 4:
            _, _, a, b = operands
            return [a, b]
        return []
    if engine == EngineKind.LOAD:
        if op == "load":
            _, _, addr = operands
            return [addr]
        if op == "vload":
            _, _, addr = operands
            return [addr]
        if op == "load_offset":
            _, _, addr, _ = operands
            return [addr]
        return []
    if engine == EngineKind.STORE:
        _, addr, src = operands
        return [addr, src]
    if engine == EngineKind.FLOW:
        if op == "select":
            _, _, cond, a, b = operands
            return [cond, a, b]
        if op == "add_imm":
            _, _, a, _ = operands
            return [a]
        if op == "vselect":
            _, _, cond, a, b = operands
            return [cond, a, b]
        if op in ("cond_jump", "cond_jump_rel"):
            _, cond, _ = operands
            return [cond]
        if op == "jump_indirect":
            _, addr = operands
            return [addr]
        return []
    if engine == EngineKind.DEBUG:
        if op == "compare":
            _, loc, _ = operands
            return [loc]
        if op == "vcompare":
            _, loc, _ = operands
            return [loc]
        return []
    return []


def scratch_writes(engine: EngineKind, op: str, operands: tuple) -> list[int]:
    if engine in (EngineKind.ALU, EngineKind.VALU):
        _, dest, *_ = operands
        return [dest]
    if engine == EngineKind.LOAD:
        if op in ("load", "const", "vload", "load_offset"):
            _, dest, *_ = operands
            return [dest]
        return []
    if engine == EngineKind.FLOW:
        if op in ("select", "add_imm", "vselect", "coreid"):
            _, dest, *_ = operands
            return [dest]
        return []
    return []


def iter_int_operands(values: Iterable) -> Iterable[int]:
    for v in values:
        if isinstance(v, int):
            yield v
