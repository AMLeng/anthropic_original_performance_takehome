"""
AST-based instruction scheduler for VLIW architectures.

This module handles:
- Building a dependency graph (AST) from sequential instruction additions
- Scheduling instructions into VLIW bundles respecting dependencies and slot limits
"""

from collections import defaultdict

from kernel_ast import ASTNode, EngineKind, scratch_reads, scratch_writes
from problem import SLOT_LIMITS


class _InstrCollector:
    """Helper class that collects instructions and adds them to the AST."""

    def __init__(self, owner):
        self._owner = owner

    def append(self, instr, note: str = "", readonly: bool = False):
        for engine, slots in instr.items():
            for slot in slots:
                self._owner.add_ast(engine, slot, note=note, readonly=readonly)

    def extend(self, instrs, note: str = ""):
        for instr in instrs:
            self.append(instr, note=note)


class ASTScheduler:
    """
    Builds a dependency graph from instructions and schedules them into VLIW bundles.

    Tracks dependencies:
    - Read-after-write (RAW): read depends on prior write to same address
    - Write-after-read (WAR): write depends on prior read from same address
    - Write-after-write (WAW): write depends on prior write to same address
    - Memory ordering: loads/stores have additional ordering constraints
    - Barriers: pause instructions act as full barriers
    """

    def __init__(self):
        self.instrs = _InstrCollector(self)
        self.ast_nodes = []
        self._node_order = 0
        self._last_write = {}
        self._last_read = {}
        self._last_load = None
        self._last_store = None
        self._last_barrier = None

    def add_ast(self, engine, slot, note: str = "", readonly: bool = False):
        """Add an instruction to the AST with automatic dependency tracking."""
        engine_kind = EngineKind(engine) if not isinstance(engine, EngineKind) else engine
        op = slot[0]
        node = ASTNode(engine=engine_kind, op=op, operands=slot, note=note, order=self._node_order)
        self._node_order += 1

        reads = scratch_reads(engine_kind, op, slot)
        writes = scratch_writes(engine_kind, op, slot)

        # Track RAW and WAR dependencies per scratch address
        for addr in reads:
            if addr in self._last_write:
                node.add_dep(self._last_write[addr])
            self._last_read[addr] = node
        for addr in writes:
            if addr in self._last_write:
                node.add_dep(self._last_write[addr])
            if addr in self._last_read:
                node.add_dep(self._last_read[addr])
            self._last_write[addr] = node

        # Memory ordering for loads and stores
        if engine_kind == EngineKind.LOAD:
            # Prevent load/store reordering, but allow load/load reordering.
            # Readonly loads (e.g., tree loads) access memory that stores never touch,
            # so they don't need store dependencies and stores don't need to wait for them.
            if self._last_store is not None and not readonly:
                node.add_dep(self._last_store)
            if not readonly:
                self._last_load = node
        elif engine_kind == EngineKind.STORE:
            # Stores should not pass prior loads or stores.
            if self._last_load is not None:
                node.add_dep(self._last_load)
            if self._last_store is not None:
                node.add_dep(self._last_store)
            self._last_store = node

        # Barrier handling for pause instructions
        if engine_kind == EngineKind.FLOW and op == "pause":
            # Pause should occur after all prior scratch writes (barrier).
            for dep in set(self._last_write.values()):
                node.add_dep(dep)
            if self._last_load is not None:
                node.add_dep(self._last_load)
            if self._last_store is not None:
                node.add_dep(self._last_store)
            if self._last_barrier is not None:
                node.add_dep(self._last_barrier)
            self._last_barrier = node
        elif self._last_barrier is not None:
            # Everything after a pause must occur after it.
            node.add_dep(self._last_barrier)

        self.ast_nodes.append(node)
        return node

    def emit_from_ast(self):
        """Schedule AST nodes into VLIW instruction bundles."""
        # Compute dist_to_pause: longest path from each node to pause nodes
        # This replaces dist_to_sink to avoid distortion from store dependencies
        pause_nodes = [n for n in self.ast_nodes if n.op == "pause"]
        dist_to_pause = {n: 0 for n in self.ast_nodes}
        for pause in pause_nodes:
            dist_to_pause[pause] = 0
        # Propagate backwards using longest path (like dist_to_sink but only to pause)
        in_degree = {n: len(n.users) for n in self.ast_nodes}
        queue = [n for n in self.ast_nodes if in_degree[n] == 0]
        while queue:
            node = queue.pop(0)
            for dep in node.deps:
                new_dist = dist_to_pause[node] + 1
                if new_dist > dist_to_pause[dep]:
                    dist_to_pause[dep] = new_dist
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        # Compute distance to nearest LOAD user (prioritize ops that enable loads)
        dist_to_load = {n: 1000000 for n in self.ast_nodes}
        for n in self.ast_nodes:
            if n.engine.value == "load":
                dist_to_load[n] = 0
        # Propagate backwards from loads through users
        changed = True
        while changed:
            changed = False
            for n in self.ast_nodes:
                for user in n.users:
                    new_dist = dist_to_load[user] + 1
                    if new_dist < dist_to_load[n]:
                        dist_to_load[n] = new_dist
                        changed = True

        def sort_key(n):
            # Priority order:
            # 1. Memory ops first (limited slots - LOAD=0, STORE=1, other=2)
            # 2. dist_to_load (lower = closer to enabling a LOAD)
            # 3. dist_to_pause (higher = earlier in DAG = unblocks more work)
            # 4. Original order for stability
            if n.engine.value == "load":
                mem_priority = 0
            elif n.engine.value == "store":
                mem_priority = 1
            else:
                mem_priority = 2
            return (mem_priority, dist_to_load[n], -dist_to_pause[n], n.order)

        indegree = {n: len(n.deps) for n in self.ast_nodes}
        ready = [n for n in self.ast_nodes if indegree[n] == 0]
        ready.sort(key=sort_key)
        instrs = []
        while ready:
            slots_used = defaultdict(int)
            bundle = {}
            scheduled = []
            remaining = []

            # Two-pass scheduling to maximize LOAD/STORE slot utilization
            # Pass 1: Schedule LOADs, STOREs, and ops close to enabling LOADs
            for node in ready:
                eng = node.engine.value
                limit = SLOT_LIMITS.get(eng, 1)
                if slots_used[eng] < limit:
                    # Prioritize: memory ops, LOAD-enabling ops
                    if eng in ("load", "store") or dist_to_load[node] <= 1:
                        slots_used[eng] += 1
                        bundle.setdefault(eng, []).append(node.operands)
                        scheduled.append(node)
                    else:
                        remaining.append(node)
                else:
                    remaining.append(node)

            # Pass 2: Fill remaining slots with other operations
            for node in remaining:
                eng = node.engine.value
                limit = SLOT_LIMITS.get(eng, 1)
                if slots_used[eng] < limit:
                    slots_used[eng] += 1
                    bundle.setdefault(eng, []).append(node.operands)
                    scheduled.append(node)

            if not scheduled:
                node = ready.pop(0)
                bundle = {node.engine.value: [node.operands]}
                scheduled = [node]
                ready = ready[1:]
            else:
                scheduled_set = set(scheduled)
                ready = [n for n in ready if n not in scheduled_set]

            instrs.append(bundle)
            for node in scheduled:
                for user in list(node.users):
                    indegree[user] -= 1
                    if indegree[user] == 0:
                        ready.append(user)
            ready.sort(key=sort_key)
        self.instrs = instrs
