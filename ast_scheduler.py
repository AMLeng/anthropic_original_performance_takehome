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
    """

    def __init__(self):
        self.instrs = _InstrCollector(self)
        self.ast_nodes = []
        self._node_order = 0
        self._last_write = {}
        self._last_read = {}
        self._last_load = None
        self._last_store = None

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
            self._last_load = node
        elif engine_kind == EngineKind.STORE:
            # Stores should not pass prior loads or stores.
            if self._last_load is not None:
                node.add_dep(self._last_load)
            if self._last_store is not None:
                node.add_dep(self._last_store)
            self._last_store = node

        # Note: pause instructions are no-ops at runtime (enable_pause=False in tests)
        # and the data dependencies (scratch reads/writes) are sufficient for correctness.
        # We no longer treat pause as a barrier since it creates unnecessary serialization.

        self.ast_nodes.append(node)
        return node

    def emit_from_ast(self):
        """Schedule AST nodes into VLIW instruction bundles."""
        # Compute dist_to_store: longest path from each node to store nodes
        # Higher values = earlier in DAG = unblocks more work toward output
        store_nodes = [n for n in self.ast_nodes if n.engine.value == "store"]
        dist_to_store = {n: 0 for n in self.ast_nodes}
        for store in store_nodes:
            dist_to_store[store] = 0
        # Propagate backwards from stores
        in_degree = {n: len(n.users) for n in self.ast_nodes}
        queue = [n for n in self.ast_nodes if in_degree[n] == 0]
        while queue:
            node = queue.pop(0)
            for dep in node.deps:
                new_dist = dist_to_store[node] + 1
                if new_dist > dist_to_store[dep]:
                    dist_to_store[dep] = new_dist
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
            # 1. dist_to_load (lower = closer to enabling a LOAD)
            # 2. dist_to_store (higher = earlier in DAG = unblocks more work toward output)
            # 3. Original order for stability
            return (dist_to_load[n], -dist_to_store[n], n.order)

        indegree = {n: len(n.deps) for n in self.ast_nodes}
        ready = [n for n in self.ast_nodes if indegree[n] == 0]
        ready.sort(key=sort_key)
        instrs = []
        while ready:
            slots_used = defaultdict(int)
            bundle = {}
            scheduled = []

            # Single-pass scheduling: fill slots up to limits
            for node in ready:
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
