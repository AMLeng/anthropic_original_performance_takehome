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
        self._readers_since_write = defaultdict(set)  # Track ALL readers since last write
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
            self._readers_since_write[addr].add(node)
        for addr in writes:
            if addr in self._last_write:
                node.add_dep(self._last_write[addr])
            # WAR: depend on ALL readers since the last write
            for reader in self._readers_since_write[addr]:
                node.add_dep(reader)
            self._readers_since_write[addr].clear()
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
            # Stores must wait for prior loads to avoid reading stale data.
            # But stores to different addresses can run in parallel (no WAW conflicts).
            if self._last_load is not None:
                node.add_dep(self._last_load)
            # Note: we don't track _last_store dependencies because all stores
            # in this kernel write to different memory addresses.

        # Note: pause instructions are no-ops at runtime (enable_pause=False in tests)
        # and the data dependencies (scratch reads/writes) are sufficient for correctness.
        # We no longer treat pause as a barrier since it creates unnecessary serialization.

        self.ast_nodes.append(node)
        return node

    def _schedule_with_chain_increment(self, store_nodes_sorted, chain_increment):
        """Schedule AST nodes with a given virtual chain increment."""
        num_stores = len(store_nodes_sorted)

        # Compute dist_to_store with virtual chain for priority
        dist_to_store = {n: 0 for n in self.ast_nodes}
        for i, store in enumerate(reversed(store_nodes_sorted)):
            dist_to_store[store] = i * chain_increment

        # Build virtual user counts: actual users + virtual chain for stores
        in_degree = {n: len(n.users) for n in self.ast_nodes}
        for i in range(num_stores - 1):
            in_degree[store_nodes_sorted[i]] += 1

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
            if node.engine.value == "store":
                idx = store_nodes_sorted.index(node)
                if idx > 0:
                    prev_store = store_nodes_sorted[idx - 1]
                    in_degree[prev_store] -= 1
                    if in_degree[prev_store] == 0:
                        queue.append(prev_store)

        # Compute distance to nearest LOAD user
        dist_to_load = {n: 1000000 for n in self.ast_nodes}
        for n in self.ast_nodes:
            if n.engine.value == "load":
                dist_to_load[n] = 0
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
            return (dist_to_load[n], -dist_to_store[n], n.order)

        # Schedule
        indegree = {n: len(n.deps) for n in self.ast_nodes}
        ready = [n for n in self.ast_nodes if indegree[n] == 0]
        ready.sort(key=sort_key)
        instrs = []
        while ready:
            slots_used = defaultdict(int)
            bundle = {}
            scheduled = []
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
        return instrs

    def emit_from_ast(self):
        """Schedule AST nodes into VLIW instruction bundles."""
        store_nodes = [n for n in self.ast_nodes if n.engine.value == "store"]
        store_nodes_sorted = sorted(store_nodes, key=lambda n: n.order)

        # Try different chain increments and pick the best
        best_instrs = None
        best_cycles = float('inf')
        for chain_increment in [1, 2, 3, 4, 5, 6, 7, 8]:
            instrs = self._schedule_with_chain_increment(store_nodes_sorted, chain_increment)
            if len(instrs) < best_cycles:
                best_cycles = len(instrs)
                best_instrs = instrs

        self.instrs = best_instrs
