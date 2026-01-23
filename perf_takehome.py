"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from kernel_ast import ASTNode, EngineKind, scratch_reads, scratch_writes
from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class _InstrCollector:
    def __init__(self, owner):
        self._owner = owner

    def append(self, instr):
        bundle_id = len(self._owner._bundles)
        self._owner._bundles.append(instr)
        for engine, slots in instr.items():
            for slot in slots:
                self._owner.add_ast(engine, slot, bundle_id=bundle_id)

    def extend(self, instrs):
        for instr in instrs:
            self.append(instr)


class KernelBuilder:
    def __init__(self):
        self.instrs = _InstrCollector(self)
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        self.ast_nodes = []
        self._node_order = 0
        self._last_write = {}
        self._last_node = None
        self._bundles = []
        self.preserve_bundle_order = False

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.add_ast(engine, slot)

    def add_instr(self, instr):
        """Add a pre-built instruction dict directly."""
        for engine, slots in instr.items():
            for slot in slots:
                self.add_ast(engine, slot)

    def add_ast(self, engine, slot, note: str = "", bundle_id: int | None = None):
        engine_kind = EngineKind(engine) if not isinstance(engine, EngineKind) else engine
        op = slot[0]
        node = ASTNode(engine=engine_kind, op=op, operands=slot, note=note, order=self._node_order)
        node.bundle_id = bundle_id
        self._node_order += 1

        reads = scratch_reads(engine_kind, op, slot)
        writes = scratch_writes(engine_kind, op, slot)

        for addr in reads:
            if addr in self._last_write:
                node.add_dep(self._last_write[addr])
        for addr in writes:
            if addr in self._last_write:
                node.add_dep(self._last_write[addr])
            self._last_write[addr] = node

        # Preserve original program order until a scheduler is introduced.
        if self._last_node is not None:
            node.add_dep(self._last_node)
        self._last_node = node

        self.ast_nodes.append(node)
        return node

    def emit_from_ast(self):
        if self.preserve_bundle_order:
            # Emit bundles in a dependency-respecting order based on node deps.
            bundle_deps = {i: set() for i in range(len(self._bundles))}
            for node in self.ast_nodes:
                if node.bundle_id is None:
                    continue
                for dep in node.deps:
                    if dep.bundle_id is None:
                        continue
                    if dep.bundle_id != node.bundle_id:
                        bundle_deps[node.bundle_id].add(dep.bundle_id)

            indegree = {i: len(bundle_deps[i]) for i in bundle_deps}
            ready = [i for i, d in indegree.items() if d == 0]
            ready.sort()
            ordered = []
            while ready:
                i = ready.pop(0)
                ordered.append(i)
                for j in bundle_deps:
                    if i in bundle_deps[j]:
                        bundle_deps[j].remove(i)
                        indegree[j] -= 1
                        if indegree[j] == 0:
                            ready.append(j)
                            ready.sort()
            self.instrs = [self._bundles[i] for i in ordered]
            return

        indegree = {n: len(n.deps) for n in self.ast_nodes}
        ready = [n for n in self.ast_nodes if indegree[n] == 0]
        ready.sort(key=lambda n: n.order)
        instrs = []
        while ready:
            node = ready.pop(0)
            instrs.append({node.engine.value: [node.operands]})
            for user in list(node.users):
                indegree[user] -= 1
                if indegree[user] == 0:
                    ready.append(user)
                    ready.sort(key=lambda n: n.order)
        self.instrs = instrs

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.instrs.append({"load": [("const", addr, val)]})
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vec_const(self, val, name=None):
        """Allocate a vector constant by broadcasting a scalar."""
        if val not in self.vec_const_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_scratch(name, VLEN)
            self.instrs.append({"valu": [("vbroadcast", vec_addr, scalar_addr)]})
            self.vec_const_map[val] = vec_addr
        return self.vec_const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized vectorized kernel.
        Key optimizations:
        - VALU for 8-wide SIMD
        - Eliminate vselect using pure VALU arithmetic
        - Precompute all chunk addresses at init (they don't change)
        - Maximize instruction packing
        """
        instrs = self.instrs
        n_chunks = batch_size // VLEN

        # Allocate and load memory layout parameters
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Load header values efficiently
        for i in range(0, len(init_vars), 2):
            n = min(2, len(init_vars) - i)
            instrs.append({"load": [("const", self.scratch[init_vars[i + j]], i + j) for j in range(n)]})
            instrs.append({"load": [("load", self.scratch[init_vars[i + j]], self.scratch[init_vars[i + j]]) for j in range(n)]})

        # Pre-allocate constants
        mul_vals = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_vals.append((1 + (1 << val3)) % (2**32))
        const_vals = list(set([0, 1, 2] + mul_vals + [s[1] for s in HASH_STAGES] + [s[4] for s in HASH_STAGES]))
        const_addrs = {}
        for i in range(0, len(const_vals), 2):
            ops = []
            for j in range(min(2, len(const_vals) - i)):
                addr = self.alloc_scratch(f"c_{const_vals[i+j]}")
                const_addrs[const_vals[i + j]] = addr
                ops.append(("const", addr, const_vals[i + j]))
            instrs.append({"load": ops})
        self.const_map = const_addrs

        # Broadcast constants to vectors
        vec_const_addrs = {}
        for val in const_vals:
            vec_addr = self.alloc_scratch(f"vc_{val}", VLEN)
            vec_const_addrs[val] = vec_addr
            instrs.append({"valu": [("vbroadcast", vec_addr, const_addrs[val])]})

        vec_n_nodes = self.alloc_scratch("vec_n_nodes", VLEN)
        instrs.append({"valu": [("vbroadcast", vec_n_nodes, self.scratch["n_nodes"])]})

        vec_zero, vec_one, vec_two = vec_const_addrs[0], vec_const_addrs[1], vec_const_addrs[2]
        hash_stage_info = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_val = (1 + (1 << val3)) % (2**32)
                hash_stage_info.append(("madd", vec_const_addrs[mul_val], vec_const_addrs[val1]))
            else:
                hash_stage_info.append((op1, vec_const_addrs[val1], op2, op3, vec_const_addrs[val3]))

        # Precompute ALL chunk addresses at initialization (they never change!)
        # This eliminates add_imm from the hot loop
        chunk_addr_idx = [self.alloc_scratch(f"cai_{c}") for c in range(n_chunks)]
        chunk_addr_val = [self.alloc_scratch(f"cav_{c}") for c in range(n_chunks)]

        # Compute chunk addresses using add_imm (only once at init, not per round!)
        for c in range(n_chunks):
            offset = c * VLEN
            instrs.append({"flow": [("add_imm", chunk_addr_idx[c], self.scratch["inp_indices_p"], offset)]})
            instrs.append({"flow": [("add_imm", chunk_addr_val[c], self.scratch["inp_values_p"], offset)]})

        # Double buffering: allow overlap of hash and next-group loads
        # Set A processes while set B loads (and vice versa)
        GROUP_SIZE = 4  # Chunks per pipeline stage (tuned for even group count)
        sets = [[], []]  # Two sets of chunk data
        for s in range(2):
            for c in range(GROUP_SIZE):
                sets[s].append({
                    "vec_node": self.alloc_scratch(f"vn_{s}_{c}", VLEN),
                    "vec_t1": self.alloc_scratch(f"vt1_{s}_{c}", VLEN),
                    "vec_t2": self.alloc_scratch(f"vt2_{s}_{c}", VLEN),
                    "tree_addrs": self.alloc_scratch(f"ta_{s}_{c}", VLEN),
                })

        # Allocate persistent storage for all chunk indices/values in scratch
        base_idx = self.alloc_scratch("vec_idx_all", n_chunks * VLEN)
        base_val = self.alloc_scratch("vec_val_all", n_chunks * VLEN)
        idx_addrs = [base_idx + c * VLEN for c in range(n_chunks)]
        val_addrs = [base_val + c * VLEN for c in range(n_chunks)]

        # Load all indices and values into scratch once (avoid per-round vload/vstore)
        for c in range(0, n_chunks, 2):
            ops = [("vload", idx_addrs[c], chunk_addr_idx[c])]
            if c + 1 < n_chunks:
                ops.append(("vload", idx_addrs[c + 1], chunk_addr_idx[c + 1]))
            instrs.append({"load": ops})
        for c in range(0, n_chunks, 2):
            ops = [("vload", val_addrs[c], chunk_addr_val[c])]
            if c + 1 < n_chunks:
                ops.append(("vload", val_addrs[c + 1], chunk_addr_val[c + 1]))
            instrs.append({"load": ops})

        instrs.append({"flow": [("pause",)]})

        # Helper to build instruction bundles for a single engine
        def build_engine_instrs(engine, ops):
            if not ops:
                return []
            limit = SLOT_LIMITS[engine]
            return [{engine: ops[i:i + limit]} for i in range(0, len(ops), limit)]

        # Helper to merge two streams of single-engine bundles without reordering
        def merge_streams(primary, secondary):
            merged = []
            i = j = 0
            while i < len(primary) or j < len(secondary):
                if i >= len(primary):
                    merged.append(secondary[j])
                    j += 1
                    continue
                if j >= len(secondary):
                    merged.append(primary[i])
                    i += 1
                    continue
                p = primary[i]
                s = secondary[j]
                if set(p.keys()).isdisjoint(s.keys()):
                    merged.append({**p, **s})
                    i += 1
                    j += 1
                else:
                    merged.append(p)
                    i += 1
            return merged

        # Helper to compute hash and index update for a set
        def build_hash_and_index_instrs(data, count):
            instrs_out = []
            # XOR
            instrs_out += build_engine_instrs(
                "valu",
                [("^", data[c]["vec_val"], data[c]["vec_val"], data[c]["vec_node"]) for c in range(count)],
            )
            # Hash
            for info in hash_stage_info:
                if info[0] == "madd":
                    _, vmul, vadd = info
                    instrs_out += build_engine_instrs(
                        "valu",
                        [("multiply_add", data[c]["vec_val"], data[c]["vec_val"], vmul, vadd) for c in range(count)],
                    )
                else:
                    op1, vc1, op2, op3, vc2 = info
                    ops = []
                    for c in range(count):
                        ops.append((op1, data[c]["vec_t1"], data[c]["vec_val"], vc1))
                        ops.append((op3, data[c]["vec_t2"], data[c]["vec_val"], vc2))
                    instrs_out += build_engine_instrs("valu", ops)
                    instrs_out += build_engine_instrs(
                        "valu",
                        [(op2, data[c]["vec_val"], data[c]["vec_t1"], data[c]["vec_t2"]) for c in range(count)],
                    )
            # Index update
            instrs_out += build_engine_instrs(
                "valu",
                [("&", data[c]["vec_t1"], data[c]["vec_val"], vec_one) for c in range(count)],
            )
            instrs_out += build_engine_instrs(
                "valu",
                [("+", data[c]["vec_t1"], data[c]["vec_t1"], vec_one) for c in range(count)],
            )
            instrs_out += build_engine_instrs(
                "valu",
                [("multiply_add", data[c]["vec_idx"], data[c]["vec_idx"], vec_two, data[c]["vec_t1"]) for c in range(count)],
            )
            instrs_out += build_engine_instrs(
                "valu",
                [("<", data[c]["vec_t1"], data[c]["vec_idx"], vec_n_nodes) for c in range(count)],
            )
            instrs_out += build_engine_instrs(
                "valu",
                [("*", data[c]["vec_idx"], data[c]["vec_idx"], data[c]["vec_t1"]) for c in range(count)],
            )
            return instrs_out

        # Main loop with pipelining
        n_groups = (n_chunks + GROUP_SIZE - 1) // GROUP_SIZE
        start_set = 0

        # Prologue: Load first group into set 0 for round 0
        g_start = 0
        g_count = min(GROUP_SIZE, n_chunks - g_start)
        cur_set = sets[0]
        cur_data = []
        for c in range(g_count):
            cur_data.append({
                "vec_idx": idx_addrs[g_start + c],
                "vec_val": val_addrs[g_start + c],
                "vec_node": cur_set[c]["vec_node"],
                "vec_t1": cur_set[c]["vec_t1"],
                "vec_t2": cur_set[c]["vec_t2"],
                "tree_addrs": cur_set[c]["tree_addrs"],
            })

        # Compute tree addresses (ALU per-lane to free VALU for hash)
        alu_ops = []
        for c in range(g_count):
            for i in range(VLEN):
                alu_ops.append(("+", cur_data[c]["tree_addrs"] + i, self.scratch["forest_values_p"], cur_data[c]["vec_idx"] + i))
        instrs.extend(build_engine_instrs("alu", alu_ops))

        # Load tree values for group 0
        load_ops = []
        for c in range(g_count):
            for i in range(VLEN):
                load_ops.append(("load", cur_data[c]["vec_node"] + i, cur_data[c]["tree_addrs"] + i))
        for i in range(0, len(load_ops), SLOT_LIMITS["load"]):
            instrs.append({"load": load_ops[i:i + SLOT_LIMITS["load"]]})

        for round_idx in range(rounds):
            # Main pipeline loop
            for g in range(n_groups):
                g_start = g * GROUP_SIZE
                g_count = min(GROUP_SIZE, n_chunks - g_start)
                cur_set = sets[(start_set + g) % 2]
                next_set = sets[(start_set + g + 1) % 2]
                cur_data = []
                for c in range(g_count):
                    cur_data.append({
                        "vec_idx": idx_addrs[g_start + c],
                        "vec_val": val_addrs[g_start + c],
                        "vec_node": cur_set[c]["vec_node"],
                        "vec_t1": cur_set[c]["vec_t1"],
                        "vec_t2": cur_set[c]["vec_t2"],
                        "tree_addrs": cur_set[c]["tree_addrs"],
                    })

                # Build hash and index update for current group
                hash_instrs = build_hash_and_index_instrs(cur_data, g_count)
                if round_idx == rounds - 1:
                    # Last round: store results as soon as this group completes
                    store_instrs = []
                    for c in range(0, g_count, 2):
                        ops = [("vstore", chunk_addr_idx[g_start + c], cur_data[c]["vec_idx"])]
                        if c + 1 < g_count:
                            ops.append(("vstore", chunk_addr_idx[g_start + c + 1], cur_data[c + 1]["vec_idx"]))
                        store_instrs.append({"store": ops})
                    for c in range(0, g_count, 2):
                        ops = [("vstore", chunk_addr_val[g_start + c], cur_data[c]["vec_val"])]
                        if c + 1 < g_count:
                            ops.append(("vstore", chunk_addr_val[g_start + c + 1], cur_data[c + 1]["vec_val"]))
                        store_instrs.append({"store": ops})
                    hash_instrs.extend(store_instrs)

                # Build next-group prep (loads + address calc + tree loads)
                next_prep_instrs = []
                next_start = None
                next_count = 0
                if g + 1 < n_groups:
                    next_start = (g + 1) * GROUP_SIZE
                    next_count = min(GROUP_SIZE, n_chunks - next_start)
                elif round_idx + 1 < rounds:
                    # Preload group 0 for the next round to avoid a per-round prologue
                    next_start = 0
                    next_count = min(GROUP_SIZE, n_chunks)

                if next_start is not None:
                    next_data = []
                    for c in range(next_count):
                        next_data.append({
                            "vec_idx": idx_addrs[next_start + c],
                            "vec_val": val_addrs[next_start + c],
                            "vec_node": next_set[c]["vec_node"],
                            "vec_t1": next_set[c]["vec_t1"],
                            "vec_t2": next_set[c]["vec_t2"],
                            "tree_addrs": next_set[c]["tree_addrs"],
                        })

                    alu_ops = []
                    for c in range(next_count):
                        for i in range(VLEN):
                            alu_ops.append(("+", next_data[c]["tree_addrs"] + i, self.scratch["forest_values_p"], next_data[c]["vec_idx"] + i))
                    next_prep_instrs += build_engine_instrs("alu", alu_ops)

                    load_ops = []
                    for c in range(next_count):
                        for i in range(VLEN):
                            load_ops.append(("load", next_data[c]["vec_node"] + i, next_data[c]["tree_addrs"] + i))
                    next_prep_instrs += build_engine_instrs("load", load_ops)

                # Overlap hash with next-group prep where safe (different engines)
                instrs.extend(merge_streams(hash_instrs, next_prep_instrs))
            # Align next round's start_set with the last group's prefetch target
            start_set = (start_set + n_groups) % 2

        instrs.append({"flow": [("pause",)]})
        self.emit_from_ast()

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
