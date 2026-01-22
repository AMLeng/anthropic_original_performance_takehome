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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_instr(self, instr):
        """Add a pre-built instruction dict directly."""
        self.instrs.append(instr)

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
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vec_const(self, val, name=None):
        """Allocate a vector constant by broadcasting a scalar."""
        if val not in self.vec_const_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_scratch(name, VLEN)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
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
        const_vals = list(set([0, 1, 2] + [s[1] for s in HASH_STAGES] + [s[4] for s in HASH_STAGES]))
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

        vec_forest_values_p = self.alloc_scratch("vec_forest_values_p", VLEN)
        instrs.append({"valu": [("vbroadcast", vec_forest_values_p, self.scratch["forest_values_p"])]})

        vec_zero, vec_one, vec_two = vec_const_addrs[0], vec_const_addrs[1], vec_const_addrs[2]
        vec_hash_consts = [(vec_const_addrs[s[1]], vec_const_addrs[s[4]]) for s in HASH_STAGES]

        # Precompute ALL chunk addresses at initialization (they never change!)
        # This eliminates add_imm from the hot loop
        chunk_addr_idx = [self.alloc_scratch(f"cai_{c}") for c in range(n_chunks)]
        chunk_addr_val = [self.alloc_scratch(f"cav_{c}") for c in range(n_chunks)]

        # Compute chunk addresses using add_imm (only once at init, not per round!)
        for c in range(n_chunks):
            offset = c * VLEN
            instrs.append({"flow": [("add_imm", chunk_addr_idx[c], self.scratch["inp_indices_p"], offset)]})
            instrs.append({"flow": [("add_imm", chunk_addr_val[c], self.scratch["inp_values_p"], offset)]})

        # Double buffering: two sets of scratch for pipelining
        # Set A processes while set B loads (and vice versa)
        GROUP_SIZE = 8  # Chunks per pipeline stage
        sets = [[], []]  # Two sets of chunk data
        for s in range(2):
            for c in range(GROUP_SIZE):
                sets[s].append({
                    "vec_idx": self.alloc_scratch(f"vi_{s}_{c}", VLEN),
                    "vec_val": self.alloc_scratch(f"vv_{s}_{c}", VLEN),
                    "vec_node": self.alloc_scratch(f"vn_{s}_{c}", VLEN),
                    "vec_t1": self.alloc_scratch(f"vt1_{s}_{c}", VLEN),
                    "vec_t2": self.alloc_scratch(f"vt2_{s}_{c}", VLEN),
                    "tree_addrs": self.alloc_scratch(f"ta_{s}_{c}", VLEN),
                })

        instrs.append({"flow": [("pause",)]})

        # Helper to emit valu ops
        def emit_valu(ops):
            for i in range(0, len(ops), SLOT_LIMITS["valu"]):
                instrs.append({"valu": ops[i:i + SLOT_LIMITS["valu"]]})

        # Helper to compute hash and index update for a set
        def emit_hash_and_index(data, count):
            # XOR
            emit_valu([("^", data[c]["vec_val"], data[c]["vec_val"], data[c]["vec_node"]) for c in range(count)])
            # Hash
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc2 = vec_hash_consts[hi]
                ops = []
                for c in range(count):
                    ops.append((op1, data[c]["vec_t1"], data[c]["vec_val"], vc1))
                    ops.append((op3, data[c]["vec_t2"], data[c]["vec_val"], vc2))
                emit_valu(ops)
                emit_valu([(op2, data[c]["vec_val"], data[c]["vec_t1"], data[c]["vec_t2"]) for c in range(count)])
            # Index update
            emit_valu([("&", data[c]["vec_t1"], data[c]["vec_val"], vec_one) for c in range(count)])
            emit_valu([("+", data[c]["vec_t1"], data[c]["vec_t1"], vec_one) for c in range(count)])
            emit_valu([("*", data[c]["vec_idx"], data[c]["vec_idx"], vec_two) for c in range(count)])
            emit_valu([("+", data[c]["vec_idx"], data[c]["vec_idx"], data[c]["vec_t1"]) for c in range(count)])
            emit_valu([("<", data[c]["vec_t1"], data[c]["vec_idx"], vec_n_nodes) for c in range(count)])
            emit_valu([("-", data[c]["vec_t1"], vec_zero, data[c]["vec_t1"]) for c in range(count)])
            emit_valu([("&", data[c]["vec_idx"], data[c]["vec_idx"], data[c]["vec_t1"]) for c in range(count)])

        # Main loop with pipelining
        n_groups = (n_chunks + GROUP_SIZE - 1) // GROUP_SIZE

        for round_idx in range(rounds):
            # Prologue: Load first group into set 0
            g = 0
            g_start = g * GROUP_SIZE
            g_count = min(GROUP_SIZE, n_chunks - g_start)
            cur_set = sets[0]

            # vload indices and values
            for c in range(0, g_count, 2):
                ops = [("vload", cur_set[c]["vec_idx"], chunk_addr_idx[g_start + c])]
                if c + 1 < g_count:
                    ops.append(("vload", cur_set[c+1]["vec_idx"], chunk_addr_idx[g_start + c + 1]))
                instrs.append({"load": ops})
            for c in range(0, g_count, 2):
                ops = [("vload", cur_set[c]["vec_val"], chunk_addr_val[g_start + c])]
                if c + 1 < g_count:
                    ops.append(("vload", cur_set[c+1]["vec_val"], chunk_addr_val[g_start + c + 1]))
                instrs.append({"load": ops})

            # Compute tree addresses
            emit_valu([("+", cur_set[c]["tree_addrs"], vec_forest_values_p, cur_set[c]["vec_idx"]) for c in range(g_count)])

            # Load tree values for group 0
            load_ops = []
            for c in range(g_count):
                for i in range(VLEN):
                    load_ops.append(("load", cur_set[c]["vec_node"] + i, cur_set[c]["tree_addrs"] + i))
            for i in range(0, len(load_ops), SLOT_LIMITS["load"]):
                instrs.append({"load": load_ops[i:i + SLOT_LIMITS["load"]]})

            # Main pipeline loop
            for g in range(n_groups):
                g_start = g * GROUP_SIZE
                g_count = min(GROUP_SIZE, n_chunks - g_start)
                cur_set = sets[g % 2]
                next_set = sets[(g + 1) % 2]

                # Compute hash and index update for current group
                emit_hash_and_index(cur_set, g_count)

                # Store results for current group (overlaps with next group's load prep)
                for c in range(0, g_count, 2):
                    ops = [("vstore", chunk_addr_idx[g_start + c], cur_set[c]["vec_idx"])]
                    if c + 1 < g_count:
                        ops.append(("vstore", chunk_addr_idx[g_start + c + 1], cur_set[c+1]["vec_idx"]))
                    instrs.append({"store": ops})
                for c in range(0, g_count, 2):
                    ops = [("vstore", chunk_addr_val[g_start + c], cur_set[c]["vec_val"])]
                    if c + 1 < g_count:
                        ops.append(("vstore", chunk_addr_val[g_start + c + 1], cur_set[c+1]["vec_val"]))
                    instrs.append({"store": ops})

                # Load next group (if any)
                if g + 1 < n_groups:
                    next_g = g + 1
                    next_start = next_g * GROUP_SIZE
                    next_count = min(GROUP_SIZE, n_chunks - next_start)

                    # vload indices and values
                    for c in range(0, next_count, 2):
                        ops = [("vload", next_set[c]["vec_idx"], chunk_addr_idx[next_start + c])]
                        if c + 1 < next_count:
                            ops.append(("vload", next_set[c+1]["vec_idx"], chunk_addr_idx[next_start + c + 1]))
                        instrs.append({"load": ops})
                    for c in range(0, next_count, 2):
                        ops = [("vload", next_set[c]["vec_val"], chunk_addr_val[next_start + c])]
                        if c + 1 < next_count:
                            ops.append(("vload", next_set[c+1]["vec_val"], chunk_addr_val[next_start + c + 1]))
                        instrs.append({"load": ops})

                    # Compute tree addresses
                    emit_valu([("+", next_set[c]["tree_addrs"], vec_forest_values_p, next_set[c]["vec_idx"]) for c in range(next_count)])

                    # Load tree values
                    load_ops = []
                    for c in range(next_count):
                        for i in range(VLEN):
                            load_ops.append(("load", next_set[c]["vec_node"] + i, next_set[c]["tree_addrs"] + i))
                    for i in range(0, len(load_ops), SLOT_LIMITS["load"]):
                        instrs.append({"load": load_ops[i:i + SLOT_LIMITS["load"]]})

        instrs.append({"flow": [("pause",)]})

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
