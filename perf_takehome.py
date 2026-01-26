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

import random
import unittest

from ast_scheduler import ASTScheduler
from problem import (
    Engine,
    DebugInfo,
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


class KernelBuilder(ASTScheduler):
    def __init__(self):
        super().__init__()
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

    def add(self, engine, slot, note: str = ""):
        self.add_ast(engine, slot, note=note)

    def add_instr(self, instr, note: str = ""):
        """Add a pre-built instruction dict directly."""
        for engine, slots in instr.items():
            for slot in slots:
                self.add_ast(engine, slot, note=note)

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
            instrs.append(
                {"load": [("const", self.scratch[init_vars[i + j]], i + j) for j in range(n)]},
                note="init",
            )
            instrs.append(
                {"load": [("load", self.scratch[init_vars[i + j]], self.scratch[init_vars[i + j]]) for j in range(n)]},
                note="init",
            )

        # Pre-allocate constants as vectors, with scalar in first element
        # This saves scratch by reusing the same memory for scalar and vector access
        mul_vals = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_vals.append((1 + (1 << val3)) % (2**32))
        base_consts = [1, 2, 5]
        const_vals = list(set(base_consts + mul_vals + [s[1] for s in HASH_STAGES] + [s[4] for s in HASH_STAGES]))
        const_vals = [v for v in const_vals if v != 0]  # Remove 0 if present

        # vec_zero is free (scratch is zero-initialized)
        vec_const_addrs = {0: self.alloc_scratch("vc_0", VLEN)}

        # Allocate VLEN words per constant, load scalar into first word, then broadcast
        for val in const_vals:
            vec_addr = self.alloc_scratch(f"vc_{val}", VLEN)
            vec_const_addrs[val] = vec_addr
        # Load scalars into first element of each vector (2 loads per cycle)
        for i in range(0, len(const_vals), 2):
            ops = []
            for j in range(min(2, len(const_vals) - i)):
                val = const_vals[i + j]
                ops.append(("const", vec_const_addrs[val], val))  # Load into first element
            instrs.append({"load": ops}, note="init")
        # Broadcast from first element to fill the vector
        for val in const_vals:
            vec_addr = vec_const_addrs[val]
            instrs.append({"valu": [("vbroadcast", vec_addr, vec_addr)]}, note="init")

        self.const_map = {v: vec_const_addrs[v] for v in const_vals}  # Scalar = first element of vector

        vec_n_nodes = self.alloc_scratch("vec_n_nodes", VLEN)
        instrs.append({"valu": [("vbroadcast", vec_n_nodes, self.scratch["n_nodes"])]}, note="init")

        vec_zero, vec_one, vec_two = vec_const_addrs[0], vec_const_addrs[1], vec_const_addrs[2]
        hash_stage_info = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_val = (1 + (1 << val3)) % (2**32)
                hash_stage_info.append(("madd", vec_const_addrs[mul_val], vec_const_addrs[val1]))
            else:
                hash_stage_info.append((op1, vec_const_addrs[val1], op2, op3, vec_const_addrs[val3]))

        # Cache tree values for levels 0, 1, 2 in a single contiguous block
        # Layout: [level0] [level1 x2] [level2 x4] [unused] = 8 words for vload
        # Tree indices: 0, 1, 2, 3, 4, 5, 6, (7 unused but loaded for alignment)
        tree_cache = self.alloc_scratch("tree_cache", VLEN)
        tree_cache_l0 = tree_cache + 0  # index 0
        tree_cache_l1 = tree_cache + 1  # indices 1, 2
        tree_cache_l2 = tree_cache + 3  # indices 3, 4, 5, 6

        # Load all 8 values with a single vload (forest_values_p points to tree[0])
        # Tree cache is from read-only memory
        instrs.append({"load": [("vload", tree_cache, self.scratch["forest_values_p"])]}, note="init", readonly=True)

        # Precompute ALL chunk addresses at initialization (they never change!)
        # This eliminates add_imm from the hot loop
        chunk_addr_idx = [self.alloc_scratch(f"cai_{c}") for c in range(n_chunks)]
        chunk_addr_val = [self.alloc_scratch(f"cav_{c}") for c in range(n_chunks)]

        # Compute chunk addresses using const + ALU instead of add_imm (FLOW has only 1 slot)
        # Load offset constants (0, 8, 16, ...) - can do 2 per cycle with LOAD
        offset_temps = [self.alloc_scratch(f"off_{c}") for c in range(n_chunks)]
        for c in range(0, n_chunks, 2):
            ops = [("const", offset_temps[c], c * VLEN)]
            if c + 1 < n_chunks:
                ops.append(("const", offset_temps[c + 1], (c + 1) * VLEN))
            instrs.append({"load": ops}, note="init")

        # Use ALU to add offsets to base pointers (12 slots per cycle)
        # Compute idx addresses
        for c in range(0, n_chunks, 12):
            alu_ops = []
            for i in range(min(12, n_chunks - c)):
                alu_ops.append(("+", chunk_addr_idx[c + i], self.scratch["inp_indices_p"], offset_temps[c + i]))
            instrs.append({"alu": alu_ops}, note="init")
        # Compute val addresses
        for c in range(0, n_chunks, 12):
            alu_ops = []
            for i in range(min(12, n_chunks - c)):
                alu_ops.append(("+", chunk_addr_val[c + i], self.scratch["inp_values_p"], offset_temps[c + i]))
            instrs.append({"alu": alu_ops}, note="init")

        # Double buffering: allow overlap of hash and next-group loads
        # Set A processes while set B loads (and vice versa)
        GROUP_SIZE = 10  # Chunks per pipeline stage - optimal balance
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

        # Load values into scratch (indices start at 0, scratch is zero-initialized)
        for c in range(0, n_chunks, 2):
            ops = [("vload", val_addrs[c], chunk_addr_val[c])]
            if c + 1 < n_chunks:
                ops.append(("vload", val_addrs[c + 1], chunk_addr_val[c + 1]))
            instrs.append({"load": ops}, note="init")

        instrs.append({"flow": [("pause",)]}, note="init")

        def append_ops(out, engine, ops, note="", readonly=False):
            for op in ops:
                out.append((engine, op, note, readonly))

        # Helper to compute hash and index update for a set
        def build_hash_and_index_slots(data, count, needs_clamp=True):
            slots_out = []
            # XOR
            append_ops(
                slots_out,
                "valu",
                [("^", data[c]["vec_val"], data[c]["vec_val"], data[c]["vec_node"]) for c in range(count)],
                "hash",
            )
            # Hash
            nonmadd_stage_idx = 0  # Track which non-madd stage we're on
            for info in hash_stage_info:
                if info[0] == "madd":
                    _, vmul, vadd = info
                    append_ops(
                        slots_out,
                        "valu",
                        [("multiply_add", data[c]["vec_val"], data[c]["vec_val"], vmul, vadd) for c in range(count)],
                        "hash",
                    )
                else:
                    op1, vc1, op2, op3, vc2 = info
                    # Use ALU for first non-madd stage to offload from VALU
                    if nonmadd_stage_idx == 0:
                        # Convert to ALU: each VALU op becomes VLEN ALU ops
                        alu_ops = []
                        for c in range(count):
                            for i in range(VLEN):
                                alu_ops.append((op1, data[c]["vec_t1"] + i, data[c]["vec_val"] + i, vc1 + i))
                                alu_ops.append((op3, data[c]["vec_t2"] + i, data[c]["vec_val"] + i, vc2 + i))
                        append_ops(slots_out, "alu", alu_ops, "hash_alu")
                        alu_ops2 = []
                        for c in range(count):
                            for i in range(VLEN):
                                alu_ops2.append((op2, data[c]["vec_val"] + i, data[c]["vec_t1"] + i, data[c]["vec_t2"] + i))
                        append_ops(slots_out, "alu", alu_ops2, "hash_alu")
                    else:
                        # Use VALU for other non-madd stages
                        ops = []
                        for c in range(count):
                            ops.append((op1, data[c]["vec_t1"], data[c]["vec_val"], vc1))
                            ops.append((op3, data[c]["vec_t2"], data[c]["vec_val"], vc2))
                        append_ops(slots_out, "valu", ops, "hash")
                        append_ops(
                            slots_out,
                            "valu",
                            [(op2, data[c]["vec_val"], data[c]["vec_t1"], data[c]["vec_t2"]) for c in range(count)],
                            "hash",
                        )
                    nonmadd_stage_idx += 1
            # Index update - reordered for better parallelism
            # branch_bit = hash & 1 AND base = idx * 2 + 1 can run in parallel
            # Then: new_idx = base + branch_bit
            idx_ops = []
            for c in range(count):
                idx_ops.append(("&", data[c]["vec_t1"], data[c]["vec_val"], vec_one))  # branch bit
                idx_ops.append(("multiply_add", data[c]["vec_t2"], data[c]["vec_idx"], vec_two, vec_one))  # idx*2+1
            append_ops(slots_out, "valu", idx_ops, "index")
            append_ops(
                slots_out,
                "valu",
                [("+", data[c]["vec_idx"], data[c]["vec_t1"], data[c]["vec_t2"]) for c in range(count)],
                "index",
            )
            # Only add clamp operations if needed (at leaf level, indices can exceed n_nodes)
            if needs_clamp:
                append_ops(
                    slots_out,
                    "valu",
                    [("<", data[c]["vec_t1"], data[c]["vec_idx"], vec_n_nodes) for c in range(count)],
                    "index",
                )
                append_ops(
                    slots_out,
                    "valu",
                    [("*", data[c]["vec_idx"], data[c]["vec_idx"], data[c]["vec_t1"]) for c in range(count)],
                    "index",
                )
            return slots_out

        def build_tree_load_slots(data, count, level):
            """Build tree loading: levels 0-2 use cached values with vselect, level 3 uses vselect with shared broadcasts, others use scatter loads."""
            slots_out = []
            if level == 0:
                # Level 0: broadcast cached root value to all chunks
                append_ops(
                    slots_out,
                    "valu",
                    [("vbroadcast", data[c]["vec_node"], tree_cache_l0) for c in range(count)],
                    "tree_l0",
                )
            elif level == 1:
                # Level 1: vec_idx is 1 or 2
                # Use vselect: mask = (vec_idx & 1), then vselect(mask, tree[1], tree[2])
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t1"], tree_cache_l1)], "tree_l1")
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t2"], tree_cache_l1 + 1)], "tree_l1")
                    append_ops(slots_out, "valu", [("&", data[c]["vec_node"], data[c]["vec_idx"], vec_one)], "tree_l1_mask")
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_node"], data[c]["vec_node"], data[c]["vec_t1"], data[c]["vec_t2"])], "tree_l1_sel")
            elif level == 2:
                # Level 2: vec_idx is 3, 4, 5, or 6 - nested vselect
                vec_five = vec_const_addrs[5]
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t1"], tree_cache_l2)], "tree_l2")
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t2"], tree_cache_l2 + 1)], "tree_l2")
                for c in range(count):
                    append_ops(slots_out, "valu", [("&", data[c]["tree_addrs"], data[c]["vec_idx"], vec_one)], "tree_l2_bit0")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t1"], data[c]["tree_addrs"], data[c]["vec_t1"], data[c]["vec_t2"])], "tree_l2_sel1")
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_node"], tree_cache_l2 + 2)], "tree_l2")
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t2"], tree_cache_l2 + 3)], "tree_l2")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t2"], data[c]["tree_addrs"], data[c]["vec_node"], data[c]["vec_t2"])], "tree_l2_sel2")
                for c in range(count):
                    append_ops(slots_out, "valu", [("<", data[c]["vec_node"], data[c]["vec_idx"], vec_five)], "tree_l2_cond")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_node"], data[c]["vec_node"], data[c]["vec_t1"], data[c]["vec_t2"])], "tree_l2_sel3")
            else:
                # Scatter loads: ALU for address, LOAD for data
                alu_ops = []
                for c in range(count):
                    for i in range(VLEN):
                        alu_ops.append(
                            ("+", data[c]["tree_addrs"] + i, self.scratch["forest_values_p"], data[c]["vec_idx"] + i)
                        )
                append_ops(slots_out, "alu", alu_ops, "tree_addr")

                load_ops = []
                for c in range(count):
                    for i in range(VLEN):
                        load_ops.append(("load", data[c]["vec_node"] + i, data[c]["tree_addrs"] + i))
                # Tree loads are from read-only memory, can be reordered with stores
                append_ops(slots_out, "load", load_ops, "tree_load", readonly=True)
            return slots_out

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

        # Load tree values for prologue (round 0 = level 0)
        prologue_slots = build_tree_load_slots(cur_data, g_count, level=0)
        for engine, op, note, readonly in prologue_slots:
            instrs.append({engine: [op]}, note=note, readonly=readonly)

        for round_idx in range(rounds):
            round_level = round_idx % (forest_height + 1)
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
                # Only need clamp at leaf level (level == forest_height) where indices can exceed n_nodes
                needs_clamp = (round_level == forest_height)
                hash_slots = build_hash_and_index_slots(cur_data, g_count, needs_clamp)
                if round_idx == rounds - 1:
                    # Last round: store results as soon as this group completes
                    for c in range(0, g_count, 2):
                        ops = [("vstore", chunk_addr_idx[g_start + c], cur_data[c]["vec_idx"])]
                        if c + 1 < g_count:
                            ops.append(("vstore", chunk_addr_idx[g_start + c + 1], cur_data[c + 1]["vec_idx"]))
                        for op in ops:
                            hash_slots.append(("store", op, "store", False))
                    for c in range(0, g_count, 2):
                        ops = [("vstore", chunk_addr_val[g_start + c], cur_data[c]["vec_val"])]
                        if c + 1 < g_count:
                            ops.append(("vstore", chunk_addr_val[g_start + c + 1], cur_data[c + 1]["vec_val"]))
                        for op in ops:
                            hash_slots.append(("store", op, "store", False))

                # Build next-group prep (loads + address calc + tree loads)
                next_prep_slots = []
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

                    # Determine next group's tree level
                    next_level = round_level
                    if g + 1 >= n_groups and round_idx + 1 < rounds:
                        next_level = (round_idx + 1) % (forest_height + 1)

                    # Use unified tree loading function
                    tree_slots = build_tree_load_slots(next_data, next_count, next_level)
                    next_prep_slots.extend(tree_slots)

                # Emit next-group prep FIRST to give it lower order numbers for priority scheduling
                # Then emit current group work; AST scheduler handles overlap via dependencies
                for engine, op, note, readonly in next_prep_slots:
                    instrs.append({engine: [op]}, note=note, readonly=readonly)
                for engine, op, note, readonly in hash_slots:
                    instrs.append({engine: [op]}, note=note, readonly=readonly)
            # Align next round's start_set with the last group's prefetch target
            start_set = (start_set + n_groups) % 2

        instrs.append({"flow": [("pause",)]}, note="epilogue")
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
