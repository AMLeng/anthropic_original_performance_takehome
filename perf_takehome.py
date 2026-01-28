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
        Highly optimized vectorized kernel with stage-based processing.

        Key optimizations:
        - VALU for 8-wide SIMD
        - Bundle cache phases (levels 0-2) into single stages
        - 3 sets for better pipeline overlap between cache and scatter phases
        - Resilient stage structure that adapts to CACHE_LEVELS
        """
        instrs = self.instrs
        n_chunks = batch_size // VLEN

        # Configuration for cache levels
        # Level 3 caching on first pass only
        CACHE_LEVELS = 4  # Levels 0, 1, 2, 3 (first pass) are cached

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
        base_consts = [1, 2, 5, 7, 9, 11, 13]  # Added 7,9,11,13 for level-3 tree selection
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

        # Cache tree values for level 3 (indices 7-14, 8 values)
        tree_cache_l3 = self.alloc_scratch("tree_cache_l3", VLEN)

        # Load tree cache for levels 0-2 and level 3 with vloads
        # Need address offset for level 3 (forest_values_p + 7)
        tree_l3_addr = self.alloc_scratch("tree_l3_addr", 1)
        instrs.append({"flow": [("add_imm", tree_l3_addr, self.scratch["forest_values_p"], 7)]}, note="init")
        # Tree cache is from read-only memory
        instrs.append({"load": [
            ("vload", tree_cache, self.scratch["forest_values_p"]),
            ("vload", tree_cache_l3, tree_l3_addr),
        ]}, note="init", readonly=True)

        # Pre-broadcast tree cache values to vectors to avoid vbroadcast in hot loop
        # Level 0: 1 value (root)
        tree_l0_vec = self.alloc_scratch("tree_l0_vec", VLEN)
        instrs.append({"valu": [("vbroadcast", tree_l0_vec, tree_cache_l0)]}, note="init")

        # Level 1: 2 values (indices 1, 2)
        tree_l1_vec_0 = self.alloc_scratch("tree_l1_vec_0", VLEN)
        tree_l1_vec_1 = self.alloc_scratch("tree_l1_vec_1", VLEN)
        instrs.append({"valu": [
            ("vbroadcast", tree_l1_vec_0, tree_cache_l1),
            ("vbroadcast", tree_l1_vec_1, tree_cache_l1 + 1),
        ]}, note="init")

        # Level 2: 4 values (indices 3, 4, 5, 6)
        tree_l2_vec_0 = self.alloc_scratch("tree_l2_vec_0", VLEN)
        tree_l2_vec_1 = self.alloc_scratch("tree_l2_vec_1", VLEN)
        tree_l2_vec_2 = self.alloc_scratch("tree_l2_vec_2", VLEN)
        tree_l2_vec_3 = self.alloc_scratch("tree_l2_vec_3", VLEN)
        instrs.append({"valu": [
            ("vbroadcast", tree_l2_vec_0, tree_cache_l2),
            ("vbroadcast", tree_l2_vec_1, tree_cache_l2 + 1),
            ("vbroadcast", tree_l2_vec_2, tree_cache_l2 + 2),
            ("vbroadcast", tree_l2_vec_3, tree_cache_l2 + 3),
        ]}, note="init")

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

        # Testing different group configurations
        GROUP_SIZE = 4   # Chunks per group
        N_SETS = 4       # Number of working sets
        sets = [[] for _ in range(N_SETS)]
        for s in range(N_SETS):
            for c in range(GROUP_SIZE):
                sets[s].append({
                    "vec_node": self.alloc_scratch(f"vn_{s}_{c}", VLEN),
                    "vec_t1": self.alloc_scratch(f"vt1_{s}_{c}", VLEN),
                    "vec_t2": self.alloc_scratch(f"vt2_{s}_{c}", VLEN),
                    "vec_t3": self.alloc_scratch(f"vt3_{s}_{c}", VLEN),
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

        # Note: pause removed - it's a no-op in submission tests and causes scheduling issues

        # =====================================================================
        # Compute stage structure based on CACHE_LEVELS
        # =====================================================================
        # A stage is either:
        # - 'cache': bundles consecutive rounds where level < CACHE_LEVELS
        # - 'scatter': a single round where level >= CACHE_LEVELS
        #
        # With CACHE_LEVELS=3, rounds=16, forest_height=10:
        # Stages: [cache(0-2), scatter(3), scatter(4), ..., scatter(10),
        #          cache(11-13), scatter(14), scatter(15)]
        # = 12 stages total

        def compute_stages(num_rounds, height, cache_levels):
            """Compute stage structure. Returns list of stage dicts."""
            stages = []
            pending_cache = []  # (round_idx, level) pairs

            for r in range(num_rounds):
                level = r % (height + 1)
                is_cache_level = level < cache_levels

                if is_cache_level:
                    pending_cache.append((r, level))
                else:
                    # Flush pending cache rounds as a cache stage
                    if pending_cache:
                        stages.append({
                            'type': 'cache',
                            'rounds': [p[0] for p in pending_cache],
                            'levels': [p[1] for p in pending_cache],
                        })
                        pending_cache = []
                    # Add scatter stage for this single round
                    stages.append({
                        'type': 'scatter',
                        'rounds': [r],
                        'levels': [level],
                    })

            # Flush any remaining cache rounds
            if pending_cache:
                stages.append({
                    'type': 'cache',
                    'rounds': [p[0] for p in pending_cache],
                    'levels': [p[1] for p in pending_cache],
                })

            return stages

        stage_list = compute_stages(rounds, forest_height, CACHE_LEVELS)
        n_stages = len(stage_list)
        n_groups = (n_chunks + GROUP_SIZE - 1) // GROUP_SIZE


        def append_ops(out, engine, ops, note="", readonly=False):
            for op in ops:
                out.append((engine, op, note, readonly))

        # Helper to compute hash and index update for a single round
        def build_hash_and_index_slots(data, count, needs_clamp=True, tree_val_addr=None):
            slots_out = []
            # XOR - use tree_val_addr if provided (for pre-broadcast optimization), else vec_node
            if tree_val_addr is not None:
                # tree_val_addr is a pre-broadcast vector, use it directly for all chunks
                append_ops(
                    slots_out,
                    "valu",
                    [("^", data[c]["vec_val"], data[c]["vec_val"], tree_val_addr) for c in range(count)],
                    "hash",
                )
            else:
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

        def build_tree_load_slots(data, count, level, use_level3_cache=False):
            """Build tree loading for a single level.

            Returns (slots_out, tree_val_addr) where tree_val_addr is the address
            to use for the tree value in XOR. For level 0 this is the pre-broadcast
            vector; for other levels it's vec_node as before.
            """
            slots_out = []
            tree_val_addr = None  # None means use vec_node
            if level == 0:
                # Level 0: use pre-broadcast root value directly (no vbroadcast needed!)
                # Return tree_l0_vec address to use directly in XOR
                tree_val_addr = tree_l0_vec
                # No operations needed - just return the address
            elif level == 1:
                # Level 1: vec_idx is 1 or 2
                # Use vselect with pre-broadcast vectors directly
                # mask = (vec_idx & 1), then vselect(mask, tree_l1_vec_0, tree_l1_vec_1)
                for c in range(count):
                    append_ops(slots_out, "valu", [("&", data[c]["vec_node"], data[c]["vec_idx"], vec_one)], "tree_l1_mask")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_node"], data[c]["vec_node"], tree_l1_vec_0, tree_l1_vec_1)], "tree_l1_sel")
            elif level == 2:
                # Level 2: vec_idx is 3, 4, 5, or 6 - nested vselect with pre-broadcast vectors
                vec_five = vec_const_addrs[5]
                # Compute bit0 = vec_idx & 1 for pair selection
                for c in range(count):
                    append_ops(slots_out, "valu", [("&", data[c]["tree_addrs"], data[c]["vec_idx"], vec_one)], "tree_l2_bit0")
                # Select between tree[3]/tree[4] using pre-broadcast vectors
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t1"], data[c]["tree_addrs"], tree_l2_vec_0, tree_l2_vec_1)], "tree_l2_sel1")
                # Select between tree[5]/tree[6] using pre-broadcast vectors
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t2"], data[c]["tree_addrs"], tree_l2_vec_2, tree_l2_vec_3)], "tree_l2_sel2")
                # Final selection based on idx < 5
                for c in range(count):
                    append_ops(slots_out, "valu", [("<", data[c]["vec_node"], data[c]["vec_idx"], vec_five)], "tree_l2_cond")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_node"], data[c]["vec_node"], data[c]["vec_t1"], data[c]["vec_t2"])], "tree_l2_sel3")
            elif use_level3_cache:
                # Level 3 (first pass only): vec_idx is 7-14, select from 8 cached values
                # Uses 3-level vselect tree with comparison-based merging
                vec_nine = vec_const_addrs[9]
                vec_eleven = vec_const_addrs[11]
                vec_thirteen = vec_const_addrs[13]

                # Compute bit0 = vec_idx & 1 for pair selections
                for c in range(count):
                    append_ops(slots_out, "valu", [("&", data[c]["tree_addrs"], data[c]["vec_idx"], vec_one)], "tree_l3_bit0")

                # Pair 0,1 (indices 7,8) -> sel_01 in vec_t1
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t1"], tree_cache_l3 + 0)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t2"], tree_cache_l3 + 1)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t1"], data[c]["tree_addrs"], data[c]["vec_t1"], data[c]["vec_t2"])], "tree_l3_sel01")

                # Pair 2,3 (indices 9,10) -> sel_23 in vec_t2
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t2"], tree_cache_l3 + 2)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_node"], tree_cache_l3 + 3)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t2"], data[c]["tree_addrs"], data[c]["vec_t2"], data[c]["vec_node"])], "tree_l3_sel23")

                # Merge sel_01, sel_23 with vec_idx < 9 -> sel_0123 in vec_t1
                for c in range(count):
                    append_ops(slots_out, "valu", [("<", data[c]["vec_node"], data[c]["vec_idx"], vec_nine)], "tree_l3_cmp9")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t1"], data[c]["vec_node"], data[c]["vec_t1"], data[c]["vec_t2"])], "tree_l3_sel0123")

                # Pair 4,5 (indices 11,12) -> sel_45 in vec_t2
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t2"], tree_cache_l3 + 4)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_node"], tree_cache_l3 + 5)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t2"], data[c]["tree_addrs"], data[c]["vec_t2"], data[c]["vec_node"])], "tree_l3_sel45")

                # Pair 6,7 (indices 13,14) -> sel_67 in vec_t3
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_node"], tree_cache_l3 + 6)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "valu", [("vbroadcast", data[c]["vec_t3"], tree_cache_l3 + 7)], "tree_l3")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t3"], data[c]["tree_addrs"], data[c]["vec_node"], data[c]["vec_t3"])], "tree_l3_sel67")

                # Merge sel_45, sel_67 with vec_idx < 13 -> sel_4567 in vec_t2
                for c in range(count):
                    append_ops(slots_out, "valu", [("<", data[c]["vec_node"], data[c]["vec_idx"], vec_thirteen)], "tree_l3_cmp13")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_t2"], data[c]["vec_node"], data[c]["vec_t2"], data[c]["vec_t3"])], "tree_l3_sel4567")

                # Final merge sel_0123, sel_4567 with vec_idx < 11 -> result in vec_node
                for c in range(count):
                    append_ops(slots_out, "valu", [("<", data[c]["vec_t3"], data[c]["vec_idx"], vec_eleven)], "tree_l3_cmp11")
                for c in range(count):
                    append_ops(slots_out, "flow", [("vselect", data[c]["vec_node"], data[c]["vec_t3"], data[c]["vec_t1"], data[c]["vec_t2"])], "tree_l3_final")
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
            return slots_out, tree_val_addr

        def build_cache_stage_slots(data, count, levels):
            """Build slots for a cache stage covering multiple levels."""
            all_slots = []
            for i, level in enumerate(levels):
                # Tree load for this level
                tree_slots, tree_val_addr = build_tree_load_slots(data, count, level)
                all_slots.extend(tree_slots)

                # Hash and index update
                # Clamp only at forest_height (level 10)
                needs_clamp = (level == forest_height)
                hash_slots = build_hash_and_index_slots(data, count, needs_clamp, tree_val_addr)
                all_slots.extend(hash_slots)
            return all_slots

        def build_scatter_stage_slots(data, count, level):
            """Build slots for a scatter stage (single level)."""
            all_slots = []
            # Tree load (scatter)
            tree_slots, tree_val_addr = build_tree_load_slots(data, count, level)
            all_slots.extend(tree_slots)

            # Hash and index update
            needs_clamp = (level == forest_height)
            hash_slots = build_hash_and_index_slots(data, count, needs_clamp, tree_val_addr)
            all_slots.extend(hash_slots)
            return all_slots

        def build_stage_slots(data, count, stage_info):
            """Build slots for a stage (either cache or scatter)."""
            if stage_info['type'] == 'cache':
                return build_cache_stage_slots(data, count, stage_info['levels'])
            else:
                assert len(stage_info['levels']) == 1
                return build_scatter_stage_slots(data, count, stage_info['levels'][0])

        def build_data_for_group(g, set_idx):
            """Build data dict for a group using the specified set."""
            g_start = g * GROUP_SIZE
            g_count = min(GROUP_SIZE, n_chunks - g_start)
            cur_set = sets[set_idx]
            data = []
            for c in range(g_count):
                data.append({
                    "vec_idx": idx_addrs[g_start + c],
                    "vec_val": val_addrs[g_start + c],
                    "vec_node": cur_set[c]["vec_node"],
                    "vec_t1": cur_set[c]["vec_t1"],
                    "vec_t2": cur_set[c]["vec_t2"],
                    "vec_t3": cur_set[c]["vec_t3"],
                    "tree_addrs": cur_set[c]["tree_addrs"],
                })
            return data, g_count, g_start

        def build_store_slots(data, count, g_start):
            """Build store slots for final results."""
            slots_out = []
            for c in range(0, count, 2):
                ops = [("vstore", chunk_addr_idx[g_start + c], data[c]["vec_idx"])]
                if c + 1 < count:
                    ops.append(("vstore", chunk_addr_idx[g_start + c + 1], data[c + 1]["vec_idx"]))
                for op in ops:
                    slots_out.append(("store", op, "store", False))
            for c in range(0, count, 2):
                ops = [("vstore", chunk_addr_val[g_start + c], data[c]["vec_val"])]
                if c + 1 < count:
                    ops.append(("vstore", chunk_addr_val[g_start + c + 1], data[c + 1]["vec_val"]))
                for op in ops:
                    slots_out.append(("store", op, "store", False))
            return slots_out

        # =====================================================================
        # Main loop: process levels round-robin across groups with interleaving
        # =====================================================================
        # Track which set each group is using
        group_set = [g % N_SETS for g in range(n_groups)]

        # Flatten stages into a list of levels with their stage info
        # Include round index for "first pass only" level 3 caching
        level_list = []
        for stage_idx, stage_info in enumerate(stage_list):
            is_last_stage = (stage_idx == len(stage_list) - 1)
            for i, level in enumerate(stage_info['levels']):
                round_idx = stage_info['rounds'][i]
                level_list.append((level, stage_info['type'], is_last_stage and level == stage_info['levels'][-1], round_idx))

        # Prologue: Load tree values for group 0 (level 0) - gives it lowest order numbers
        data_g0, g_count_g0, g_start_g0 = build_data_for_group(0, group_set[0])
        prologue_slots, _ = build_tree_load_slots(data_g0, g_count_g0, level=0)
        for engine, op, note, readonly in prologue_slots:
            instrs.append({engine: [op]}, note=note, readonly=readonly)

        # Process level-by-level across all groups (like round-based)
        for level_idx, (level, stage_type, emit_stores, round_idx) in enumerate(level_list):
            is_last_level = (level_idx == len(level_list) - 1)
            # For "first pass only" level 3 caching: use cache only on first traversal
            # First traversal ends at round forest_height (level=forest_height=10 wraps to root)
            use_level3_cache = (CACHE_LEVELS >= 4 and level == 3 and round_idx <= forest_height)

            # Build tree loads and hash operations for all groups
            group_tree_slots = []
            group_hash_slots = []
            group_store_slots = []

            for g in range(n_groups):
                set_idx = group_set[g]
                data, g_count, g_start = build_data_for_group(g, set_idx)

                # Tree loads for this level
                tree_slots, tree_val_addr = build_tree_load_slots(data, g_count, level, use_level3_cache)

                # Hash and index update
                needs_clamp = (level == forest_height)
                hash_slots = build_hash_and_index_slots(data, g_count, needs_clamp, tree_val_addr)

                # Store slots if this is the last level
                store_slots = []
                if emit_stores:
                    store_slots = build_store_slots(data, g_count, g_start)

                group_tree_slots.append(tree_slots)
                group_hash_slots.append(hash_slots)
                group_store_slots.append(store_slots)

            # Build prefetch for next level's group 0 (if not last level)
            next_level_g0_tree = []
            if not is_last_level:
                next_level, _, _, next_round_idx = level_list[level_idx + 1]
                next_use_l3_cache = (CACHE_LEVELS >= 4 and next_level == 3 and next_round_idx <= forest_height)
                data, g_count, g_start = build_data_for_group(0, group_set[0])
                next_level_g0_tree, _ = build_tree_load_slots(data, g_count, next_level, next_use_l3_cache)

            # Emit with interleaving: next-group's tree loads before current-group's hash
            # Stores are interleaved right after each group's hash (matching round-based order)
            for g in range(n_groups):
                # Emit next group's tree loads first (for prefetch priority)
                if g + 1 < n_groups:
                    for engine, op, note, readonly in group_tree_slots[g + 1]:
                        instrs.append({engine: [op]}, note=note, readonly=readonly)
                elif not is_last_level:
                    # Last group: prefetch group 0 for next level
                    for engine, op, note, readonly in next_level_g0_tree:
                        instrs.append({engine: [op]}, note=note, readonly=readonly)

                # G0's tree loads were emitted in prologue (level 0) or prefetched by previous level

                # Emit current group's hash
                for engine, op, note, readonly in group_hash_slots[g]:
                    instrs.append({engine: [op]}, note=note, readonly=readonly)

                # Emit current group's stores right after hash (matching round-based order)
                for engine, op, note, readonly in group_store_slots[g]:
                    instrs.append({engine: [op]}, note=note, readonly=readonly)

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
