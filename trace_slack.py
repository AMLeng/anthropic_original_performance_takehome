"""
Detailed tracing to understand slack differences between emission orders.
"""

from collections import defaultdict
from kernel_ast import ASTNode, EngineKind, scratch_reads, scratch_writes
from problem import SLOT_LIMITS, VLEN, HASH_STAGES

TARGET_NODES = ["g1_r5_tree_addr", "g1_r5_tree_load", "g1_r5_hash"]


class TracingScheduler:
    def __init__(self, name):
        self.name = name
        self.ast_nodes = []
        self._node_order = 0
        self._last_write = {}
        self._last_read = {}
        self._last_load = None
        self._last_store = None
        self._last_barrier = None
        self.node_by_order = {}

    def add_ast(self, engine, slot, note="", readonly=False):
        engine_kind = EngineKind(engine) if not isinstance(engine, EngineKind) else engine
        op = slot[0]
        node = ASTNode(engine=engine_kind, op=op, operands=slot, note=note, order=self._node_order)
        self._node_order += 1
        self.node_by_order[node.order] = node

        reads = scratch_reads(engine_kind, op, slot)
        writes = scratch_writes(engine_kind, op, slot)

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

        if engine_kind == EngineKind.LOAD:
            if self._last_store is not None and not readonly:
                node.add_dep(self._last_store)
            self._last_load = node
        elif engine_kind == EngineKind.STORE:
            if self._last_load is not None:
                node.add_dep(self._last_load)
            if self._last_store is not None:
                node.add_dep(self._last_store)
            self._last_store = node

        if engine_kind == EngineKind.FLOW and op == "pause":
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
            node.add_dep(self._last_barrier)

        self.ast_nodes.append(node)
        return node

    def compute_metrics(self):
        """Compute all metrics and return detailed info."""
        # dist_to_pause (longest path to pause, going backwards through users)
        dist_to_pause = {n: 0 for n in self.ast_nodes}
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

        # earliest_start (longest path from sources, going forward through deps)
        earliest_start = {n: 0 for n in self.ast_nodes}
        dep_count = {n: len(n.deps) for n in self.ast_nodes}
        queue = [n for n in self.ast_nodes if dep_count[n] == 0]
        while queue:
            node = queue.pop(0)
            for user in node.users:
                new_start = earliest_start[node] + 1
                if new_start > earliest_start[user]:
                    earliest_start[user] = new_start
                dep_count[user] -= 1
                if dep_count[user] == 0:
                    queue.append(user)

        critical_path_length = max(earliest_start.values()) + 1 if earliest_start else 1

        # Compute slack
        slack = {}
        for n in self.ast_nodes:
            slack[n] = critical_path_length - earliest_start[n] - dist_to_pause[n]

        return {
            'dist_to_pause': dist_to_pause,
            'earliest_start': earliest_start,
            'critical_path_length': critical_path_length,
            'slack': slack,
        }

    def find_nodes_by_note_prefix(self, prefix):
        """Find all nodes whose note starts with prefix."""
        return [n for n in self.ast_nodes if n.note.startswith(prefix)]

    def trace_path_to_sink(self, node, metrics):
        """Trace the longest path from node to sink (pause)."""
        dist_to_pause = metrics['dist_to_pause']
        path = [node]
        current = node
        while dist_to_pause[current] > 0:
            # Find the user with the highest dist_to_pause
            best_user = None
            best_dist = -1
            for user in current.users:
                if dist_to_pause[user] > best_dist:
                    best_dist = dist_to_pause[user]
                    best_user = user
            if best_user is None:
                break
            path.append(best_user)
            current = best_user
        return path

    def trace_path_from_source(self, node, metrics):
        """Trace the longest path from source to node."""
        earliest_start = metrics['earliest_start']
        path = [node]
        current = node
        while earliest_start[current] > 0:
            # Find the dep with the highest earliest_start
            best_dep = None
            best_start = -1
            for dep in current.deps:
                if earliest_start[dep] > best_start:
                    best_start = earliest_start[dep]
                    best_dep = dep
            if best_dep is None:
                break
            path.insert(0, best_dep)
            current = best_dep
        return path

    def get_dependency_breakdown(self, node):
        """Get breakdown of why this node has its dependencies."""
        deps_by_type = {
            'scratch_raw': [],  # Read-after-write
            'scratch_war': [],  # Write-after-read
            'scratch_waw': [],  # Write-after-write
            'last_load': [],    # Memory ordering (load after store)
            'last_store': [],   # Memory ordering (store after load/store)
            'barrier': [],      # Pause barrier
        }
        # We can't easily reconstruct this after the fact, but we can show the deps
        return list(node.deps)


def build_common_init(scheduler, n_chunks, forest_height):
    """Build common initialization code."""
    init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                 "forest_values_p", "inp_indices_p", "inp_values_p"]
    scratch = {}
    scratch_ptr = [0]

    def alloc_scratch(name=None, length=1):
        addr = scratch_ptr[0]
        if name is not None:
            scratch[name] = addr
        scratch_ptr[0] += length
        return addr

    for v in init_vars:
        alloc_scratch(v, 1)

    for i in range(0, len(init_vars), 2):
        n = min(2, len(init_vars) - i)
        for j in range(n):
            scheduler.add_ast("load", ("const", scratch[init_vars[i + j]], i + j), note="init")
        for j in range(n):
            scheduler.add_ast("load", ("load", scratch[init_vars[i + j]], scratch[init_vars[i + j]]), note="init")

    mul_vals = []
    for op1, val1, op2, op3, val3 in HASH_STAGES:
        if op1 == "+" and op2 == "+" and op3 == "<<":
            mul_vals.append((1 + (1 << val3)) % (2**32))
    base_consts = [1, 2, 5]
    const_vals = list(set(base_consts + mul_vals + [s[1] for s in HASH_STAGES] + [s[4] for s in HASH_STAGES]))
    const_vals = [v for v in const_vals if v != 0]

    vec_const_addrs = {0: alloc_scratch("vc_0", VLEN)}
    for val in const_vals:
        vec_addr = alloc_scratch(f"vc_{val}", VLEN)
        vec_const_addrs[val] = vec_addr

    for i in range(0, len(const_vals), 2):
        for j in range(min(2, len(const_vals) - i)):
            val = const_vals[i + j]
            scheduler.add_ast("load", ("const", vec_const_addrs[val], val), note="init")

    for val in const_vals:
        scheduler.add_ast("valu", ("vbroadcast", vec_const_addrs[val], vec_const_addrs[val]), note="init")

    vec_n_nodes = alloc_scratch("vec_n_nodes", VLEN)
    scheduler.add_ast("valu", ("vbroadcast", vec_n_nodes, scratch["n_nodes"]), note="init")

    vec_zero, vec_one, vec_two = vec_const_addrs[0], vec_const_addrs[1], vec_const_addrs[2]
    hash_stage_info = []
    for op1, val1, op2, op3, val3 in HASH_STAGES:
        if op1 == "+" and op2 == "+" and op3 == "<<":
            mul_val = (1 + (1 << val3)) % (2**32)
            hash_stage_info.append(("madd", vec_const_addrs[mul_val], vec_const_addrs[val1]))
        else:
            hash_stage_info.append((op1, vec_const_addrs[val1], op2, op3, vec_const_addrs[val3]))

    tree_cache = alloc_scratch("tree_cache", VLEN)
    tree_cache_l0 = tree_cache + 0
    tree_cache_l1 = tree_cache + 1
    tree_cache_l2 = tree_cache + 3
    scheduler.add_ast("load", ("vload", tree_cache, scratch["forest_values_p"]), note="init", readonly=True)

    chunk_addr_idx = [alloc_scratch(f"cai_{c}") for c in range(n_chunks)]
    chunk_addr_val = [alloc_scratch(f"cav_{c}") for c in range(n_chunks)]
    offset_temps = [alloc_scratch(f"off_{c}") for c in range(n_chunks)]

    for c in range(0, n_chunks, 2):
        scheduler.add_ast("load", ("const", offset_temps[c], c * VLEN), note="init")
        if c + 1 < n_chunks:
            scheduler.add_ast("load", ("const", offset_temps[c + 1], (c + 1) * VLEN), note="init")

    for c in range(0, n_chunks, 12):
        for i in range(min(12, n_chunks - c)):
            scheduler.add_ast("alu", ("+", chunk_addr_idx[c + i], scratch["inp_indices_p"], offset_temps[c + i]), note="init")
    for c in range(0, n_chunks, 12):
        for i in range(min(12, n_chunks - c)):
            scheduler.add_ast("alu", ("+", chunk_addr_val[c + i], scratch["inp_values_p"], offset_temps[c + i]), note="init")

    GROUP_SIZE = 10
    N_SETS = 2
    sets = [[] for _ in range(N_SETS)]
    for s in range(N_SETS):
        for c in range(GROUP_SIZE):
            sets[s].append({
                "vec_node": alloc_scratch(f"vn_{s}_{c}", VLEN),
                "vec_t1": alloc_scratch(f"vt1_{s}_{c}", VLEN),
                "vec_t2": alloc_scratch(f"vt2_{s}_{c}", VLEN),
                "tree_addrs": alloc_scratch(f"ta_{s}_{c}", VLEN),
            })

    base_idx = alloc_scratch("vec_idx_all", n_chunks * VLEN)
    base_val = alloc_scratch("vec_val_all", n_chunks * VLEN)
    idx_addrs = [base_idx + c * VLEN for c in range(n_chunks)]
    val_addrs = [base_val + c * VLEN for c in range(n_chunks)]

    for c in range(0, n_chunks, 2):
        scheduler.add_ast("load", ("vload", val_addrs[c], chunk_addr_val[c]), note="init")
        if c + 1 < n_chunks:
            scheduler.add_ast("load", ("vload", val_addrs[c + 1], chunk_addr_val[c + 1]), note="init")

    scheduler.add_ast("flow", ("pause",), note="init")

    return {
        'scratch': scratch, 'vec_const_addrs': vec_const_addrs, 'vec_n_nodes': vec_n_nodes,
        'vec_one': vec_one, 'vec_two': vec_two, 'hash_stage_info': hash_stage_info,
        'tree_cache_l0': tree_cache_l0, 'tree_cache_l1': tree_cache_l1, 'tree_cache_l2': tree_cache_l2,
        'chunk_addr_idx': chunk_addr_idx, 'chunk_addr_val': chunk_addr_val,
        'sets': sets, 'idx_addrs': idx_addrs, 'val_addrs': val_addrs,
        'GROUP_SIZE': GROUP_SIZE, 'N_SETS': N_SETS,
    }


def build_hash_slots(data, count, g, round_num, state, forest_height):
    slots_out = []
    note_prefix = f"g{g}_r{round_num}"
    vec_one = state['vec_one']
    vec_two = state['vec_two']
    vec_n_nodes = state['vec_n_nodes']
    hash_stage_info = state['hash_stage_info']
    needs_clamp = (round_num % (forest_height + 1) == forest_height)

    for c in range(count):
        slots_out.append(("valu", ("^", data[c]["vec_val"], data[c]["vec_val"], data[c]["vec_node"]), f"{note_prefix}_hash"))

    nonmadd_stage_idx = 0
    for info in hash_stage_info:
        if info[0] == "madd":
            _, vmul, vadd = info
            for c in range(count):
                slots_out.append(("valu", ("multiply_add", data[c]["vec_val"], data[c]["vec_val"], vmul, vadd), f"{note_prefix}_hash"))
        else:
            op1, vc1, op2, op3, vc2 = info
            if nonmadd_stage_idx == 0:
                for c in range(count):
                    for i in range(VLEN):
                        slots_out.append(("alu", (op1, data[c]["vec_t1"] + i, data[c]["vec_val"] + i, vc1 + i), f"{note_prefix}_hash_alu"))
                        slots_out.append(("alu", (op3, data[c]["vec_t2"] + i, data[c]["vec_val"] + i, vc2 + i), f"{note_prefix}_hash_alu"))
                    for c in range(count):
                        for i in range(VLEN):
                            slots_out.append(("alu", (op2, data[c]["vec_val"] + i, data[c]["vec_t1"] + i, data[c]["vec_t2"] + i), f"{note_prefix}_hash_alu"))
            else:
                for c in range(count):
                    slots_out.append(("valu", (op1, data[c]["vec_t1"], data[c]["vec_val"], vc1), f"{note_prefix}_hash"))
                    slots_out.append(("valu", (op3, data[c]["vec_t2"], data[c]["vec_val"], vc2), f"{note_prefix}_hash"))
                for c in range(count):
                    slots_out.append(("valu", (op2, data[c]["vec_val"], data[c]["vec_t1"], data[c]["vec_t2"]), f"{note_prefix}_hash"))
            nonmadd_stage_idx += 1

    for c in range(count):
        slots_out.append(("valu", ("&", data[c]["vec_t1"], data[c]["vec_val"], vec_one), f"{note_prefix}_index"))
        slots_out.append(("valu", ("multiply_add", data[c]["vec_t2"], data[c]["vec_idx"], vec_two, vec_one), f"{note_prefix}_index"))
    for c in range(count):
        slots_out.append(("valu", ("+", data[c]["vec_idx"], data[c]["vec_t1"], data[c]["vec_t2"]), f"{note_prefix}_index"))

    if needs_clamp:
        for c in range(count):
            slots_out.append(("valu", ("<", data[c]["vec_t1"], data[c]["vec_idx"], vec_n_nodes), f"{note_prefix}_index"))
        for c in range(count):
            slots_out.append(("valu", ("*", data[c]["vec_idx"], data[c]["vec_idx"], data[c]["vec_t1"]), f"{note_prefix}_index"))

    return slots_out


def build_tree_load_slots(data, count, level, g, round_num, state):
    slots_out = []
    note_prefix = f"g{g}_r{round_num}"
    tree_cache_l0 = state['tree_cache_l0']
    tree_cache_l1 = state['tree_cache_l1']
    tree_cache_l2 = state['tree_cache_l2']
    vec_one = state['vec_one']
    vec_five = state['vec_const_addrs'][5]
    scratch = state['scratch']

    if level == 0:
        for c in range(count):
            slots_out.append(("valu", ("vbroadcast", data[c]["vec_node"], tree_cache_l0), f"{note_prefix}_tree_l0"))
    elif level == 1:
        for c in range(count):
            slots_out.append(("valu", ("vbroadcast", data[c]["vec_t1"], tree_cache_l1), f"{note_prefix}_tree_l1"))
            slots_out.append(("valu", ("vbroadcast", data[c]["vec_t2"], tree_cache_l1 + 1), f"{note_prefix}_tree_l1"))
            slots_out.append(("valu", ("&", data[c]["vec_node"], data[c]["vec_idx"], vec_one), f"{note_prefix}_tree_l1_mask"))
            slots_out.append(("flow", ("vselect", data[c]["vec_node"], data[c]["vec_node"], data[c]["vec_t1"], data[c]["vec_t2"]), f"{note_prefix}_tree_l1_sel"))
    elif level == 2:
        for c in range(count):
            slots_out.append(("valu", ("vbroadcast", data[c]["vec_t1"], tree_cache_l2), f"{note_prefix}_tree_l2"))
        for c in range(count):
            slots_out.append(("valu", ("vbroadcast", data[c]["vec_t2"], tree_cache_l2 + 1), f"{note_prefix}_tree_l2"))
        for c in range(count):
            slots_out.append(("valu", ("&", data[c]["tree_addrs"], data[c]["vec_idx"], vec_one), f"{note_prefix}_tree_l2_bit0"))
        for c in range(count):
            slots_out.append(("flow", ("vselect", data[c]["vec_t1"], data[c]["tree_addrs"], data[c]["vec_t1"], data[c]["vec_t2"]), f"{note_prefix}_tree_l2_sel1"))
        for c in range(count):
            slots_out.append(("valu", ("vbroadcast", data[c]["vec_node"], tree_cache_l2 + 2), f"{note_prefix}_tree_l2"))
        for c in range(count):
            slots_out.append(("valu", ("vbroadcast", data[c]["vec_t2"], tree_cache_l2 + 3), f"{note_prefix}_tree_l2"))
        for c in range(count):
            slots_out.append(("flow", ("vselect", data[c]["vec_t2"], data[c]["tree_addrs"], data[c]["vec_node"], data[c]["vec_t2"]), f"{note_prefix}_tree_l2_sel2"))
        for c in range(count):
            slots_out.append(("valu", ("<", data[c]["vec_node"], data[c]["vec_idx"], vec_five), f"{note_prefix}_tree_l2_cond"))
        for c in range(count):
            slots_out.append(("flow", ("vselect", data[c]["vec_node"], data[c]["vec_node"], data[c]["vec_t1"], data[c]["vec_t2"]), f"{note_prefix}_tree_l2_sel3"))
    else:
        for c in range(count):
            for i in range(VLEN):
                slots_out.append(("alu", ("+", data[c]["tree_addrs"] + i, scratch["forest_values_p"], data[c]["vec_idx"] + i), f"{note_prefix}_tree_addr"))
        for c in range(count):
            for i in range(VLEN):
                slots_out.append(("load", ("load", data[c]["vec_node"] + i, data[c]["tree_addrs"] + i), f"{note_prefix}_tree_load", True))

    return slots_out


def build_store_slots(data, count, g_start, g, round_num, state):
    slots_out = []
    note_prefix = f"g{g}_r{round_num}"
    chunk_addr_idx = state['chunk_addr_idx']
    chunk_addr_val = state['chunk_addr_val']
    for c in range(0, count, 2):
        slots_out.append(("store", ("vstore", chunk_addr_idx[g_start + c], data[c]["vec_idx"]), f"{note_prefix}_store"))
        if c + 1 < count:
            slots_out.append(("store", ("vstore", chunk_addr_idx[g_start + c + 1], data[c + 1]["vec_idx"]), f"{note_prefix}_store"))
    for c in range(0, count, 2):
        slots_out.append(("store", ("vstore", chunk_addr_val[g_start + c], data[c]["vec_val"]), f"{note_prefix}_store"))
        if c + 1 < count:
            slots_out.append(("store", ("vstore", chunk_addr_val[g_start + c + 1], data[c + 1]["vec_val"]), f"{note_prefix}_store"))
    return slots_out


def emit_slots(scheduler, slots):
    for slot in slots:
        if len(slot) == 4:
            engine, op, note, readonly = slot
            scheduler.add_ast(engine, op, note=note, readonly=readonly)
        else:
            engine, op, note = slot
            scheduler.add_ast(engine, op, note=note)


def build_kernel_stage_based(scheduler, forest_height, n_chunks, rounds, state):
    CACHE_LEVELS = 3
    GROUP_SIZE = state['GROUP_SIZE']
    N_SETS = state['N_SETS']
    sets = state['sets']
    idx_addrs = state['idx_addrs']
    val_addrs = state['val_addrs']

    def compute_stages(num_rounds, height, cache_levels):
        stages = []
        pending_cache = []
        for r in range(num_rounds):
            level = r % (height + 1)
            if level < cache_levels:
                pending_cache.append((r, level))
            else:
                if pending_cache:
                    stages.append({'type': 'cache', 'rounds': [p[0] for p in pending_cache], 'levels': [p[1] for p in pending_cache]})
                    pending_cache = []
                stages.append({'type': 'scatter', 'rounds': [r], 'levels': [level]})
        if pending_cache:
            stages.append({'type': 'cache', 'rounds': [p[0] for p in pending_cache], 'levels': [p[1] for p in pending_cache]})
        return stages

    stage_list = compute_stages(rounds, forest_height, CACHE_LEVELS)
    n_stages = len(stage_list)
    n_groups = (n_chunks + GROUP_SIZE - 1) // GROUP_SIZE

    def build_data(g, set_idx):
        g_start = g * GROUP_SIZE
        g_count = min(GROUP_SIZE, n_chunks - g_start)
        cur_set = sets[set_idx]
        data = [{"vec_idx": idx_addrs[g_start + c], "vec_val": val_addrs[g_start + c],
                 "vec_node": cur_set[c]["vec_node"], "vec_t1": cur_set[c]["vec_t1"],
                 "vec_t2": cur_set[c]["vec_t2"], "tree_addrs": cur_set[c]["tree_addrs"]} for c in range(g_count)]
        return data, g_count, g_start

    group_set = [g % N_SETS for g in range(n_groups)]

    for stage_idx in range(n_stages):
        stage_info = stage_list[stage_idx]
        is_last_stage = (stage_idx == n_stages - 1)

        for g in range(n_groups):
            set_idx = group_set[g]
            data, g_count, g_start = build_data(g, set_idx)

            if stage_info['type'] == 'cache':
                for level_idx, level in enumerate(stage_info['levels']):
                    round_num = stage_info['rounds'][level_idx]
                    emit_slots(scheduler, build_tree_load_slots(data, g_count, level, g, round_num, state))
                    emit_slots(scheduler, build_hash_slots(data, g_count, g, round_num, state, forest_height))
            else:
                level = stage_info['levels'][0]
                round_num = stage_info['rounds'][0]
                emit_slots(scheduler, build_tree_load_slots(data, g_count, level, g, round_num, state))
                emit_slots(scheduler, build_hash_slots(data, g_count, g, round_num, state, forest_height))

            if is_last_stage:
                emit_slots(scheduler, build_store_slots(data, g_count, g_start, g, stage_info['rounds'][-1], state))

        group_set = [(s + 1) % N_SETS for s in group_set]

    scheduler.add_ast("flow", ("pause",), note="epilogue")


def build_kernel_round_based(scheduler, forest_height, n_chunks, rounds, state):
    GROUP_SIZE = state['GROUP_SIZE']
    N_SETS = state['N_SETS']
    sets = state['sets']
    idx_addrs = state['idx_addrs']
    val_addrs = state['val_addrs']

    n_groups = (n_chunks + GROUP_SIZE - 1) // GROUP_SIZE

    def build_data(g, set_idx):
        g_start = g * GROUP_SIZE
        g_count = min(GROUP_SIZE, n_chunks - g_start)
        cur_set = sets[set_idx]
        data = [{"vec_idx": idx_addrs[g_start + c], "vec_val": val_addrs[g_start + c],
                 "vec_node": cur_set[c]["vec_node"], "vec_t1": cur_set[c]["vec_t1"],
                 "vec_t2": cur_set[c]["vec_t2"], "tree_addrs": cur_set[c]["tree_addrs"]} for c in range(g_count)]
        return data, g_count, g_start

    start_set = 0

    # Prologue
    data, g_count, g_start = build_data(0, 0)
    emit_slots(scheduler, build_tree_load_slots(data, g_count, 0, 0, 0, state))

    for round_idx in range(rounds):
        round_level = round_idx % (forest_height + 1)
        for g in range(n_groups):
            g_start = g * GROUP_SIZE
            g_count = min(GROUP_SIZE, n_chunks - g_start)
            cur_set_idx = (start_set + g) % 2
            next_set_idx = (start_set + g + 1) % 2

            cur_data = build_data(g, cur_set_idx)[0]
            hash_slots = build_hash_slots(cur_data, g_count, g, round_idx, state, forest_height)

            if round_idx == rounds - 1:
                hash_slots.extend(build_store_slots(cur_data, g_count, g_start, g, round_idx, state))

            next_prep_slots = []
            if g + 1 < n_groups:
                next_start = (g + 1) * GROUP_SIZE
                next_count = min(GROUP_SIZE, n_chunks - next_start)
                next_data = build_data(g + 1, next_set_idx)[0]
                next_level = round_level
                next_prep_slots = build_tree_load_slots(next_data, next_count, next_level, g + 1, round_idx, state)
            elif round_idx + 1 < rounds:
                next_data = build_data(0, next_set_idx)[0]
                next_count = min(GROUP_SIZE, n_chunks)
                next_level = (round_idx + 1) % (forest_height + 1)
                next_prep_slots = build_tree_load_slots(next_data, next_count, next_level, 0, round_idx + 1, state)

            emit_slots(scheduler, next_prep_slots)
            emit_slots(scheduler, hash_slots)

        start_set = (start_set + n_groups) % 2

    scheduler.add_ast("flow", ("pause",), note="epilogue")


def format_node(n):
    return f"[{n.order}] {n.engine.value}:{n.op} ({n.note})"


def analyze_and_compare():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_chunks = batch_size // VLEN

    # Build both versions
    print("Building stage-based...")
    sched_stage = TracingScheduler("stage-based")
    state_stage = build_common_init(sched_stage, n_chunks, forest_height)
    build_kernel_stage_based(sched_stage, forest_height, n_chunks, rounds, state_stage)
    metrics_stage = sched_stage.compute_metrics()

    print("Building round-based...")
    sched_round = TracingScheduler("round-based")
    state_round = build_common_init(sched_round, n_chunks, forest_height)
    build_kernel_round_based(sched_round, forest_height, n_chunks, rounds, state_round)
    metrics_round = sched_round.compute_metrics()

    lines = []
    lines.append("=" * 80)
    lines.append("DETAILED SLACK ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Stage-based: {len(sched_stage.ast_nodes)} nodes, critical_path={metrics_stage['critical_path_length']}")
    lines.append(f"Round-based: {len(sched_round.ast_nodes)} nodes, critical_path={metrics_round['critical_path_length']}")
    lines.append("")

    # Find sample nodes from g1_r5
    for prefix in ["g1_r5_tree_addr", "g1_r5_tree_load", "g1_r5_hash"]:
        lines.append("=" * 80)
        lines.append(f"ANALYSIS FOR: {prefix}")
        lines.append("=" * 80)

        nodes_stage = sched_stage.find_nodes_by_note_prefix(prefix)
        nodes_round = sched_round.find_nodes_by_note_prefix(prefix)

        if not nodes_stage or not nodes_round:
            lines.append(f"  No nodes found with prefix {prefix}")
            continue

        # Take first node as sample
        node_stage = nodes_stage[0]
        node_round = nodes_round[0]

        lines.append("")
        lines.append("STAGE-BASED:")
        lines.append(f"  Node: {format_node(node_stage)}")
        lines.append(f"  earliest_start: {metrics_stage['earliest_start'][node_stage]}")
        lines.append(f"  dist_to_pause:  {metrics_stage['dist_to_pause'][node_stage]}")
        lines.append(f"  slack:          {metrics_stage['slack'][node_stage]}")
        lines.append(f"  critical_path:  {metrics_stage['critical_path_length']}")
        lines.append(f"  num_deps:       {len(node_stage.deps)}")
        lines.append(f"  num_users:      {len(node_stage.users)}")

        lines.append("")
        lines.append("  Dependencies (what this node waits for):")
        for dep in sorted(node_stage.deps, key=lambda x: x.order):
            lines.append(f"    <- {format_node(dep)}")

        lines.append("")
        lines.append("  Direct users (what waits for this node):")
        for user in sorted(node_stage.users, key=lambda x: x.order)[:10]:
            lines.append(f"    -> {format_node(user)}")
        if len(node_stage.users) > 10:
            lines.append(f"    ... and {len(node_stage.users) - 10} more")

        lines.append("")
        lines.append("  Path TO sink (longest path to pause):")
        path_to_sink = sched_stage.trace_path_to_sink(node_stage, metrics_stage)
        for i, n in enumerate(path_to_sink[:15]):
            lines.append(f"    [{i}] {format_node(n)}")
        if len(path_to_sink) > 15:
            lines.append(f"    ... {len(path_to_sink) - 15} more steps to pause")

        lines.append("")
        lines.append("  Path FROM source (longest path from init):")
        path_from_source = sched_stage.trace_path_from_source(node_stage, metrics_stage)
        for i, n in enumerate(path_from_source[-15:]):
            lines.append(f"    [{len(path_from_source) - 15 + i}] {format_node(n)}")

        lines.append("")
        lines.append("-" * 40)
        lines.append("ROUND-BASED:")
        lines.append(f"  Node: {format_node(node_round)}")
        lines.append(f"  earliest_start: {metrics_round['earliest_start'][node_round]}")
        lines.append(f"  dist_to_pause:  {metrics_round['dist_to_pause'][node_round]}")
        lines.append(f"  slack:          {metrics_round['slack'][node_round]}")
        lines.append(f"  critical_path:  {metrics_round['critical_path_length']}")
        lines.append(f"  num_deps:       {len(node_round.deps)}")
        lines.append(f"  num_users:      {len(node_round.users)}")

        lines.append("")
        lines.append("  Dependencies (what this node waits for):")
        for dep in sorted(node_round.deps, key=lambda x: x.order):
            lines.append(f"    <- {format_node(dep)}")

        lines.append("")
        lines.append("  Direct users (what waits for this node):")
        for user in sorted(node_round.users, key=lambda x: x.order)[:10]:
            lines.append(f"    -> {format_node(user)}")
        if len(node_round.users) > 10:
            lines.append(f"    ... and {len(node_round.users) - 10} more")

        lines.append("")
        lines.append("  Path TO sink (longest path to pause):")
        path_to_sink = sched_round.trace_path_to_sink(node_round, metrics_round)
        for i, n in enumerate(path_to_sink[:15]):
            lines.append(f"    [{i}] {format_node(n)}")
        if len(path_to_sink) > 15:
            lines.append(f"    ... {len(path_to_sink) - 15} more steps to pause")

        lines.append("")
        lines.append("  Path FROM source (longest path from init):")
        path_from_source = sched_round.trace_path_from_source(node_round, metrics_round)
        for i, n in enumerate(path_from_source[-15:]):
            lines.append(f"    [{len(path_from_source) - 15 + i}] {format_node(n)}")

        lines.append("")

    # Compare critical paths
    lines.append("=" * 80)
    lines.append("CRITICAL PATH COMPARISON")
    lines.append("=" * 80)

    # Find the node with highest earliest_start (end of critical path)
    max_es_stage = max(sched_stage.ast_nodes, key=lambda n: metrics_stage['earliest_start'][n])
    max_es_round = max(sched_round.ast_nodes, key=lambda n: metrics_round['earliest_start'][n])

    lines.append("")
    lines.append("STAGE-BASED critical path end:")
    lines.append(f"  {format_node(max_es_stage)}")
    lines.append(f"  earliest_start = {metrics_stage['earliest_start'][max_es_stage]}")
    path = sched_stage.trace_path_from_source(max_es_stage, metrics_stage)
    lines.append(f"  Path length: {len(path)}")
    lines.append("  Last 20 nodes on critical path:")
    for i, n in enumerate(path[-20:]):
        lines.append(f"    [{len(path) - 20 + i}] {format_node(n)}")

    lines.append("")
    lines.append("ROUND-BASED critical path end:")
    lines.append(f"  {format_node(max_es_round)}")
    lines.append(f"  earliest_start = {metrics_round['earliest_start'][max_es_round]}")
    path = sched_round.trace_path_from_source(max_es_round, metrics_round)
    lines.append(f"  Path length: {len(path)}")
    lines.append("  Last 20 nodes on critical path:")
    for i, n in enumerate(path[-20:]):
        lines.append(f"    [{len(path) - 20 + i}] {format_node(n)}")

    # Write output
    output_file = "/home/aleng/original_performance_takehome_branch/slack_trace.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Output written to {output_file}")

    # Also print summary
    print("\nSUMMARY:")
    print(f"Stage-based critical path: {metrics_stage['critical_path_length']}")
    print(f"Round-based critical path: {metrics_round['critical_path_length']}")


if __name__ == "__main__":
    analyze_and_compare()
