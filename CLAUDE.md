# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Anthropic's performance take-home challenge - a kernel optimization task for a custom VLIW SIMD architecture simulator. The goal is to minimize cycle count while maintaining correctness.

**Best known AI performance:** ~1363 cycles. Best human performance is better but undisclosed.

## Commands

```bash
# Run submission tests (use this for verification)
python tests/submission_tests.py

# Run all tests locally
python perf_takehome.py

# Run a specific test
python perf_takehome.py Tests.test_kernel_cycles

# Run with performance trace (generates trace.json)
python perf_takehome.py Tests.test_kernel_trace

# Hot-reloading trace visualization (opens Perfetto UI)
python watch_trace.py

# Validate no test modifications before submission
git diff origin/main tests/
```

## Critical Constraints

**DO NOT modify anything in the `tests/` folder.** The test folder contains `frozen_problem.py` which is used for submission validation. Modifying tests to improve scores is cheating and will invalidate your submission.

## Architecture

### Files
- `perf_takehome.py` - Main file with `KernelBuilder` class to implement your kernel
- `problem.py` - Machine simulator, data structures (Tree, Input, Machine, Core), reference implementations
- `watch_trace.py` - Debug server for visualizing performance traces

### Machine Simulator
The simulator models a VLIW SIMD architecture with these engine types and slot limits:
- **alu** (12 slots): Arithmetic/logic (+, -, *, //, ^, &, |, <<, >>, %, <, ==)
- **valu** (6 slots): Vector operations (vbroadcast, multiply_add, vector arithmetic)
- **load** (2 slots): Memory/constant loading
- **store** (2 slots): Memory stores
- **flow** (1 slot): Control flow (select, jump, cond_jump, halt)
- **debug** (64 slots): Debug traces (ignored in submission)

Key constants: `VLEN = 8` (vector length), `SCRATCH_SIZE = 1536` (register space in 32-bit words)

### Instruction Format
Instructions are dicts mapping engine types to instruction slot lists:
```python
{"alu": [("*", dest, a, b)], "load": [("load", dest, addr)]}
```
- First operand is typically the destination
- All numbers are scratch addresses except for `const`, `jump`, and special flow instructions
- Instructions within a cycle execute in parallel across engines

### KernelBuilder Pattern
- `alloc_scratch(name, length)` - Allocate scratch space for variables
- `scratch_const(val, name)` - Load constants with deduplication
- Build instruction lists as tuples and convert to instruction dicts

## Performance Baseline

Starting baseline: 147734 cycles. Impressive threshold: <1487 cycles.
