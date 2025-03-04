import os

from moe_gen.op_builder.core_engine import CoreEngineBuilder

use_jit = False
try:
    from moe_gen import core_engine
except ImportError:
    print("Do not detect pre-installed core engine, use JIT mode.")
    use_jit = True

# print("run core engine importer")

core_engine = CoreEngineBuilder().load() if use_jit else core_engine
