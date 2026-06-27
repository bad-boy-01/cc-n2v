"""
lib/backends/ — Model-specific inference backends for CC-Novel2Video.

Each backend owns the complete lifecycle for one model family:
  load()     — download weights, configure pipeline, move to device
  generate() — run inference, return PIL.Image
  warmup()   — 64×64 smoke-test (verifies full stack, ~1-2 sec)
  unload()   — delete pipeline, gc.collect(), empty CUDA cache

Plugins in plugins/image/ are thin adapters that delegate here.
Adding a new image model = one new backend file + one tiny plugin stub.
"""
