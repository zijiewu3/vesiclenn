Refer to `setup.py` for how to set up entry points so that `crease_ga` can discover this plugin.

To access `cga_bbvesicle` from `crease_ga`:
1. Install `crease_ga` from source.
2. Install `cga_bbvesicle` from source.
3. The `scatterer_generator` class, mirroring the `scatterer_generator` in builtin `shape`s of `crease_ga`, can be accessed by:
   ```
   import crease_ga.plugins as plugins
   BBvesicle_sg = plugins['bbvesicle'].load()
   new_sg = BBvesicle_sg()
   ```
