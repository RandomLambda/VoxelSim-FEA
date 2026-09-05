# GPU-edition-only wheels

Drop CuPy wheel(s) here to have them bundled into the GitHub-only GPU
edition. `build_extension.py` copies every `*.whl` in this folder into the
staged `wheels/` directory and appends it to `blender_manifest.toml`'s
`wheels = [...]` list, but **only when building the GPU edition** -- the
standard/hosted edition never sees this folder and stays CuPy-free.

Example (pick the CUDA toolkit(s) you want to support):

    cupy_cuda12x-13.x.x-cp311-cp311-manylinux2014_x86_64.whl
    cupy_cuda12x-13.x.x-cp311-cp311-win_amd64.whl

Get these from https://pypi.org/project/cupy-cuda12x/#files (or the matching
`cupy-cuda11x` package for older toolkits) -- download as-is, don't rename.

Note: CuPy wheels are large (tens to 100+ MB each). Since the GPU zip is
distributed as a GitHub Release asset rather than committed to the repo,
that's fine for the release upload, but don't commit the raw `.whl` files
here to git -- keep them out of version control (e.g. via `.gitignore`) and
only drop them in locally right before running `build_extension.py`.
