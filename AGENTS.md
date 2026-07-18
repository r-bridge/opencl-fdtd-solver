# Agent instructions

## API documentation

The public Python API is documented in [`docs/API.md`](docs/API.md).

When changing anything users import from `opencl_fdtd_solver` (see `__all__` in `opencl_fdtd_solver/__init__.py`), or the public methods/attributes of those types:

1. **Update `docs/API.md` in the same change** so names, signatures, defaults, and behavior match the code.
2. **Do not leave stale claims** (removed parameters, wrong dtypes, deprecated APIs presented as primary, etc.).
3. Prefer documenting the **public** surface only; internal modules (`cpml`, `kernels`, …) stay out unless they become part of `__all__`.
4. Run `python -m unittest tests.test_api_docs -v` — CI enforces that every `__all__` export and each public method of the documented classes appears in `docs/API.md`.

Physics narrative lives in `docs/PHYSICS.md`; keep API facts in `API.md`, not duplicated inconsistently across both.
