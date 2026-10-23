# harness

The Python package is `harness`, at the repository root beside `pyproject.toml`. The repository directory name is not the package name.

Keep that flat layout. `module-root = ""` under `[tool.uv.build-backend]` is load-bearing: the build reads `harness/`, not `src/harness/`.

Use uv to install and to run the `harness` console script.


# Code Standards

- Religiously follow clean code principles. But keep it simple in terms of architecture as this is only a workshop
- Wherever docstrings are used they must be bulleted or numbered list
- Every file opens with a module docstring naming the concept the file teaches, and the design rules it embodies. 
- Every function public or private carries a docstring . Where behavior is subtle — the thought-signature echo, the edit uniqueness rule, the torn-line tolerance — a short comment states the constraint the code serves.
- Every function must have type definition and type hints. Complex type must be defined in harnes.schema.{module_name}.py
- Keep .env.example file update to date with actual env vars that are being used. Use dotenv to load env vars