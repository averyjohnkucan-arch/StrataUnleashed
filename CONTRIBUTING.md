# Contributing

Strata Unleashed builds on Strata 0.1.38. Keep user-facing setup simple and technical claims tied to measurements.

## Code map

| File | Responsibility |
| --- | --- |
| `install.sh`, `install.ps1`, `install_unleashed.py` | Download/check/extract a release and start setup |
| `tools/unleashed_wizard.py` | Model selection and the interactive setup flow |
| `tools/unleashed_catalog.py` | Pinned catalogs, complete shard grouping and header checks |
| `tools/unleashed_download.py` | Resume, integrity verification and download locking |
| `tools/unleashed_hardware.py` | Hardware scan, resource estimates and recommendations |
| `tools/unleashed_storage.py` | Drive identification and read-only speed sampling |
| `tools/unleashed_chat.py` | No-system-prompt terminal inference testing |
| `unleashed.py` | Model preparation, tuning cache and launch |
| `tools/unleashed_tune.py` | Measured search and final validation |
| `src/`, `include/`, `serve/` | Strata engine and local API |

Keep upstream attribution and licenses. Do not reformat unrelated vendored code. Do not publish models, credentials, local configs or benchmark transcripts accidentally; `work/`, `.venv/`, `models/` and release archives are ignored.

## Tests

After installing `requirements-unleashed.txt` in the local environment:

```bash
python -m unittest tools.test_unleashed_experience tools.test_unleashed_wizard tools.test_unleashed
```

Use the appropriate local Python on Windows. The tests cover model selection, memory estimates, download integrity, archive extraction, terminal prompt rules and storage diagnostics. Platform simulations are not a substitute for native Windows testing. Linux direct-I/O tests skip when the filesystem does not support them.

Engine changes need the relevant numerical checks and real-model validation described in [UNLEASHED.md](UNLEASHED.md). Do not repeat a long tuning sweep for a documentation-only change.

## Release discipline

The engine version stays in CMake. `UNLEASHED_VERSION` identifies the fork's packaging revision, such as `0.1.38-r2`. Preserve old release tags. Package the exact binary whose measurements are cited, record source/binary hashes, verify extracted launchers, and label Windows source/build artifacts accurately. Never claim every catalog model was inference-tested when only its header was checked.
