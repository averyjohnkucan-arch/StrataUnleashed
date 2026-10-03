# Set up Strata Unleashed for a user

Use the fork's model-selection/download workflow, not the upstream installer's separate catalog.

1. Read [INSTALL.md](INSTALL.md). Run `cli-unleashed.sh --scan --json` (Windows: `CLI-UNLEASHED.bat --scan --json`) and explain hardware or build-tool constraints plainly.
2. Inspect `--list --json` and choose a model matching the user's preference and estimated resources. Prefer ISTA for compact/coding use, Atomic for larger models and HuiHui for uncensored use. Do not call an unsupported or merely header-checked model inference-validated.
3. The user can run the README's one-command installer, or download/extract the release and run the CLI. Keep files in the selected installation folder; existing model shards can be opened read-only with `--local-model`.
4. With an explicitly selected model, `--model-id ID --yes --chat` downloads, verifies, prepares, tunes and opens terminal test chat. `--tune-only` stops after configuration. `--download-only` stops after verified weights. Initial setup can take a while; keep the user informed.
5. Check the storage report. If it is not identified as NVMe, tell the user their experience may vary. Treat the bounded read-speed sample as a diagnostic, not a throughput prediction.
6. Test an answer, report what passed, and give the exact restart command. Terminal test chat has no system prompt and requires no API server.

Keep HTTP serving on localhost unless the user asks otherwise. External exposure requires authentication. Do not install or change drivers implicitly. Do not upload credentials, prompts or private model data. User instructions take precedence over these guidelines.

[Every CLI option](UNLEASHED-CLI.md) · [Troubleshooting](TROUBLESHOOTING.md)
