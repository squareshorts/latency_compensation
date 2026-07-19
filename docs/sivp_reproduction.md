# Strengthening-analysis reproduction

From `C:\work\auto`, with the selective AV2 payload and detector checkpoints available locally, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce_sivp_strengthening.ps1
```

The scripts are resumable at detector/log Parquet or JSON boundaries. Raw AV2 files live under the ignored `data/av2` tree; model weights are ignored by `*.pt`. Aggregate outputs are written to `results/sivp_strengthening`.

The reproduction order preserves information access: extension-log selection and gate freezing precede extension propagation and gate evaluation. The original held-out gate diagnosis is never relabeled prospective.
