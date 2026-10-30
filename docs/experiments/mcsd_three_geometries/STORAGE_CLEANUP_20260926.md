# H20 storage cleanup, 2026-09-26

The user requested removal of unnecessary old caches, especially 12B and
width-1024 artifacts. On 2026-09-26 only regenerable sampling caches were
removed. On 2026-10-01 the user explicitly approved removal of the eleven
listed old checkpoint directories, and that deletion is now complete.

## Completed follow-up, 2026-10-01

- Removed the four old width-1024 checkpoint directories and seven old smoke
  checkpoint directories: 40,966,318,341 bytes (38.15 GiB).
- Removed 124 additional, regenerable sample-index cache directories from
  the inactive old project: 2,058,196,908 bytes (1.92 GiB).
- Total logical size removed: 43,024,515,249 bytes (40.07 GiB).
- All 1B/3B corpora, tokenizer, logs/configs/results, old width-256 weights,
  and all six new width-384 3B final checkpoints remain preserved.
- No user jobs were active during deletion. All seven smoke jobs were
  rechecked as completed; preserved evidence hashes passed verification.
- The immediate NFS accounting still showed 11.45 GiB free; a later fresh
  filesystem query showed **52 GiB available, 49% used**, before new job setup.
- Newly submitted width-256 LR runs disable checkpoints for this first search.

Remote audit:
`/home/u3013198/projects/SSO_test-three-geometries/storage_cleanup/20261001T041420Z/manifest.json`.
Local copy: `results/manifold/storage_cleanup_20261001/manifest.json`.
The logs and compact evidence archive are not backups of the deleted weights.

## Completed action

- Deleted 23 old sampling-cache directories, totaling 1,009,371,712 logical
  bytes (0.940 GiB). This includes old-project `data_cache` and width-1024
  `results/.../data_cache` directories; not the corpus itself.
- Old 12B result directories were already nearly empty (about 1.5 KiB total
  in the two discovered result roots). They are not the present space problem.
- Kept the 1B and 3B indexed corpora and tokenizer, all logs/configurations,
  old width-256 3B weights, and every new 3B checkpoint and runtime cache.
- Checked that the only active jobs were Stiefel 4106583/4106585/4106587.
  Their logs continued advancing normally after cleanup, with no logged NaN
  or skipped updates at the latest checked intervals.
- Preserved evidence checksums were verified. Active training logs may grow
  and were excluded from the immutable-checksum comparison.

Remote audit:
`/home/u3013198/projects/SSO_test-three-geometries/storage_cleanup/20260926T020835Z/manifest.json`.
Local copy: `results/manifold/storage_cleanup_20260926/manifest.json` (ignored
runtime artifact). It enumerates every deleted path and its original files.
The immediate filesystem snapshot reported 10.526 GiB available; NFS free
space accounting did not change during that process, so logical bytes deleted
must not be presented as a measured immediate increase in free space.
The subsequent fresh `df -B1` query reported 12,298,747,904 bytes available
(approximately 11.45 GiB / 12.30 GB), with 89% filesystem usage.

## September 26 inventory (deletion completed October 1)

| Artifacts | Approximate size | Consequence of deletion |
| --- | ---: | --- |
| Four completed width-1024 3B final checkpoints | 24.95 GiB | No direct resume or checkpoint-based reevaluation |
| Seven completed short-test checkpoint directories | 13.21 GiB | Must rerun smoke to reproduce its saved weights |

These paths were initially retained pending explicit confirmation. The user
provided that confirmation on October 1, and the follow-up above records the
completed deletion. Original logs and the compact evidence archive remain.

## Old width-1024 results preserved before any weight cleanup

These are the old spectral-constraint 587M-model, 3B-token experiments, not the
new width-384 Frobenius/Stiefel study. All finished at step 5723 with exit 0.

| Method | Job | Final validation loss | Slurm elapsed |
| --- | ---: | ---: | --- |
| MuonSphere | 3893055 | 2.959356 | 2d 16:08:22 |
| SSO | 3893054 | 2.974528 | 2d 23:03:10 |
| Cached exact-SVD MCSD-TP | 3893056 | 2.969867 | 6d 00:16:15 |
| Cached exact-SVD MCSD-TP-PGD, gap=3e-5, scale=0.10 | 3893057 | 2.965966 | 6d 02:26:21 |

Source: old-project `logs/*389305[4567].out` final validation-set endpoint,
Slurm accounting, and `results/width1024_cached_exact_svd_3b/submissions.tsv`.
The old live-progress document predates the two MCSD endpoints; use these
final values rather than its historical pending entries.
