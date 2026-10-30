# 6B preparation restart: 2026-10-20

## Status

- Old job `4262916` failed on 2026-10-13 before any new token shard was
  committed. The failing operation was the Hugging Face dataset API request
  on a CPU compute node (`Network is unreachable`).
- A new batch network check, `4287616`, confirmed that compute-node access
  still fails. The official `hpc2021-io1.hku.hk` transfer node can access the
  pinned dataset. This follows the distinction between compute and IO nodes
  in the [HKU user guide](https://hpc.hku.hk/hpc/hpc2021/userguide/).
- New CPU job **4287721** started at **2026-10-20 11:10 HKT** on `GPI-2-1`,
  with **8 CPUs, 32 GiB RAM and zero GPUs**.
- The IO transfer service runs on `hpc2021-io1.hku.hk`, PID `2446870`,
  with a seven-day maximum lifetime and an exclusive process lock.
- At **11:18:12 HKT**, 14 extra indexed shards contained **317,720,501**
  newly committed tokens. The preserved base contains **3,000,012,394**
  tokens. Total available base plus new shards: **3,317,732,895** tokens.
  The extra-data target is approximately 3B tokens, so the extension was
  about **10.6%** complete, not 55.3% of a fresh download.
- Remaining filesystem space was **57.14 GiB**. `PREPARED` did not yet
  exist: the 6B dataset was **not yet ready for training**.

These are timestamped observations, not a live status page. The raw snapshot
is `results/manifold/status_20261020/data6b_status.json`.

## Transport-only change

The original sampler, extension program, Megatron index builder, tokenizer,
source revision, source-file ordering, RNG seed, mixture quotas, and exact-token
deduplication rules remain frozen. The old SQLite exclusion records were reused:
2,234,668 base training documents plus the original validation exclusions.

`scripts/data/staged_olmo_transport.py` replaces only the sampler's two
transport callbacks. On the IO node it downloads pinned public compressed
source files to shared storage, one file at a time. On the CPU node it reads
the same JSON records locally, then delegates to the original sampler and
index builder. It does not tunnel compute traffic or run tokenization on the
login node.

Each staged file is checked by size and SHA256 before consumption. The next
source request removes only the previous raw file owned by this service.
An 8-GiB reserve is enforced for downloads and index shards. A separate
`transport_provenance.json` records the adapter and metadata hashes without
rewriting the frozen SQLite data identity.

Local tests cover pinned source filtering, path validation and equivalent
plain/gzip JSON ordering. The remote run has additionally consumed real Zstandard
files, resumed exclusion records, and committed new indexed shards successfully.

## Locations

- Frozen preparation bundle:
  `/home/u3013198/projects/olmo6b-data-prep-20261013`
- Transport adapter and new batch script:
  `/home/u3013198/projects/olmo6b-transport-20261020`
- CPU log:
  `logs/prepare_olmo6b_staged_4287721.out` within the transport directory.
- Transfer log:
  `logs/io_transfer.log` within the transport directory.
- Output:
  `/home/u3013198/projects/SSO_test/data/olmo_mix_1124_6b`
- Frozen Megatron source:
  `/home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005/Megatron-LM`

## Completion and recovery

The original program merges the preserved 3B base and new indexed shards,
checks counts/hashes, and writes `PREPARED` only on successful completion.
The adapter then writes the staging `DONE` marker so the IO service exits.
Do not launch 6B training until the original dataset verification succeeds.

If interrupted, first inspect Slurm state and both logs. Resume only after
ensuring the previous consumer and IO service have stopped; both have exclusive
locks. Reuse the same pinned bundle, tokenizer, manifest, output directory and
adapter. Do not delete `state.sqlite`, the base/validation datasets or tokenizer.
The sampler resumes committed shards and source cursors, not an uncommitted
partial token shard. A failed raw download may leave a `.part` file; inspect
`error.json` before retrying and archive/remove that stale error only after
the underlying transfer problem is resolved.

No new H20 training jobs were submitted as part of this restart.
