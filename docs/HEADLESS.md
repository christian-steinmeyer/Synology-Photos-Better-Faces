# Headless deployment — Synology Photos Better Faces

No companion UI, ports, reverse proxy, or login. Recognition uses the upstream Synopticon sync, detection, embeddings, and clustering engine. Only native Synology Photos People albums are used for reviewing newly discovered people.

## Configure and deploy

1. On your Disk Station, open Container Manager, create a new Project and choose a project folder on persistent storage, e.g. /volume1/docker/better-faces.
2. Copy the YAML from deploy/compose.yaml into the project YAML editor (or import the single Compose file).
3. Replace the NAS URL, usernames, passwords and account roster directly in the YAML. Remove or add account entries as needed. One account may specify both personal and shared; shared must occur only once.
4. Build/start the project and check its logs. The first start downloads the pinned SCRFD and ArcFace model weights automatically, then syncs and processes the library. This needs outbound HTTPS to GitHub, but inference and photos remain local.
5. The project folder needs only data/ and models/ persistent directories; the Compose file is the only configuration. No .env, TOML or accounts.json required.

All upstream settings can be overridden with SYNOPTICON_<SECTION>__<KEY> variables. Lists are JSON strings. SYNOPTICON_CONFIG=/dev/null selects an empty TOML source using upstream configuration support. The headless image does not include a .env file. Keep the live Compose file private; DSM/Docker administrators can inspect container environment variables. For passwords containing dollar signs, escape each dollar as two dollar signs in Compose. Use a DSM account with the appropriate Photos permissions and keep 2FA enabled; per-account otp_env can reference a temporary code during initial authorization if needed.

The account roster contains password environment variable names, not passwords. Each account has independent sessions, person IDs and database at data/<account-id>/synopticon.db. Models are shared in models/.

The initial pass uses SPBF_EXTRACT_LIMIT=0 (all pending photos). Set a positive value to bound each cycle. SPBF_DOWNLOAD_MODELS=false disables automatic bootstrap; in that case you must provision core models first. Optional YOLO, AdaFace and MagFace weights may be installed separately; the pipeline can run with core models alone. Check model-weight license terms before use.

## Resource sizing

The Compose file contains illustrative limits, not hardware requirements. Adjust
them to your Disk Station and other running services.

- **CPU:** `cpus` limits CPU time in logical-core equivalents. Keep capacity
  available for DSM and other containers. Adjust
  `SYNOPTICON_INFERENCE__INTRA_OP_THREADS` to match the CPU allocation.
- **RAM:** `mem_limit` caps container memory, not reserved memory. The required
  memory depends on model selection, photo sizes and library size (especially
  during clustering). Start conservatively, monitor actual usage in Container
  Manager, and raise the limit if needed, without starving DSM. Reducing
  `SYNOPTICON_INFERENCE__BATCH_SIZE` may reduce inference memory use.
- **Storage:** `SYNOPTICON_STORAGE__ORIGINALS_CACHE_GB` is the originals cache
  budget per account, not the total storage footprint. The persistent volumes
  also contain embeddings, face crops, metadata, and model weights. Allow
  additional space and back up the data volume.
- **Workload:** Set a positive `SPBF_EXTRACT_LIMIT` to bound extraction work
  per space per cycle; `0` processes all pending photos. It does not impose a
  limit on clustering memory or the size of the metadata synchronization.

## Native Who's this? integration

By default SPBF_WRITE_ENABLED=false. Enable BOTH SPBF_WRITE_ENABLED=true and SPBF_NATIVE_REVIEW_ENABLED=true only after validating on disposable photos for EACH account/space. The first face in an unnamed cluster is temporarily added to an existing staging person and separated with an empty name to create a native unnamed People group. Remaining faces are added directly to that new group. A pre-existing unnamed person is automatically selected as staging if available. Otherwise add staging_person_ids to the relevant SPBF_ACCOUNTS_JSON entry, e.g. `"staging_person_ids":{"personal":12345}`.

Writes are non-atomic, but a per-face journal records intentions and results, and subsequent runs reread Synology Photos before modifying anything. Interrupted operations can temporarily leave a face in the staging group until recovery. Every write is audited.

## Calibrated automatic assignment

Enable SPBF_AUTO_ASSIGN_ENABLED=true only alongside SPBF_CALIBRATION_APPROVED=true, after evaluating identity precision against your own labeled holdout. SPBF_MIN_SIMILARITY is an embedding similarity, NOT a calibrated probability. Existing named annotations are never automatically reassigned, merged or deleted.

Suspicious existing labels (reassignment/merge proposals) and weak detections (potential non-face hits) are exported as data/<account-id>/spbf-issues.json. Synology Photos does not expose a verified native “suspicious assignment” flag. These cases therefore cannot be represented accurately in Who's this? without removing an existing label. No automatic deletions are attempted.

## Limitations

- Synology Photos uses unofficial APIs; compatibility varies by DSM and Photos release. Validate against your Disk Station before enabling writes.
- Shared Space writebacks and persistence across reindex remain unverified.
- Identity suggestions cannot be displayed as candidate names in native Who's this?.
- First start downloads the pinned core model weights unless SPBF_DOWNLOAD_MODELS=false; optional models require separate provisioning.
- The first CPU-only scan of a large library can take considerable time; later scans are incremental.
- AGPL-3.0-or-later license inherited from Synopticon.

## Operational CLI (from Container Manager / Compose)

- `docker compose run --rm better-faces accounts`: list configured DSM principals without exposing secrets.
- `docker compose run --rm better-faces check`: verify NAS API access as every configured principal.
- `docker compose run --rm better-faces once`: sync, detect, cluster, and process queued native People operations.
- `docker compose run --rm better-faces preview`: dry-run queued native writes per account and space; no NAS changes.
- `docker compose run --rm better-faces evaluate`: run Synopticon's label holdout evaluation separately for each user's library.
- `docker compose up -d`: run the recurring headless scheduler.

The initial scan can run across the entire library with `SPBF_EXTRACT_LIMIT=0`.
This is resumable and CPU-limited but may require extended wall-clock time.
Set a positive limit to bound work per cycle.

Automatic assignment is always disabled until both the master write switch and calibration approval are enabled.

The container uses `docker/Dockerfile.headless`, which deliberately excludes
Synopticon's Vue frontend and FastAPI review dependencies. All configuration
is supplied directly through Compose environment variables.
