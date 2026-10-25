# Requirements to Deliverables Matrix

| Requirement | Source authority | Planned evidence | Status after Phase 00 |
|---|---|---|---|
| Strict four-column ingestion; string IDs; integer ratings 1-5; no nulls or duplicate user-item pairs; UTC timestamps | Assignment schema table | `src/lab11c/data.py`, audit JSON, tests | Audited, implementation pending |
| Fixed split: train before 2024-05-21 UTC, validation before 2024-06-13 UTC, then test; 121/38/41 events | Assignment schema table and prompt | immutable split manifest and assertions | Audited, implementation pending |
| Problem 1.1 prediction, observed-entry objective, dimensions, regularization, training-only fitted quantities | Assignment P1.1 | model source, notebook derivation, tests | Pending |
| Problem 1.2 fixed candidates and validation-only model selection with complete record | Assignment P1.2 | search CSV/JSON, notebook | Pending; companion candidates unavailable |
| Problem 1.3 serialized-state reconstruction, printed one-event update, finite/dimension checks, unknown-ID fallback | Assignment P1.3 | model artifact, unit tests, trace JSON | Pending |
| Six recommenders: popularity, content, user CF, item CF, factor, hybrid | Assignment P2 | source modules, rankings, integration tests | Pending |
| Content TF-IDF, rating-weighted profile, cosine, zero-profile rule | Assignment and teaching prompt | feature artifact, score trace, tests | Pending; independent formula variant must be declared |
| CF observed masks, co-rated overlap, centered similarity, shrinkage, deterministic support/fallback | Prompt and assignment P2 | score traces and support audit | Pending; exact companion settings unavailable |
| Hybrid per-request min-max, constant-to-zero, alpha blend, endpoints, label-free scaling | Assignment P2.1 | alpha search and score trace | Pending |
| Candidate universe = 70 training-known items; exclude train-seen items; deterministic tie order | Assignment P2.2 | rank trace and assertions | Pending |
| Validation metrics P@5/R@5/NDCG@5; positives rating >=4; separate ranking/rating denominators and cold/future-only segments | Assignment P2.3 and prompt | validation tables and tests | Pending |
| Freeze configuration before test, including hashes, versions, policies, checksums | Prompt freeze gate | `artifacts/frozen_experiment_manifest.json` | Pending |
| Untouched test of all methods; rating error where meaningful; coverage; diversity; cold-start breakdown | Assignment P2.3/P3.2 | result CSV/JSON and notebook | Pending |
| Compare against stated nine evaluable users, zero ranking metrics, no CF support, and coverage values | Assignment reference paragraph | comparison report | Pending; comparison is not tuning authority |
| Leakage audit of every fitted object and isolated deliberately failing future-data assertion | Assignment P3.1 | audit JSON and negative unit test | Pending |
| User-cluster bootstrap and paired differences; acknowledge n=9 limitation | Assignment P3.2 | bootstrap samples/summary | Pending |
| Runtime, peak memory, input size, environment | Assignment P3.2 | resource audit JSON | Pending |
| User-randomized A/B design with one primary outcome, guardrails, MDE, power, duration, fixed stop, exposure and privacy | Assignment P3.3 | notebook and defense | Pending |
| Executed notebook Parts 1.1-3.3, two-page defense, evidence folder, reproducible ZIP | Submission protocol | notebook, PDF, ZIP, QA report | Pending; student ID unavailable |

## Acceptance criteria

All automated tests pass; the mathematical update equals the printed values within numerical tolerance; dataset hashes match; no test metric is computed before the frozen manifest exists and validates; a fresh-kernel notebook execution succeeds; the defense PDF is exactly two pages and passes text and visual inspection; and the archive can be listed and its checksums verified.
