# Autonomous Execution Log

## Phase 00 Requirements and Skill Audit

- Date: 2026-10-25 Asia/Bangkok
- Selected skills: `documents`, `pdf`, `karpathy-guidelines`, `scrutinize`
- Completed: cloned the authorized GitHub repository while preserving history and remote; recursively inventoried the course folder; hashed all supplied sources; read all assignment paragraphs and the complete requirements table; inspected the supplementary workflow and all four CSV schemas; created the requirements matrix, inventory, skill registry, implementation plan, open-issues register, and minimal project layout; copied immutable source evidence into the repository.
- Files created or modified: `README.md`, `.gitignore`, `requirements.txt`, `pyproject.toml`, `data/raw/*`, `references/*`, `docs/*`.
- Verification: assignment extraction reported 44 paragraphs, one 6x4 table, one section, and no inline shapes; all six present-source SHA256 values were recorded; repository remote resolves to the user-specified URL.
- Error recovery: initial DOCX text printing failed deterministically because the Windows console used `cp874` and could not encode U+2212; UTF-8 output fixed the issue and the complete extraction then succeeded.
- Outstanding issues: OI-01 through OI-06 in `docs/OPEN_ISSUES.md`.
- Checkpoint: **PHASE_00_PARTIAL**. The audit itself is complete, but exact instructor reproduction is impossible because the Week 12 file, companion script, reference notebook, and reference outputs are missing.
- Next phase: Phase 01 strict ingestion, split audit, and teaching foundations.

## Phase 01 Data Audit and Mathematical Teaching Foundations

- Date: 2026-10-25 Asia/Bangkok
- Selected skills: `karpathy-guidelines`; `caveman-commit` for the checkpoint message.
- Completed: strict string-preserving CSV ingestion; required-column, null, blank, integer-range, timestamp, and duplicate-pair rejection; UTC conversion; disjoint chronological split; source-hash manifest; training-known/future-only and warm/cold audit; transparent mask, observed-mean, log-TF, IDF, TF-IDF, L2, cosine, rating-profile, and centered-CF teaching calculations.
- Declared variant: unsmoothed `ln(N/df)` IDF, `1+ln(count)` TF, global-mean-centered overlap cosine, and absolute-rating-weighted normalized content profiles. These are independent choices because the Week 12 deck and companion are absent.
- Files created or modified: `src/lab11c/data.py`, `src/lab11c/teaching.py`, `src/lab11c/io_utils.py`, `scripts/run_phase01.py`, `tests/test_data.py`, `tests/test_teaching.py`, and `artifacts/evidence/*`.
- Test results: `pytest -q` -> 9 passed. Phase script verified 200 events, 87 users, 90 products, split sizes 121/38/41, 70 training products, and nine test ranking-eligible users.
- Error recovery: two initial tests failed because one expectation confused all validation-eligible users with warm-only users and one regex failed to escape brackets. Evidence showed implementation behavior was correct; tests were corrected and the full suite passed.
- Outstanding issues: instructor-specific TF-IDF and CF examples cannot be verified without OI-01/OI-02 resources.
- Checkpoint: **PHASE_01_PASS**.
- Next phase: Phase 02 matrix factorization.

## Phase 02 Matrix Factorization

- Date: 2026-10-25 Asia/Bangkok
- Selected skills: `karpathy-guidelines`, `scrutinize`.
- Completed: explicit rating predictor with training-only mean, biases, latent factors, deterministic initialization, full-batch gradient descent, L2 regularization, validation-only grid search, explicit unknown-ID fallbacks, JSON state serialization/reconstruction, and full search preservation.
- Mathematical convention: objective is half the mean observed squared error plus half lambda times the unaveraged squared parameter norm; each epoch uses simultaneous gradients from the old state. For one row, this exactly becomes the assignment's printed update.
- Files created or modified: `src/lab11c/matrix_factorization.py`, `tests/test_matrix_factorization.py`, `scripts/run_phase02.py`, `docs/MATRIX_FACTORIZATION.md`, `artifacts/evidence/mf_*`, and `artifacts/models/selected_mf_state.json`.
- Test results: full suite -> 12 passed. The mandatory trace produced prediction 3.24, residual 0.76, `bu=0.2370`, `bi=-0.0615`, `p=[0.2180,0.4018]`, and `q=[0.5051,0.1147]`. Serialized dimensions are 71x8 and 70x8; reconstructed predictions match.
- Validation selection: 12 predeclared candidates; selected factors 8, learning rate 0.2, regularization 0.01, 300 epochs, seed 202411; RMSE 1.7746769324 on 17 eligible validation events. Test scores were not used.
- Scrutinize trace: source row -> encoded training indices -> simultaneous old-state gradient -> saved arrays -> reconstructed ID mappings -> identical prediction. The simpler batch implementation was retained because it matches the stated mean-loss objective directly.
- Outstanding issues: selected grid is an independent variant because companion candidates are unavailable.
- Checkpoint: **PHASE_02_PASS**.
- Next phase: Phase 03 six recommendation methods and hybrid construction.

## Phase 03 Six Recommendation Methods and Hybrid Construction

- Date: 2026-10-25 Asia/Bangkok
- Selected skills: `karpathy-guidelines`; `caveman-commit` for the checkpoint message.
- Completed: popularity, TF-IDF content, user CF, item CF, matrix factorization, and content-factor hybrid; one shared training-known candidate universe; training-seen exclusion; popularity fallback; co-rated, centered, shrunk deterministic neighborhoods; per-request hybrid scaling; exact tie policy; full score and ranking traces for validation user 28.
- Independent settings: minimum co-rated overlap 2, shrinkage 10, maximum 20 neighbors, minimum one contributing neighbor, positive similarities only. These are frozen settings selected without held-out labels because the companion is absent.
- Files created or modified: `src/lab11c/recommenders.py`, `tests/test_recommenders.py`, `scripts/run_phase03.py`, `artifacts/evidence/recommender_configuration.json`, and `artifacts/results/phase03_*_trace.csv`.
- Test results: full suite -> 17 passed. Tests cover constant-component scaling, hybrid endpoints, candidate exclusion, maximum Top-5, uniqueness, deterministic ordering, and unknown-user fallback across all six methods.
- Checkpoint: **PHASE_03_PASS**.
- Next phase: Phase 04 validation, alpha selection, and configuration freeze.

## Phase 04 Validation Model Selection and Freeze

- Date: 2026-10-25 Asia/Bangkok
- Selected skills: `scrutinize`, `karpathy-guidelines`; `caveman-commit` for the checkpoint message.
- Completed: binary user-level P@5, R@5, and NDCG@5; explicit ranking/rating denominators; warm/cold segments; validation evaluation for all six methods; complete alpha grid; factor MAE/RMSE; coverage/diversity; immutable hash-verified experiment manifest.
- Alpha result: all five alpha values tied at validation Precision@5 0.02, Recall@5 0.10, and NDCG@5 0.0430676558 across ten eligible users. The predeclared tie rule selected alpha 0.50 as closest to a balanced blend. The reported reference alpha 0.75 was not forced.
- Freeze evidence: `artifacts/frozen_experiment_manifest.json` and `artifacts/evidence/CONFIG_FROZEN`; configuration hash `6e1576037b46550bf1f198590adb207f931cc1a21f206520f916254256bfa117`.
- Freeze contents: model and recommender configuration, alpha, seed, candidate and fallback policies, metric definitions, tie rules, all four CSV hashes, software versions, fitted-state checksums, fit cutoff, and validation-artifact hashes.
- Files created or modified: `src/lab11c/evaluation.py`, `src/lab11c/freeze.py`, `tests/test_evaluation_freeze.py`, `scripts/run_phase04.py`, validation evidence/results, and the frozen manifest.
- Test results: full suite -> 20 passed, including metric arithmetic and manifest tamper detection.
- Scrutinize trace: validation rows -> eligible positive sets -> shared ranking path -> per-user contributions -> aggregate alpha record -> declared tie rule -> manifest payload -> canonical hash -> reload verification. No final-test performance was evaluated before the gate.
- Outstanding issue: selected alpha differs from the reference because the available evidence produces a complete primary-metric tie under the declared independent implementation.
- Checkpoint: **PHASE_04_PASS** and **CONFIG_FROZEN**.
- Next phase: Phase 05 untouched test evaluation.

## Phase 05 Untouched Test Evaluation

- Date: 2026-10-25 Asia/Bangkok
- Selected skills: `scrutinize`.
- Completed: validated the frozen-manifest hash before loading test rows; loaded the frozen alpha 0.50 and recommender settings; evaluated all six methods without refitting; saved aggregate, per-user, and per-rank outputs; calculated coverage, content diversity, warm/cold segments, denominators, and MF rating error on 18 eligible events.
- Verified matches: nine evaluable users; popularity, user CF, item CF, and factor ranking metrics are all zero; both CF methods fall back for all nine users; popularity and CF coverage is exactly 5/70.
- Honest mismatches: content and hybrid each hit ProductID 552 for warm user 33, so their P@5 is 0.0222222, R@5 is 0.1111111, and NDCG@5 is 0.0701033 and 0.0478530 respectively rather than zero. Content/hybrid coverage is 29/70 rather than 19/70 and 22/70. Factor coverage is 11/70 rather than 12/70. These differences are consistent with unavailable companion conventions and were not used to change the experiment.
- Files created or modified: `scripts/run_phase05.py`, `artifacts/results/test_*`, and `artifacts/evidence/test_reference_comparison.json`.
- Test results: full suite -> 20 passed before test evaluation; frozen configuration hash remained `6e1576037b46550bf1f198590adb207f931cc1a21f206520f916254256bfa117`.
- Checkpoint: **PHASE_05_PASS**.
- Next phase: Phase 06 leakage, bootstrap, and resource audit.

## Phase 06 Leakage Bootstrap and Resource Audit

- Date: 2026-10-25 Asia/Bangkok
- Selected skills: `scrutinize`, `karpathy-guidelines`; `caveman-commit` for the checkpoint message.
- Completed: eight temporal provenance assertions plus candidate-universe assertion; isolated contaminated negative test; 10,000-replicate user-cluster bootstrap for every method and metric; paired hybrid-minus-popularity bootstrap using identical user draws; deterministic resource benchmark replay; environment and input-size capture.
- Leakage result: every training-fitted object ends at 2024-05-20 18:42:23 UTC, both selection decisions end at 2024-06-12 15:54:30 UTC, and the candidate set contains exactly 70 training items with zero future-only overlap. The contaminated copy at 2024-06-13 00:30:36 UTC failed as expected; the real experiment was unchanged.
- Uncertainty result: n=9 user clusters. Hybrid-minus-popularity paired differences are 0.0222222 P@5 [0, 0.0666667], 0.1111111 R@5 [0, 0.3333333], and 0.0478530 NDCG@5 [0, 0.1435589]. Every interval includes zero; the discrete intervals are not evidence of model superiority.
- Resource result: isolated frozen MF replay 0.0569 s; all-method evaluation 0.2278 s; sampled peak RSS about 135 MB. These measurements are environment-specific and the replay never replaced the frozen state.
- Files created or modified: `src/lab11c/audit.py`, `src/lab11c/bootstrap.py`, `tests/test_audit_bootstrap.py`, `scripts/run_phase06.py`, bootstrap evidence/results, `leakage_audit.json`, and `resource_audit.json`.
- Test results: full suite -> 23 passed; deliberately contaminated assertion failed inside both unit test and isolated audit path as required.
- Scrutinize trace: frozen hash -> fitted timestamps and candidate set -> isolated failure proof -> previously saved per-user test rows -> cluster resamples -> paired differences. Recommendation positions were never treated as independent units.
- Checkpoint: **PHASE_06_PASS**.
- Next phase: Phase 07 A/B experiment, notebook, methodological defense, archive, and final QA.
