# Final QA Report

## Overall status

**CONDITIONAL.** The implementation, tests, executed notebook, two-page defense, statistical evidence, and staging archive are complete. Final submission remains conditional because the actual student ID is unavailable and four instructor resources named by the assignment were not supplied. No substitute files were fabricated.

## Phase checkpoints

| Phase | Status | Evidence |
|---|---|---|
| 00 | PARTIAL | Complete requirements/source audit; missing Week 12 deck, companion, reference notebook, and reference outputs documented |
| 01 | PASS | Strict ingestion, hashes, 121/38/41 split, 70-item catalog, teaching calculations, tests |
| 02 | PASS | Factor objective, 12-candidate validation search, exact one-event update, serialization |
| 03 | PASS | Six methods, shared candidate policy, score/rank traces, deterministic fallback and tie tests |
| 04 | PASS and CONFIG_FROZEN | Validation records and hash-verified frozen configuration before test evaluation |
| 05 | PASS | Untouched test results with per-user lists, denominators, coverage, diversity, cold-start and reference comparison |
| 06 | PASS | Leakage audit, expected negative test, cluster bootstrap, paired differences, resource audit |
| 07 | CONDITIONAL | Notebook and PDF pass; archive uses `STUDENTID_REQUIRED` until the user supplies the real ID |

## Verified results

- Data: 200 events, 87 users, 90 products; chronological partitions 121 train, 38 validation, and 41 test; 70 training-known products; no missing required values or duplicate user-item pairs.
- Matrix factorization: 8 factors, learning rate 0.2, lambda 0.01, 300 epochs, seed 202411; validation RMSE 1.7746769324 on 17 eligible rating events.
- Mathematical check: prediction 3.24, residual 0.76, updated user bias 0.2370, item bias -0.0615, user factors [0.2180, 0.4018], and item factors [0.5051, 0.1147].
- Validation: all alpha candidates tied at P@5 0.02, R@5 0.10, and NDCG@5 0.0430676558 across ten users; the predeclared balanced tie rule selected alpha 0.50.
- Test: nine ranking-eligible users, including two cold users; 18 rating-error-eligible events. Popularity, user CF, item CF, and factors have zero ranking metrics. Content and hybrid each have P@5 0.0222222 and R@5 0.1111111; NDCG@5 is 0.0701033 and 0.0478530.
- Coverage: popularity/user CF/item CF 5/70; content 29/70; factors 11/70; hybrid 29/70.
- CF: user and item CF fall back for all nine evaluable users because no supported neighborhoods exist under the frozen independent settings.
- Leakage: every training-fitted object ends at 2024-05-20 18:42:23 UTC; selection information ends at 2024-06-12 15:54:30 UTC; candidate/future-only overlap is empty. The isolated contaminated assertion fails as required.
- Bootstrap: 10,000 user-cluster replicates, seed 202406. Hybrid-minus-popularity paired intervals include zero for P@5, R@5, and NDCG@5, so superiority is unsupported.
- Notebook: 21 cells, nine code cells with execution counts 1-9, zero error outputs, and successful `nbformat` validation after a fresh-kernel execution.
- Defense: exactly two PDF pages; both pages rendered to PNG and were visually inspected with no clipping, overlap, broken table, or missing glyphs. Required methodological sections are present in extracted text.
- Tests: 26 tests passed, including final notebook, exact two-page PDF, and archive-integrity checks.

## Independently declared variants

The missing teaching references required explicit choices: logarithmic TF `1+ln(count)`, unsmoothed `ln(N/df)` IDF, absolute-rating-weighted content profiles, globally centered co-rated cosine, minimum overlap two, shrinkage ten, positive similarities only, the factor grid, and the alpha tie preference for 0.50. These choices were frozen before test evaluation.

## Reference comparison

The implementation matches the reference's nine users, all-zero popularity/CF/factor ranking metrics, no supported CF neighborhoods, and 5/70 popularity/CF coverage. It does not match the reported all-zero content/hybrid metrics or their coverages, and factor coverage is 11/70 rather than 12/70. The differences were preserved after the freeze and are likely attributable to unavailable companion conventions; that explanation is plausible, not verified.

## Unverified assumptions and deployment limits

The source does not establish randomized exposure, collection provenance, real catalog semantics, item release dates, or causal outcomes. Synthetic descriptions cannot establish production diversity. Offline relevance cannot be interpreted as incremental conversion, revenue, or margin. The A/B proposal therefore labels its 10% baseline, two-point MDE, and resulting 8,086-user target as illustrative.

## Submission and Git status

- Repository: `https://github.com/OpellieS/Hybrid-and-Validation.git`
- Pushed checkpoints: `077b51d` requirements/data audit; `53ad3f7` factor/hybrid models; `4484cde` frozen validation; `b34c1c6` test/statistical audit. The final Phase 07 commit and exact pushed HEAD are reported in the final handoff because a commit cannot contain its own hash.
- Staging archive: `submission/961731_STUDENTID_REQUIRED_Lab11C.zip`; SHA256 is stored in the adjacent `.sha256` sidecar after packaging.
- Archive excludes `.git`, virtual environments, caches, credentials, temporary renders, and the ZIP itself. CRC validation and required-entry checks are recorded in `artifacts/evidence/archive_integrity.json`.

## Remaining user action

Supply the actual student ID, rename the staging archive to `961731_<studentID>_Lab11C.zip`, and confirm whether the missing instructor resources can be obtained. If supplied later, exact-reference reproduction requires a new, clearly versioned experiment rather than overwriting this frozen record.
