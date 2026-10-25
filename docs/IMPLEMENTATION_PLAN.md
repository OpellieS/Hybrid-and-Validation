# Implementation Plan

1. Lock raw hashes and strict ingestion; generate one immutable chronological split manifest and teaching-calculation fixtures.
2. Implement a small explicit-feedback bias-plus-factor model whose objective and stochastic update use one declared scaling convention; select only on validation.
3. Implement popularity, content, user CF, item CF, factor, and hybrid scoring behind one deterministic ranking function and trace schema.
4. Evaluate validation candidates, select factor settings and alpha without consulting test outcomes, then write and validate a frozen manifest containing state checksums.
5. Require the frozen manifest before loading test rows; evaluate all six methods and write event-, user-, and aggregate-level evidence.
6. Run fitted-object leakage assertions, an isolated contaminated negative test, paired user-cluster bootstrap, and resource measurements.
7. Build a fresh-kernel executed notebook, exactly two-page methodological-defense PDF, QA report, and staging ZIP because no student ID was supplied.

Success is determined by assertions and reproducible artifacts, not by making the metrics look attractive. Given the assignment's reported zero test ranking metrics, a nonzero result is a discrepancy to investigate, not a target to celebrate or suppress.
