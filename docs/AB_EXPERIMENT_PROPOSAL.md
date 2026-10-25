# Prespecified Online A B Experiment

## Decision and variants

Randomize at the pseudonymous user level with a stable assignment held for the entire experiment. Control serves the current production recommender; if no production personalized system exists, the illustrative control is the frozen training-popularity policy. Treatment serves the frozen content-factor hybrid with alpha 0.50. The assignment unit is the user, not the page view, because cross-session exposure would otherwise contaminate outcomes.

## Primary outcome and assumptions

The sole primary outcome is the proportion of exposed users who complete at least one attributable purchase within seven days after their first eligible recommendation exposure. This outcome, its attribution window, and eligibility rules must be finalized before launch. For power planning only, assume a 10% control conversion rate and a two-percentage-point absolute minimum detectable effect (10% to 12%). With a two-sided 5% type-I error and 80% power, the standard independent-proportions approximation requires 3,841 users per arm. Inflating by 5% for logging loss or exclusions gives 8,086 exposed users total. These are illustrative assumptions, not estimates from the supplied rating data.

## Guardrails and execution

Guardrails are recommendation-service error rate, p95 response latency, opt-out or complaint rate, refund/cancellation rate, and gross margin per exposed user. Randomization balance is checked on pre-experiment activity but is not used to reassign users. Run for at least 14 consecutive days and two complete weekly business cycles, continuing longer if 8,086 eligible exposures have not accumulated. Stop only at the later of the prespecified duration and sample threshold, except for documented safety, privacy, or severe reliability thresholds. Do not repeatedly test the primary outcome or stop when a favorable p-value appears.

## Exposure and privacy logging

Log experiment ID, stable arm, pseudonymous user ID, request and exposure timestamps, eligible candidate set version, displayed ProductIDs and ranks, model/configuration hash, scores needed for debugging, device/context fields approved in advance, click/cart/purchase timestamps, order value, cancellation/refund status, and service latency/errors. Distinguish assignment, recommendation generation, actual rendering, and viewability; an assigned but unrendered list is not an exposure.

Apply purpose limitation, data minimization, access control, encryption, retention/deletion schedules, and a documented Thai PDPA review. Keep the re-identification key outside the analysis dataset. Do not log free text or protected attributes merely because storage is cheap. Publish the analysis population, exclusions, missing-exposure handling, and intent-to-treat result. Offline ranking accuracy from this lab is not causal evidence of revenue or conversion uplift; only randomized exposure can estimate that effect under the experiment's assumptions.
