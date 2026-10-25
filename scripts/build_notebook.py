from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parents[1]
OUT = ROOT / "notebooks" / "Lab11C_Hybrid_and_Validation.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        md(
            """
# Lab 11C Hybrid Recommendation and Validation

**Course 961731 Business Data Analytics**

This notebook is the executable record for Problems 1.1-3.3. It runs the phase scripts in temporal order, freezes configuration before opening final-test performance, and distinguishes verified outputs from independent variants. The available data supports a teaching evaluation, not a causal claim about business uplift.
"""
        ),
        code(
            """
from pathlib import Path
import json, os, subprocess, sys, inspect
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / 'src' / 'lab11c').exists():
    ROOT = Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
pd.set_option('display.max_columns', 30)
print('Repository:', ROOT)
"""
        ),
        md(
            """
## Reproduce Phases 01 through 06

Each script writes compact machine-readable evidence. Phase 04 creates and validates the freeze hash; Phase 05 refuses to proceed without it. Phase 06 replays the frozen training configuration only for resource measurement and never replaces the selected state.
"""
        ),
        code(
            """
phase_outputs = []
env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
for phase in range(1, 7):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / f'run_phase{phase:02d}.py')],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    line = result.stdout.strip().splitlines()[-1]
    phase_outputs.append(line)
display(pd.DataFrame({'Phase': range(1, 7), 'Result': phase_outputs}))
"""
        ),
        md(
            """
## Data audit and teaching foundations

Identifiers remain strings, ratings must be integers from 1 to 5, and Unix seconds are parsed in UTC. The chronological masks are train before 21 May 2024, validation from that boundary to before 13 June, and test from 13 June onward. Learned transformations use training data only. Missing ratings remain unknown.

Because the Week 12 deck and companion script were absent, the declared content variant uses lowercase whitespace tokens, `1 + ln(count)` term frequency, unsmoothed `ln(N/df)` IDF, L2 document normalization, and an absolute-rating-weighted L2 user profile. CF uses globally centered overlap cosine, minimum overlap 2, shrinkage 10, positive similarities, and deterministic neighbors.
"""
        ),
        code(
            """
audit = json.loads((ROOT/'artifacts/evidence/data_audit.json').read_text())
display(pd.DataFrame({
    'Quantity': ['Rows','Users','Products','Train','Validation','Test','Training catalog','Test ranking users'],
    'Verified value': [audit['rows'],audit['unique_users'],audit['unique_products'],audit['train']['events'],audit['validation']['events'],audit['test']['events'],audit['train']['items'],audit['test']['ranking_eligible_users']]
}))
teaching = json.loads((ROOT/'artifacts/evidence/teaching_examples.json').read_text())
display(pd.Series({
    'Observed tiny ratings': teaching['observed_count'],
    'Global observed mean': teaching['global_observed_mean'],
    'Cosine A-B': teaching['cosine_A_B'],
    'Cosine A-C': teaching['cosine_A_C'],
    'CF U0-U1 overlap': teaching['cf_U0_U1_overlap'],
    'CF U0-U1 similarity': teaching['cf_U0_U1_similarity'],
}))
"""
        ),
        md(
            r"""
## Problem 1.1 Objective and dimensions

For observed training pairs only,

$$\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i.$$

The optimized objective is

$$J=\frac{1}{2|\Omega|}\sum_{(u,i)\in\Omega}(r_{ui}-\hat r_{ui})^2+\frac{\lambda}{2}\left(\|b_u\|_2^2+\|b_i\|_2^2+\|P\|_F^2+\|Q\|_F^2\right).$$

`P` is `U x d`, `Q` is `I x d`, and the global mean and every parameter are fitted on training only. Full-batch simultaneous gradients keep the objective scaling explicit. Unknown IDs receive only terms that truly exist: both unknown gives the training mean; a known user or known item adds only its fitted bias.
"""
        ),
        code(
            """
from lab11c.matrix_factorization import ExplicitMatrixFactorization
print(inspect.getsource(ExplicitMatrixFactorization.one_event_update))
update = ExplicitMatrixFactorization.one_event_update(
    4.0, 3.0, 0.2, -0.1, np.array([0.2,0.4]), np.array([0.5,0.1]), 0.05, 0.1
)
display(pd.Series({k: v for k,v in update.items()}))
np.testing.assert_allclose(
    [update['prediction'], update['residual'], update['user_bias'], update['item_bias']],
    [3.24, 0.76, 0.2370, -0.0615]
)
np.testing.assert_allclose(update['user_factors'], [0.2180,0.4018])
np.testing.assert_allclose(update['item_factors'], [0.5051,0.1147])
"""
        ),
        md(
            """
## Problems 1.2 and 1.3 Fit select and verify

The missing companion required an independent grid declared before scoring. Selection minimizes validation RMSE on warm-user, training-known-item rating events. The complete record is retained; final-test results never select the model.
"""
        ),
        code(
            """
search = pd.read_csv(ROOT/'artifacts/evidence/mf_validation_search.csv')
display(search[['factors','learning_rate','regularization','epochs','seed','validation_mae','validation_rmse','selected']])
state = json.loads((ROOT/'artifacts/models/selected_mf_state.json').read_text())
model = ExplicitMatrixFactorization.from_dict(state)
from lab11c.data import load_interactions, temporal_split
split = temporal_split(load_interactions(ROOT/'data/raw/interaction_events-2.csv'))
row = split.train.iloc[0]
score = model.predict_one(row.UserID,row.ProductID)
rebuilt = ExplicitMatrixFactorization.from_dict(model.to_dict()).predict_one(row.UserID,row.ProductID)
display(pd.Series({'saved_state_score': score, 'reconstructed_score': rebuilt, 'finite': np.isfinite(score), 'P_shape': model.user_factors.shape, 'Q_shape': model.item_factors.shape}))
assert score == rebuilt
"""
        ),
        md(
            """
## Problems 2.1 and 2.2 Blend and rank

Six methods share one 70-item training-known universe: popularity, content, user CF, item CF, matrix factorization, and the content-factor hybrid. Training-seen items are removed. Unknown users use popularity. The hybrid maps each component across that request's eligible candidates to [0,1], maps a constant component to zero, and computes `alpha * content + (1-alpha) * factor`. No held-out labels enter scaling. Ordering is score descending, training popularity descending, then lexical ProductID ascending.
"""
        ),
        code(
            """
alpha_search = pd.read_csv(ROOT/'artifacts/results/validation_alpha_search.csv')
display(alpha_search[['alpha','eligible_users','precision_at_5','recall_at_5','ndcg_at_5','catalog_coverage','selected']])
freeze = json.loads((ROOT/'artifacts/frozen_experiment_manifest.json').read_text())
display(pd.Series({'status': freeze['status'], 'selected_alpha': freeze['selected_alpha'], 'configuration_hash': freeze['configuration_hash'], 'fit_max_time': freeze['fitted_max_event_time']}))
trace = pd.read_csv(ROOT/'artifacts/results/phase03_ranking_trace.csv')
display(trace[['Method','Rank','ProductID','Score','TrainingPopularity']])
"""
        ),
        md(
            """
## Problem 2.3 Untouched test comparison

Ranking eligibility requires at least one held-out positive (`Rating >= 4`) whose item is training-known. Cold users remain evaluable through the declared popularity fallback. Rating MAE/RMSE is separate and uses only warm-user, training-known-item events. The freeze manifest is validated before test rows are loaded.
"""
        ),
        code(
            """
test_metrics = pd.read_csv(ROOT/'artifacts/results/test_method_metrics.csv')
display(test_metrics[['method','eligible_users','precision_at_5','recall_at_5','ndcg_at_5','catalog_coverage','mean_intra_list_diversity','fallback_users','rating_events','rating_mae','rating_rmse']])
comparison = json.loads((ROOT/'artifacts/evidence/test_reference_comparison.json').read_text())
display(pd.DataFrame(comparison['comparisons'])[['method','all_ranking_metrics_zero_match','coverage_observed','coverage_reference','coverage_match','no_supported_cf_neighborhood_match']])
"""
        ),
        md(
            """
**Interpretation.** Nine users are evaluable. Popularity, both CF methods, and factors have zero P@5, R@5, and NDCG@5. The independent content and hybrid variants each recover one positive for user 33, so they do not reproduce the reported all-zero result. Popularity/CF coverage matches 5/70; factor coverage is 11/70 instead of 12/70; content and hybrid each cover 29/70 instead of 19/70 and 22/70. These mismatches are reported, not tuned away.
"""
        ),
        md(
            """
## Problem 3.1 Leakage and reproducibility audit

Each training-fitted object must end strictly before 21 May; validation-selected hyperparameters and alpha must end strictly before 13 June. The deliberately contaminated copy is expected to fail. It is isolated and never replaces real state.
"""
        ),
        code(
            """
leakage = json.loads((ROOT/'artifacts/evidence/leakage_audit.json').read_text())
display(pd.DataFrame(leakage['fitted_object_time_assertions']))
display(pd.Series({'candidate_items': leakage['candidate_assertion']['candidate_item_count'], 'future_only_overlap': leakage['candidate_assertion']['future_only_overlap'], 'negative_test_failed_as_expected': leakage['negative_test']['actual_failure'], 'negative_test_message': leakage['negative_test']['message']}))
"""
        ),
        md(
            """
## Problem 3.2 User cluster bootstrap and resources

Users, not recommendation slots, are resampled. Paired comparisons draw the same users for both methods by bootstrapping each user's metric difference. With only nine clusters, percentile intervals are discrete and unstable; an interval that includes zero cannot support a superiority claim.
"""
        ),
        code(
            """
bootstrap = pd.read_csv(ROOT/'artifacts/results/bootstrap_summary.csv')
display(bootstrap[(bootstrap['Type']=='paired_difference') | ((bootstrap['Type']=='model') & (bootstrap['Metric']=='NDCGAt5'))][['Type','MethodA','MethodB','Metric','EligibleUsers','PointEstimate','CI2.5','CI97.5']])
resources = json.loads((ROOT/'artifacts/evidence/resource_audit.json').read_text())
display(pd.DataFrame([
    {'stage':'frozen training replay', **resources['training_measurement']},
    {'stage':'six-method evaluation', **resources['evaluation_measurement']},
])[['stage','wall_seconds','peak_process_rss_bytes','peak_python_traced_bytes','scope']])
"""
        ),
        md(
            """
## Problem 3.3 Prespecified online experiment

Randomize pseudonymous users 1:1 with stable assignment. Control is the current production policy; if absent, use frozen popularity. Treatment is the frozen alpha-0.50 hybrid. The **single primary outcome** is purchase conversion among exposed users within seven days of first eligible exposure.

Illustrative planning assumptions are 10% control conversion, a 2 percentage-point absolute MDE, two-sided alpha 0.05, and 80% power. The independent-proportions approximation requires 3,841 users per arm; 5% inflation gives 8,086 exposed users total. Run at least 14 days and two full weekly cycles, continuing until the sample threshold is reached. Use a fixed stopping rule except for prespecified safety/privacy/reliability triggers.

Guardrails: service errors, p95 latency, opt-out/complaint rate, refunds/cancellations, and gross margin per exposed user. Logs distinguish assignment, generated list, rendered exposure, viewability, ranks/items/model hash, outcome timestamps, and service health. Apply purpose limitation, minimization, encryption, access control, retention/deletion, pseudonymization, and Thai PDPA review. Offline ranking quality is not causal business uplift.
"""
        ),
        md(
            """
## Final conclusion

The pipeline is temporally valid and reproducible under the declared independent conventions. The evidence does **not** establish that the hybrid is superior: the sample is tiny, the paired intervals include zero, the metadata is synthetic, and the offline data lacks randomized exposure. The correct next step is a privacy-reviewed, user-randomized online test - not another round of test-set tuning.
"""
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
