# Matrix Factorization Definition and Verification

For every observed training rating, prediction is

`r_hat(u,i) = mu + b_u + b_i + p_u dot q_i`,

where `mu` is the training-only global mean, `b_u` has one value per training user, `b_i` has one value per training item, `P` is `U x d`, and `Q` is `I x d`. Unknown users/items receive no random factors: both unknown gives `mu`; a known user adds only `b_u`; a known item adds only `b_i`. Rating-error reporting clips predictions to [1,5], while ranking uses raw scores.

The optimized reference objective is `0.5 * mean_observed(error^2) + 0.5 * lambda * (sum(b_u^2) + sum(b_i^2) + sum(P^2) + sum(Q^2))`. Training uses simultaneous full-batch gradient descent, so all terms in one epoch come from the old state. The error-gradient contribution is divided by the number of observed rows; the L2 gradient is not. This is exactly consistent with the displayed objective. At one observed row, the update reduces to `parameter += learning_rate * (residual * counterpart - lambda * parameter)` and reproduces every printed assignment value.

The companion candidate list was unavailable. Before validation evaluation, the independent grid was fixed at factors {2,4,8}, learning rate {0.2,0.5}, regularization {0.001,0.01}, 300 epochs, initialization standard deviation 0.05, and seed 202411. Selection minimizes RMSE on the 17 validation events whose user and item both occur in training. Test data is not loaded by the Phase 02 script.
