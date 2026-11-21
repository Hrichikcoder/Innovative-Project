# Mutual Information–Aware Federated Learning

This repository contains FedAvg, FedCurv, and MI-IAA federated-learning experiments for CIFAR-10 and the supported MNIST/MedMNIST datasets.

## Important clarification: what MI means in this implementation

The current implementation does **not** calculate mutual information by removing a client parameter and comparing model accuracy with and without that parameter. It also does not perform parameter ablation or train two copies of a model to measure a parameter's causal effect on the score.

Instead, it estimates the statistical dependence between each CNN latent feature and the data label using MINE (Mutual Information Neural Estimation).

For every client, the CNN produces:

- `z`: a 128-dimensional latent feature vector
- `y`: the one-hot encoded class label
- `PerFeatureMINENetwork`: a separate critic associated with each latent dimension

For latent feature `z[k]`, the critic receives only `(z[k], y)` and estimates:

\[
\widehat I(Z_k;Y)
=
\mathbb{E}_{(z_k,y)\sim p(z_k,y)}[T_k(z_k,y)]
-
\log\mathbb{E}_{(z_k,y')\sim p(z_k)p(y)}[e^{T_k(z_k,y')}]
\]

The joint expectation uses real `(latent feature, label)` pairs. The marginal expectation uses the same latent features paired with labels that have been randomly shuffled. The code averages the 128 per-feature estimates into one client score:

```python
feature_mi_estimates = torch.mean(t_joint, dim=0) - log_mean_exp
avg_mi_estimate = torch.mean(feature_mi_estimates)
```

This is an estimated lower bound on mutual information, not an exact entropy calculation and not a measurement of parameter impact.

## How MI is used during local training

The implementation does the following for each client:

1. Copy the latest global CNN.
2. Run the CNN to obtain logits and latent features `z`.
3. Train 128 MINE critics using real and shuffled label pairs.
4. Freeze the MINE critic and estimate client MI.
5. Minimize:

\[
L
=
L_{CE}
-\lambda_{MI}\widehat I(Z;Y)
+\frac{\lambda_{reg}}{2}\|w-w_{global}\|^2
\]

The current constants are:

```python
LAMBDA_MI = 0.01
LAMBDA_REG = 0.01
MINE_LR = 0.001
MI_TEMPERATURE = 5.0
```

Subtracting the estimated MI encourages the CNN representation to preserve label-relevant information. The FedProx-style term limits drift from the global model.

The critic is separated from the classifier graph so that MINE's own loss does not unintentionally update the CNN. An exponential moving average is used to stabilize the critic's marginal partition estimate.

## Current parameter importance calculation

The current server aggregation uses a separate **update-magnitude proxy**, not accuracy ablation.

For every parameter tensor `l`, the code computes:

```python
update_magnitude = torch.abs(
    final_weights[key] - initial_weights[key]
).mean().item()
```

This measures how much that client changed the parameter tensor during local training. It does not measure how much model accuracy changes when the parameter is removed or reduced.

For client `i` and parameter tensor `l`, the server first computes:

\[
s_{i,l}
=
\operatorname{normalizedMI}_i
\times
\operatorname{updateMagnitude}_{i,l}
\times
\frac{n_i}{\sum_j n_j}
\]

The server then:

1. Standardizes these scores per parameter tensor.
2. Applies a softmax across clients.
3. Uses the resulting `alpha[i, l]` weights for that tensor.
4. Aggregates each tensor using:

\[
w_l^{global}=\sum_i \alpha_{i,l}w_{i,l}
\]

See `average_weights_mi_iaa()` in `new_codes/mine.py`.

## What the current method measures

The implementation answers:

- How statistically dependent is each latent feature on the label?
- How much did each client change each parameter tensor?
- How many samples does the client contribute?

It does **not** answer:

- How much global or client accuracy decreases if one parameter is masked?
- How much accuracy changes if one neuron, layer, or client update is removed?
- Whether the result is a causal effect or a correlation between labels and latent features?

`Avg Layer Weight %` is therefore a server aggregation weight. It should not be described as a causal accuracy or parameter-influence result.

## Simplest parameter-impact experiment

The simplest defensible extension is **layer-wise leave-out evaluation**. Do not start by evaluating every individual parameter; that is expensive and usually unstable.

For each local model:

1. Store the full client model.
2. Create a copy of the global model or the aggregated model.
3. Mask or remove one complete parameter tensor in turn.
4. Replace that tensor with a baseline, such as zero or the global tensor.
5. Evaluate the affected model on a fixed validation split.
6. Compute:

\[
\Delta_{i,l}
=
\operatorname{Accuracy}(full_{i})
-
\operatorname{Accuracy}(without_{i,l})
\]

A larger positive value indicates that the parameter tensor appears more important for that client/model.

A practical pseudocode implementation is:

```python
import copy


def measure_layer_impact(model_state, dataloader, baseline_state,
                          evaluate_fn, device):
    impact = {}

    full_model = copy.deepcopy(model_state)
    full_acc = evaluate_fn(full_model, dataloader, device)

    for name in model_state:
        altered_state = copy.deepcopy(full_model)

        # A simple baseline for a parameter-ablation experiment:
        altered_state[name] = copy.deepcopy(baseline_state[name])

        altered_acc = evaluate_fn(altered_state, dataloader, device)
        impact[name] = full_acc - altered_acc

    return impact
```

The baseline should be fixed before the experiment. Options include:

- The corresponding parameter in the previous global model
- Zeroes
- A small random tensor

The previous global model is usually the easiest and most meaningful baseline for federated training.

Because one forward pass is not enough to separate genuine importance from noise, run the experiment on a fixed validation batch or client test set and, where possible, repeat it over several seeds or rounds.

## Recommended improvement path

### 1. Preserve the current MINE result, but label it accurately

- Keep per-feature MINE estimates.
- Report the mean and variance of feature estimates.
- Store the critic loss and MI estimate independently.
- Clearly describe MI as a learned statistical estimate.
- Do not present MINE scores as exact information or causal parameter importance.

### 2. Add stability checks

- Compare MINE estimates over multiple shuffles.
- Use a held-out sample for final reporting.
- Track MI values and critic losses per round.
- Measure variance across batches and across random seeds.
- Check for negative or exploding estimates.

### 3. Account for non-IID client distributions

The Dirichlet split in `new_codes/data_pipeline.py` can make the label distribution differ significantly between clients.

Report for each client:

- Raw MINE estimate
- Label entropy
- Number of samples
- Local accuracy
- Aggregation weight

This helps distinguish meaningful high-MI clients from clients whose MI score mainly reflects a highly skewed local distribution.

### 4. Normalize for label entropy

A possible, not-yet-implemented normalization is:

\[
MI_{normalized}
=
\frac{\widehat I(Z;Y)}{H(Y)+\epsilon}
\]

This helps compare clients with different label-distribution sizes. It is a rescaling, not a replacement for MINE.

### 5. Add layer-aware ablation results

Before changing aggregation, collect layer-wise accuracy impacts for a fixed validation set. Then compare:

- `FedAvg`: sample-count weighting
- `MI-IAA`: current MINE + update-magnitude weighting
- `MI-IAA-ablation`: MINE or update-magnitude weighting plus measured layer impact
- `FedProx` or `FedCurv`: existing alternatives

A safe aggregation strategy is to blend the new impact weight with a lower bound based on FedAvg:

\[
\alpha_{i,l}^{final}
=
(1-\beta)\frac{n_i}{\sum_jn_j}
+
\beta\alpha_{i,l}^{impact}
\]

This prevents a noisy MI or ablation estimate from silencing a client completely.

## Suggested experiments

For a controlled comparison, run the same seed and partition for:

1. FedAvg
2. FedCurv
3. MINE regularization only
4. MINE aggregation only
5. Full MI-IAA
6. Full MI-IAA with layer-wise impact weights

Record:

- Global test accuracy
- Average personalized client accuracy
- Per-class accuracy
- Raw MINE estimates
- MI variance
- Critic loss
- Client aggregation weights
- Layer-wise accuracy impact
- Training rounds and wall-clock time

The best version should be selected using held-out performance and stability, not by choosing the implementation that produces the largest MI estimate or the largest client weights.

## Running the MI-IAA experiment

Run from `new_codes/`:

```bash
python mine.py --dataset cifar10
```

Supported choices are:

```text
cifar10
mnist
pathmnist
bloodmnist
```
