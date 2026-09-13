# Hierarchical Multi-Mouse Trajectory Model

## Goal

This repository should implement and evaluate a hierarchical spatiotemporal model for multi-mouse trajectory prediction.

The key idea is to separate:

1. **Local, within-mouse dynamics**
   - Each mouse is represented by 12 keypoints.
   - Each mouse has its own spatiotemporal graph.
   - This graph learns per-keypoint hidden states and a compact mouse-level global representation.

2. **Social, cross-mouse dynamics**
   - Each mouse-level global representation becomes a node in a higher-level social graph.
   - The social graph models how the mice influence one another.
   - The social context is recomputed at every autoregressive decoder step.

3. **Keypoint-level trajectory decoding**
   - The decoder predicts the future distribution of every keypoint.
   - The local keypoint hidden states retain fine-grained pose information.
   - The social context provides interaction-aware conditioning.

The architecture must remain strictly causal. No information from time `t+1` may be used to predict time `t+1`.

---

## Input

Assume:

- `B` = batch size
- `T_obs` = number of observed frames
- `T_pred` = number of frames to predict
- `M` = number of mice, initially `3`
- `K` = number of keypoints per mouse, `12`
- `D_h` = hidden size, initially `128`

Raw input:

```text
X.shape = [B, T_obs, M, K, 2]
```

For one mouse:

```text
X_i.shape = [B, T_obs, K, 2]
```

The final dimension contains `(x, y)` coordinates.

---

## High-Level Architecture

```text
Raw keypoints
    |
    v
Per-mouse local spatiotemporal graph
    |
    +--> Keypoint hidden states H_i^t     [K, D_h]
    |
    +--> Global mouse state G_i^t         [D_h]
             |
             v
      Cross-mouse social graph
             |
             v
      Social context S_i^t                [D_h]
             |
             v
Decoder(H_i^t, S_i^t)
             |
             v
Next-keypoint distribution C_i^(t+1)      [K, 5]
             |
             v
Sample / mean coordinates                 [K, 2]
             |
             v
Update H_i^(t+1)
             |
             v
Update G_i^(t+1)
             |
             v
Recompute S_i^(t+1)
             |
             v
Repeat autoregressively
```

---

# 1. Local Per-Mouse Spatiotemporal Encoder

Each mouse is modeled independently first.

For mouse `i` at time `t`, maintain:

```text
H_i^t ∈ R^(K x D_h)
```

With the current defaults:

```text
H_i^t.shape = [12, 128]
```

These are the hidden states of the 12 keypoints.

The local graph should model:

- spatial relationships between keypoints within the same mouse
- temporal relationships for each keypoint across frames

This is conceptually similar to the original Social Attention / spatiotemporal graph architecture, but applied to the articulated pose of one mouse.

---

# 2. Global Mouse State

Each mouse also maintains one global state:

```text
G_i^t ∈ R^(D_h)
```

Example:

```text
G_i^t.shape = [128]
```

This global state is not simply an average of the keypoint hidden states.

It should be a learned representation that summarizes the current pose and recent dynamics of the entire mouse.

A reasonable implementation is a learnable global graph node that participates in the local per-mouse graph and is updated at every frame.

Conceptually:

```text
G_i^t = GlobalUpdate(
    G_i^(t-1),
    H_i^t
)
```

The exact update can be implemented with attention/message passing plus a recurrent update.

The important point is:

> The global mouse state evolves over time and is not a fixed pooled vector for the entire observation window.

For all mice:

```text
G^t.shape = [M, D_h]
```

With three mice:

```text
G^t.shape = [3, 128]
```

---

# 3. Cross-Mouse Social Graph

The higher-level social graph operates on one node per mouse.

At time `t`:

```text
G^t = [G_1^t, G_2^t, G_3^t]
```

with:

```text
G^t.shape = [3, 128]
```

The cross-mouse graph models social interactions using:

- mouse-level node states
- spatial edge states between mice
- temporal edge states for each mouse
- attention over spatial interactions

The output is one social context vector per mouse:

```text
S_i^t ∈ R^(D_h)
```

For all mice:

```text
S^t.shape = [3, 128]
```

Important:

`S_i^t` is specific to mouse `i`.

It is not one shared context vector copied to all mice.

Each `S_i^t` is computed using information from the other mice and the current social graph state.

---

# 4. Decoder

For each mouse `i`, at autoregressive step `t`, the decoder receives:

```text
H_i^t ∈ R^(12 x 128)
S_i^t ∈ R^(128)
```

The social context can be broadcast across keypoints:

```text
S_i^t -> [12, 128]
```

Then concatenate:

```text
Z_i^t = concat(H_i^t, broadcast(S_i^t))
```

giving:

```text
Z_i^t.shape = [12, 256]
```

A decoder MLP / recurrent decoder produces:

```text
C_i^(t+1).shape = [12, 5]
```

where each keypoint predicts:

```text
(mu_x, mu_y, sigma_x, sigma_y, rho)
```

These parameterize a bivariate Gaussian.

For all three mice:

```text
C^(t+1).shape = [3, 12, 5]
```

If only deterministic predictions are required for evaluation:

```text
pred_xy = C[..., :2]
```

which gives:

```text
pred_xy.shape = [3, 12, 2]
```

---

# 5. Hidden-State Update

After predicting coordinates for time `t+1`, update the local keypoint hidden states.

For mouse `i`:

```text
H_i^(t+1) = LocalRecurrentUpdate(
    H_i^t,
    C_i^(t+1)
)
```

More precisely, the update should use the previous hidden state and the new keypoint input.

This is the standard recurrent pattern:

```text
new_hidden = RNN(previous_hidden, new_input)
```

The new coordinate prediction alone is not sufficient because it does not contain all information stored in the previous hidden state.

---

# 6. Global-State Update

After all mice have updated local hidden states:

```text
H_1^(t+1)
H_2^(t+1)
H_3^(t+1)
```

update each mouse's global state:

```text
G_i^(t+1) = GlobalUpdate(
    G_i^t,
    H_i^(t+1)
)
```

For all mice:

```text
G^(t+1).shape = [3, 128]
```

This step keeps the mouse-level representation synchronized with the newly predicted local pose.

---

# 7. Social-Context Update

Once all mouse global states have been updated for `t+1`, recompute the cross-mouse social graph.

```text
S^(t+1) = SocialGraphUpdate(
    G^(t+1),
    previous_edge_states
)
```

Output:

```text
S^(t+1).shape = [3, 128]
```

This new social context is then used for the next decoder step.

---

# 8. Causal Autoregressive Order

This ordering is critical.

At prediction step `t`:

```text
H^t, G^t, S^t
    |
    v
Predict C^(t+1)
    |
    v
Update H^(t+1)
    |
    v
Update G^(t+1)
    |
    v
Update S^(t+1)
    |
    v
Predict C^(t+2)
```

In equations, for mouse `i`:

```text
C_i^(t+1) = f_theta(H_i^t, S_i^t)

H_i^(t+1) = u_theta(H_i^t, C_i^(t+1))

G_i^(t+1) = g_theta(G_i^t, H_i^(t+1))

S_i^(t+1) = s_theta(
    G_1^(t+1),
    G_2^(t+1),
    G_3^(t+1),
    edge_states^t
)
```

Then:

```text
C_i^(t+2) = f_theta(H_i^(t+1), S_i^(t+1))
```

There must be no path from any quantity at `t+1` into the prediction of `C^(t+1)` before that prediction is made.

This prevents causal leakage.

---

# 9. Synchronous Multi-Mouse Update

All mice must advance together.

Do not perform:

```text
mouse 1 -> update social graph
mouse 2 -> update social graph
mouse 3 -> update social graph
```

That would introduce ordering bias.

Instead:

```text
1. Predict all three mice at t+1
2. Update all three local hidden states
3. Update all three global states
4. Recompute the social graph once
5. Continue to t+2
```

Pseudo-code:

```python
for step in range(T_pred):

    # 1. Decode all mice using current social context
    dist_params = decoder(H, S)                 # [B, M, K, 5]

    # 2. Obtain coordinates
    pred_xy = distribution_to_xy(dist_params)   # [B, M, K, 2]

    # 3. Update local states for all mice
    H = local_update(H, pred_xy)                 # [B, M, K, D_h]

    # 4. Update mouse-level global states
    G = global_update(G, H)                      # [B, M, D_h]

    # 5. Recompute social interaction state
    S = social_graph(G)                          # [B, M, D_h]

    predictions.append(dist_params)
```

---

# 10. Teacher Forcing

During training, support teacher forcing.

At step `t+1`, instead of feeding the predicted coordinate back into the local update:

```text
C_pred^(t+1)
```

optionally use the ground-truth coordinate:

```text
C_gt^(t+1)
```

For example:

```python
if training and random.random() < teacher_forcing_ratio:
    next_xy = gt_xy[:, step]
else:
    next_xy = pred_xy
```

The hidden-state update then becomes:

```text
H^(t+1) = LocalUpdate(H^t, next_xy)
```

The causal ordering remains valid because the ground truth is only used as the input for the next decoder state during training.

At inference time, teacher forcing must be disabled.

---

# 11. Training Objective

Start simple.

The primary training objective is trajectory prediction.

If predicting a bivariate Gaussian per keypoint:

```text
L = negative log-likelihood
```

over:

```text
mu_x, mu_y, sigma_x, sigma_y, rho
```

Do not immediately add structural, pose, or auxiliary losses.

First determine whether the architecture can learn useful trajectories.

Additional objectives can be introduced later only if justified by evaluation.

---

# 12. Evaluation Metrics

Do not rely only on ADE.

Evaluate three different properties.

## 12.1 Trajectory Accuracy

Measure whether the mouse moves to the correct location.

Recommended:

```text
Center-of-mass ADE
Center-of-mass FDE
Keypoint ADE
Keypoint FDE
```

## 12.2 Pose Accuracy

Measure whether the predicted mouse has the correct orientation.

Example:

```text
orientation = vector(tail_base -> nose)
```

Metric:

```text
mean angular error(pred_orientation, gt_orientation)
```

## 12.3 Structural Consistency

Measure whether the predicted body remains anatomically plausible.

For every anatomical edge `(i, j)`:

```text
error =
abs(
    predicted_distance(i, j)
    -
    ground_truth_distance(i, j)
)
```

Average over anatomical edges, mice, and prediction frames.

These metrics should initially remain evaluation metrics, not training losses.

---

# 13. Baselines

## Baseline A: Flat Graph

Treat all keypoints as nodes in one global graph.

With three mice:

```text
36 keypoints total
```

This is the existing flat-graph baseline.

## Baseline B: Single-Mouse Local Graph

Train the local per-mouse graph without cross-mouse interactions.

Purpose:

Determine whether the existing local architecture can model the articulated pose and trajectory of one mouse.

If:

```text
single-mouse works
flat multi-mouse fails
```

then the likely issue is representation and mixing of intra-mouse versus inter-mouse relationships.

## Model C: Hierarchical Model

Use:

```text
per-mouse local graph
    ->
global mouse states
    ->
cross-mouse social graph
    ->
keypoint decoder
```

This is the main proposed model.

---

# 14. Research Questions

The implementation should help answer:

### RQ1
Can the current spatiotemporal graph model accurately predict a single mouse's articulated trajectory?

### RQ2
Does the flat 36-keypoint graph fail because it mixes anatomical and social relationships?

### RQ3
Does a hierarchical representation improve trajectory prediction?

### RQ4
Does the hierarchical representation improve pose orientation and structural consistency even when ADE is similar?

### RQ5
Do learned global mouse states encode behaviorally meaningful information?

The last question can later be evaluated using linear probes, clustering, transition-aligned activation analysis, attention visualization, and behavior labels from MABe.

Do not make this a requirement for the first implementation.

---

# 15. Recommended Implementation Order

Do not implement everything at once.

### Phase 1
Verify the current flat-graph baseline.

- confirm tensor shapes
- confirm Gaussian head
- confirm autoregressive rollout
- verify normalization / denormalization
- compare mean versus sampled ADE

### Phase 2
Run the single-mouse experiment.

Input:

```text
[B, T, 1, 12, 2]
```

Determine whether local spatiotemporal modeling is sufficient.

### Phase 3
Implement the per-mouse global node.

Verify:

```text
H_i^t -> [12, 128]
G_i^t -> [128]
```

### Phase 4
Build the cross-mouse social graph.

Verify:

```text
G^t -> [3, 128]
S^t -> [3, 128]
```

### Phase 5
Implement the coupled autoregressive decoder.

Verify the exact causal order:

```text
decode
-> local update
-> global update
-> social update
-> next decode
```

### Phase 6
Compare flat versus hierarchical models.

Evaluate trajectory error, pose error, and structural error.

### Phase 7
Only after the model works:

- behavior representation probing
- auxiliary losses
- interpretability
- richer graph structure
- alternative global-node aggregation

---

# 16. Important Constraints for the Coding Agent

1. **Do not silently redesign the architecture.**
2. **Keep tensor shapes explicit in code.** Assertions are encouraged.
3. **Do not update mice sequentially.** Update all mice synchronously.
4. **Do not introduce future information into the current prediction.**
5. **Do not freeze social context during autoregressive rollout.** Recompute it after each joint mouse update.
6. **Do not collapse the 12 keypoint states into only one vector for decoding.**
7. **Do not add auxiliary losses until the baseline architecture is validated.**
8. **Do not assume sampled trajectories and Gaussian-mean trajectories have the same ADE.**
9. **Log all intermediate shapes and state norms during early debugging.**
10. **Make every module independently testable.**

Suggested module structure:

```text
models/
    local_mouse_encoder.py
    global_mouse_node.py
    social_graph.py
    trajectory_decoder.py
    hierarchical_model.py

training/
    train.py
    rollout.py
    losses.py

evaluation/
    trajectory_metrics.py
    pose_metrics.py
    structural_metrics.py

tests/
    test_shapes.py
    test_causality.py
    test_single_mouse.py
    test_rollout.py
```

---

# 17. Core Concept

The architecture should learn two levels of representation:

### Local representation

"What is this mouse doing internally?"

```text
12 keypoint hidden states + global mouse state
```

### Social representation

"How is this mouse being influenced by the other mice?"

```text
social context from the cross-mouse graph
```

The decoder combines:

```text
local pose memory
+
social interaction context
```

to predict:

```text
future articulated pose
```

The objective is not merely to predict where a mouse moves.

It is to forecast the future pose of an articulated agent while preserving the distinction between:

```text
within-mouse dynamics
and
between-mouse interactions
```
