# BrainML Social Attention: Research Revival Plan

## Purpose

This document is a handoff for a coding agent working on the existing `social_navigate-dist-sn` repository.

The immediate goal is **not** to redesign everything or add foundation models. The first goal is to make the existing research code trustworthy, reproduce a clean baseline, and then test one concrete architectural hypothesis:

> **Does a hierarchical representation of each mouse, learned from its 12 body keypoints before modeling inter-mouse interactions, improve trajectory forecasting and produce more behaviorally meaningful representations than treating all 36 keypoints as independent social agents?**

The existing repository should be treated as historical research code. Preserve it as a baseline, but audit it carefully before trusting any result.

---
## 0. Preface 

This repo tracks a submodule called `socialAttention` which is the author's actual implementation of their work. However, there is a catch. The original implementation is not completely free of logical/mathematical errors and hence, this should be used with __caution__. 


## 1. Dataset and problem setup

Use the MABe22 Mouse Triplets pose dataset.

Each sequence contains three interacting mice recorded from a top-down camera at 30 Hz. Each mouse is represented by 12 fixed anatomical keypoints. Therefore, the raw pose tensor for a batch can be conceptualized as:

```text
X: [B, T, M, K, C]

B = batch size
T = number of frames
M = 3 mice
K = 12 keypoints per mouse
C = 2 coordinates (x, y)
```

For example:

```text
[B, 20, 3, 12, 2]
```

The repository currently reformulates this as 36 graph nodes per frame.

MABe22 also provides behavior-oriented downstream labels for evaluating learned representations. The public Mouse Triplets task includes frame-level chasing labels and sequence-level light-cycle labels. The broader MABe22 benchmark was specifically designed to evaluate whether learned representations preserve information about behavior and experimental conditions.

### Immediate tasks

We care about two related problems:

1. **Trajectory forecasting**
   - Input: observed mouse pose history.
   - Output: future mouse keypoint positions, represented probabilistically.

2. **Behavior representation learning**
   - Do not initially train the forecasting network directly on behavior labels.
   - Instead, ask whether the internal representation learned through future-motion prediction already contains information that makes behavior linearly decodable.

The second question is intentionally different from plain supervised behavior classification.

---

## 2. Existing repository: what it currently does

The code extends the Social Attention / Structural-RNN formulation from pedestrian trajectories to MABe mouse pose trajectories.

Important existing defaults in the repository include:

```text
sequence length:        20 frames
prediction length:      12 frames
observation length:      8 frames
node input size:         2
edge input size:         2
node embedding size:    64
edge embedding size:    64
node RNN hidden size:  128
edge RNN hidden size:  256
attention size:         64
node output size:        5
```

The five output values correspond to the parameters of a bivariate Gaussian for the next 2D position:

```text
mu_x, mu_y, sigma_x, sigma_y, rho
```

The old graph formulation models individual keypoints as nodes. With 3 mice x 12 keypoints, this gives 36 nodes.

The historical sparse graph implementation deliberately avoided spatial edges between keypoints belonging to the same mouse to reduce graph size. The experimental notes describe:

```text
36 temporal edges
864 cross-mouse spatial edges
900 total edges
```

A newer `st_graph_full.py` constructs a fuller graph. These variants must be audited and clearly named rather than silently mixed.

---

## 3. Phase 0: correctness and paper-fidelity audit

Do this before adding the hierarchical architecture.

### 3.1 Compare the current implementation with the original Social Attention paper/code

Audit at least:

- node definition
- temporal edge construction
- spatial edge construction
- edge feature definition
- node feature definition
- spatial edge RNN
- temporal edge RNN
- attention calculation
- node RNN update
- bivariate Gaussian output head
- training loss
- inference rollout
- hidden-state initialization and persistence

Do not assume that old code is correct because it trains.

### 3.2 Audit mouse-specific modifications

Check carefully:

- mapping from `(mouse_id, keypoint_id)` to node ID
- graph indexing
- duplicate or missing edges
- whether edge tensors match `edgesPresent`
- whether x/y indices are correct everywhere
- temporal alignment between inputs and shifted targets
- handling of missing keypoints
- coordinate normalization
- sequence slicing
- train/validation leakage
- stochastic sequence window sampling

There are suspicious areas in the historical graph code and sampling/loss path. Write tests before changing behavior.

### 3.3 Geodesic loss audit

The existing project includes a Lie-space/geodesic structural loss intended to preserve mouse anatomy.

Do **not** assume the current implementation provides a differentiable structural-loss path.

The repository's function named `sample_gaussian_2d_reparam` should be audited because parts of the historical implementation detach tensors through operations such as `.data`, CPU transfer, and NumPy conversion. A function being named "reparam" is not evidence that gradients actually flow.

Required test:

```python
loss.backward()
assert gaussian_parameter.grad is not None
```

Test this specifically for the structural-loss path.

### 3.4 Minimum tests to add

Add small unit/integration tests for:

- expected node count
- expected edge count for each graph variant
- exact input/output shapes
- no NaNs in Gaussian parameters
- valid sigma values
- rho constrained to valid range
- gradient flow from Gaussian NLL
- gradient flow from geodesic loss
- deterministic one-batch overfit test
- autoregressive inference shape and stability

A tiny subset should be able to overfit. If it cannot, stop and debug before running real experiments.

---

# 4. Baseline A: flat keypoint Social Attention

This is the existing conceptual formulation and should remain as the primary comparison baseline.

### Input

```text
[B, T, 3, 12, 2]
```

Flatten mouse/keypoint dimensions:

```text
[B, T, 36, 2]
```

Each anatomical keypoint is treated as a graph node.

### Important point

Do not prejudge this model as bad. The hypothesis is that it may be inefficient or may learn less clean social representations because it must discover mouse-level grouping implicitly. The experiment should demonstrate that rather than assume it.

### Outputs

For each of 36 keypoints:

```text
[B, T, 36, 5]
```

where 5 represents bivariate Gaussian parameters.

---

# 5. Baseline B / proposed model: hierarchical Social Attention

## Central idea

Separate two fundamentally different forms of interaction:

1. **within-mouse anatomical interaction**
2. **between-mouse social interaction**

The old flat graph combines these concepts at the keypoint level.

The hierarchical model first learns a representation of each individual mouse from its 12 keypoints. Social Attention then operates over 3 mouse representations instead of 36 independent keypoint nodes.

This gives two interpretable attention levels:

```text
keypoint attention: body-part relationships within one mouse
social attention:   interactions between mice
```

---

# 6. Hierarchical architecture with exact tensor shapes

Use these dimensions as the **first implementation**, not as immutable architecture choices.

## 6.1 Raw input

```text
X = [B, T, 3, 12, 2]
```

No learned keypoint-ID embedding is required initially because keypoint ordering is fixed and anatomically consistent.

Do not add unnecessary embeddings until an ablation justifies them.

## 6.2 Per-keypoint feature projection

For each mouse at each time step, isolate:

```text
[B, T, 3, 12, 2]
```

Flatten the leading dimensions temporarily:

```text
[B*T*3, 12, 2]
```

Apply a shared point-wise MLP:

```text
Linear(2 -> 64)
ReLU / GELU
Linear(64 -> 64)   # optional; keep simple initially
```

Result:

```text
H_kp = [B*T*3, 12, 64]
```

Reshaped conceptual form:

```text
[B, T, 3, 12, 64]
```

## 6.3 Within-mouse self-attention

Apply self-attention over the 12 keypoints independently for every mouse and frame.

Input:

```text
[B*T*3, 12, 64]
```

For a simple 4-head attention layer:

```text
d_model = 64
n_heads = 4
head_dim = 16
```

Attention matrix per mouse/frame/head:

```text
[4, 12, 12]
```

Batched:

```text
[B*T*3, 4, 12, 12]
```

Self-attention output:

```text
[B*T*3, 12, 64]
```

This stage answers:

> Which body parts are useful for representing the current pose of this mouse, conditional on the other body parts?

## 6.4 Attention pooling

Attention pooling is **not** the average of the self-attention matrix.

Instead, assign a learned scalar relevance score to each contextualized keypoint representation.

For each keypoint representation `h_i in R^64`:

```text
score_i = w^T tanh(W h_i)
alpha = softmax(score over 12 keypoints)
mouse_vector = sum_i alpha_i * h_i
```

Shapes:

```text
contextualized keypoints: [B*T*3, 12, 64]
pooling logits:           [B*T*3, 12, 1]
pooling weights:          [B*T*3, 12, 1]
mouse summary:            [B*T*3, 64]
```

Restore dimensions:

```text
mouse_summary = [B, T, 3, 64]
```

Do **not** add a 64 -> 128 projection merely because 128 sounds larger. Start with the 64D bottleneck.

## 6.5 Global translation and velocity

The learned 64D mouse summary should primarily capture articulated pose. The social model also needs global motion information.

Compute a simple mouse reference position for each frame, preferably using a stable anatomical reference or centroid. Keep the choice explicit and testable.

For each mouse:

```text
position = [x, y]            -> 2D
velocity = [dx, dy]          -> 2D
```

Then:

```text
global_motion = [B, T, 3, 4]
```

Concatenate:

```text
node_features = concat(mouse_summary, global_motion)
              = [B, T, 3, 68]
```

Then apply the Social Attention node embedding:

```text
Linear(68 -> 64)
```

Result:

```text
node_embedding = [B, T, 3, 64]
```

This is the point at which the hierarchical representation enters the mouse-level Social Attention network.

---

# 7. Mouse-to-mouse edge representation

With only 3 mice, construct social edges between mouse-level nodes rather than keypoints.

Start with simple relative motion features.

For mouse `i` relative to mouse `j`:

```text
delta_position = position_j - position_i      # 2D
```

Optionally later include:

```text
delta_velocity                                 # 2D
relative heading
relative pose-summary features
pairwise distance
```

But do not put all of these into the first run.

The first implementation should be as close as possible to the original Social Attention edge semantics.

For three mice, directed spatial interactions can be represented as:

```text
3 * 2 = 6 directed social edges
```

plus 3 temporal self-edges.

This is dramatically smaller than the flat 36-keypoint graph.

---

# 8. Mouse-level temporal/social model

Reuse the conceptual Social Attention structure where possible rather than replacing it with a generic Transformer.

For each mouse node:

```text
node embedding:             64D
node recurrent state:      128D
social edge recurrent state: use existing baseline dimensionality initially
attention representation:   64D
```

The goal is to answer whether the hierarchical representation helps, not whether a completely different architecture helps.

A clean experiment changes **one major modeling assumption at a time**.

---

# 9. Critical decoder problem

Compressing 12 keypoints into one 64D mouse vector solves encoding, but trajectory forecasting still requires predicting future locations for all 12 keypoints.

Do not overlook this.

## Recommended first decoder

After the mouse-level Social Attention / recurrent state produces a contextual mouse state:

```text
mouse_context = [B, T, 3, 128]
```

use a shared keypoint decoder conditioned on:

1. the mouse context
2. the corresponding keypoint's current contextualized encoder feature
3. optionally its last observed coordinate

For keypoint `k`:

```text
input_k = concat(
    mouse_context,          # 128
    keypoint_feature_k,     # 64
    last_xy_k               # 2
)

input_k dimension = 194
```

Then:

```text
MLP(194 -> 128 -> 5)
```

Output:

```text
[B, T, 3, 12, 5]
```

The same decoder weights should be shared across keypoints initially.

This preserves the anatomical details that were compressed during pooling while allowing social context to affect each keypoint forecast.

### Alternative decoder for later

A structured pose decoder or graph decoder can be tested later, but it should not block the first hierarchical experiment.

---

# 10. Losses

## 10.1 Primary trajectory loss

Use the bivariate Gaussian negative log-likelihood from the baseline.

This should remain the primary objective so that the flat and hierarchical models are directly comparable.

## 10.2 Structural / geodesic loss

Only enable this after its gradient path is verified.

Run experiments both:

```text
without structural loss
with structural loss
```

Do not attribute improvements to it without an ablation.

## 10.3 Do not add behavior supervision yet

For the representation-learning experiment, behavior labels must not shape the trajectory encoder initially.

Otherwise the claim that forecasting *learns* behavioral information becomes circular.

---

# 11. Core experiments

Keep the experiment matrix small enough to finish and interpret.

## Experiment 1: reproduce historical flat baseline

```text
Flat 36-keypoint Social Attention
Gaussian trajectory loss
No geodesic loss initially
```

Goal: establish a trusted forecasting baseline.

## Experiment 2: hierarchical representation

```text
12 keypoints
-> shared point MLP
-> within-mouse self-attention
-> attention pooling
-> 64D mouse summary
-> mouse-level Social Attention
-> keypoint-conditioned decoder
```

Keep training procedure and evaluation split identical to Experiment 1.

## Experiment 3: structural loss ablation

Once gradient flow is confirmed:

```text
Flat + geodesic
Hierarchical + geodesic
```

Only run this after Experiments 1 and 2 work.

---

# 12. Trajectory evaluation

At minimum report:

## ADE

Average Displacement Error across predicted frames.

## FDE

Final Displacement Error at the final prediction horizon.

Report metrics at the same coordinate scale for every model.

Also strongly consider:

## Anatomical consistency error

Measure deviation of predicted inter-keypoint / bone lengths from the observed anatomy.

This is particularly important because a model could obtain reasonable point-wise displacement while generating physically implausible mouse poses.

Possible metric:

```text
mean absolute relative bone-length error
```

computed over a fixed anatomical edge set.

---

# 13. Behavior representation experiment

This is a separate evaluation of what the trajectory model has learned.

## Research question

> Does learning to predict future social motion produce latent representations that encode recognizable mouse behavior, even though behavior labels were never used during trajectory training?

This is **not** ordinary behavior classification.

## Representation extraction

After trajectory training, freeze the model.

Extract at least two representations:

### A. Mouse-level representation

For hierarchical model:

```text
[B, T, 3, 64]     # pooled pose representation
```

or the later temporal node state:

```text
[B, T, 3, 128]
```

Test both if inexpensive.

### B. Scene-level representation

Pool across the three mice, e.g. mean or attention pooling:

```text
[B, T, 128]
```

This representation is useful for scene-level / any-mouse behavior labels such as chasing.

## Linear probe

Freeze the trajectory network completely.

Train only a linear classifier on behavior labels:

```text
z_t -> Linear(D -> number_of_labels)
```

For binary chasing:

```text
z_t -> Linear(D -> 1)
```

The purpose of the linear probe is diagnostic:

> How easily can behavior be extracted from the representation without letting a powerful downstream classifier learn behavior from scratch?

## Baselines for behavior decoding

Compare against at least:

1. raw or simple pose/motion features + linear classifier
2. randomly initialized encoder representation + linear classifier
3. flat Social Attention representation + linear classifier
4. hierarchical Social Attention representation + linear classifier

This is what makes the result meaningful.

## Metrics

Because behavior labels can be imbalanced, do not rely on accuracy alone.

Use appropriate metrics such as:

```text
Average Precision / mAP
F1 or Macro-F1 where appropriate
AUROC as a secondary metric
```

For the MABe22 public chasing task, use a frame-level evaluation aligned with the provided labels.

---

# 14. Representation interpretability

If the linear probe demonstrates that behavior is decodable, investigate **where** that information appears.

Do not begin with interpretability before demonstrating that behavioral information exists in the representation.

## 14.1 Event-aligned hidden-state analysis

For a behavior episode beginning at frame `T0` and ending at `T1`, align hidden activations around behavior onset:

```text
T0 - pre_window ... T0 ... T1 ... T1 + post_window
```

For each latent dimension, calculate an event-triggered average across many examples.

Look for dimensions whose activation:

- rises before behavior onset
- peaks during behavior
- falls after behavior ends
- consistently repeats across episodes

This is stronger than showing one interesting trajectory.

## 14.2 Keypoint attention maps

For the hierarchical model, log within-mouse attention and attention-pooling weights.

Possible questions:

- Does grooming increase attention on forepaws / head-related keypoints?
- Does chasing emphasize nose, body direction, or tail orientation?
- Do these patterns emerge consistently across episodes?

Do not interpret individual attention weights as causal evidence.

Treat them as descriptive until tested with interventions.

## 14.3 Social attention maps

Inspect mouse-to-mouse attention over time.

Questions:

- Does attention between two mice increase before chasing begins?
- Which mouse becomes the dominant interaction partner?
- Does social attention reorganize during behavior transitions?

## 14.4 Functional ablation

If a latent component or attention pathway appears associated with behavior, test it.

Examples:

- zero selected hidden dimensions
- suppress selected keypoint features
- mask an interaction edge
- replace attention distribution with uniform weights

Then measure:

```text
change in trajectory ADE/FDE
change in behavior linear-probe performance
behavior-specific change in forecasting error
```

This moves the analysis from correlation toward functional evidence.

---

# 15. Efficiency comparison

One motivation for the hierarchical architecture is computational.

Measure rather than merely claim it.

Log:

```text
number of graph nodes
number of graph edges
parameters
GPU memory peak
training step time
examples/second
inference latency
```

Compare flat versus hierarchical models.

This can become a meaningful result even if ADE/FDE are similar.

---

# 16. Tomorrow's execution order

The goal for the first day is a credible baseline plus one hierarchical run, not the full research program.

## Stage 1: repository cleanup

1. Create a clean branch.
2. Make environment installation reproducible.
3. Remove hardcoded cluster/device assumptions.
4. Add CPU/CUDA device selection.
5. Make single-GPU execution the default.
6. Add deterministic seeds.
7. Add a tiny debug dataset mode.
8. Add shape/gradient tests.

## Stage 2: baseline sanity

1. Load a very small MABe subset.
2. Visualize several sequences.
3. Verify normalization and keypoint identities.
4. Run flat graph construction.
5. Assert graph counts.
6. Overfit one tiny batch.
7. Run autoregressive rollout.
8. Compute ADE/FDE.

Do not proceed until this works.

## Stage 3: hierarchical encoder

Implement only:

```text
2D point
-> 64D point MLP
-> one self-attention block
-> attention pooling
-> 64D mouse summary
```

Add shape tests and attention logging.

Expected shapes:

```text
input:                 [B, T, 3, 12, 2]
point embeddings:      [B, T, 3, 12, 64]
self-attention output: [B, T, 3, 12, 64]
pooling weights:       [B, T, 3, 12]
mouse summary:         [B, T, 3, 64]
```

## Stage 4: mouse-level Social Attention

Add:

```text
mouse summary 64D + global motion 4D
-> node input 68D
-> node embedding 64D
-> Social Attention / recurrent context
```

## Stage 5: decoder

Use contextual mouse state + per-keypoint contextual feature + previous coordinate to output five Gaussian parameters per keypoint.

Verify:

```text
[B, T, 3, 12, 5]
```

## Stage 6: first comparison

Run flat and hierarchical models under the same small configuration.

Save:

- train loss
- validation NLL
- ADE
- FDE
- runtime
- peak GPU memory

Do not spend the first day hyperparameter tuning.

---

# 17. Compute environment

The historical repository contains distributed / SLURM code, but that is not required for the first experiments.

Prefer a free Kaggle GPU notebook if local training is too slow.

For the first implementation:

```text
single GPU
mixed precision only after correctness
small dataset subset
short run
```

Only scale after the model can overfit a tiny batch and produce valid trajectories.

Keep distributed training support in the repo, but do not let DDP debugging consume the research day.

---

# 18. What not to do yet

Do **not** do these in the first pass:

- GLM / LLM integration
- cross-dataset group-size generalization
- CalMS21 transfer
- 2-mouse vs 3-mouse representation alignment
- CKA / RSA across foundation models
- complex graph transformers
- keypoint identity embeddings without evidence they are needed
- large hyperparameter sweeps
- rebuilding the entire repository from scratch
- claiming attention maps are explanations

These are follow-up experiments after a trusted hierarchical baseline exists.

---

# 19. Follow-up research directions

If the first experiments work, extend in this order.

## A. Representation quality

Compare flat and hierarchical representations with behavior linear probes.

## B. Behavior-transition interpretability

Perform event-aligned hidden-state and attention analyses around annotated behaviors.

## C. Causal / functional tests

Ablate selected components and quantify the effect on behavior-specific forecasting.

## D. Group-size generalization

Investigate whether mouse-level social representations can transfer across datasets with different numbers of interacting animals.

This requires careful handling because MABe22 Mouse Triplets uses 3 mice, while datasets such as CalMS21 use pairs of mice and have different keypoint schemas and behavior taxonomies.

Do not compare representations directly by KL divergence unless they are distributions defined in a common aligned space.

For representation-space comparison, later consider methods such as:

- CKA
- representational similarity analysis
- Procrustes alignment followed by distance comparison

## E. Foundation-model comparison

Only after the task and evaluation pipeline are mature, test whether an open-weight foundation model can encode the same trajectory windows and compare representation geometry / downstream behavior decodability.

The foundation model should answer an existing research question. It should not become the research question merely because open weights are available.

---

# 20. Criteria for a successful first milestone

A successful first milestone does **not** require beating the old model dramatically.

It requires:

- trustworthy data loading
- verified graph construction
- a reproducible flat baseline
- a working hierarchical model
- identical evaluation for both
- clean ADE/FDE comparison
- efficiency measurements
- saved internal representations and attention weights

A particularly strong result would be any of the following:

1. hierarchical model improves ADE/FDE
2. hierarchical model matches forecasting accuracy with substantially lower graph complexity
3. hierarchical representation produces better behavior linear-probe performance
4. interpretable keypoint/social attention changes emerge around behavior transitions
5. structural loss improves anatomical consistency without harming trajectory accuracy

Any one of these can justify continuing the project.

---

# 21. Research story

The project should ultimately answer a coherent sequence of questions:

### Question 1
Can Social Attention be extended from single-point pedestrians to articulated, multi-agent animal trajectories?

### Question 2
Is it better to model every body keypoint as an independent social agent, or explicitly learn a hierarchical mouse representation first?

### Question 3
Does future-motion prediction force that representation to encode meaningful social behavior even without behavior supervision?

### Question 4
Where does behavioral information appear in the learned representation, and does perturbing those components affect forecasting?

This is substantially stronger than "trained a behavior classifier" or "fine-tuned a model on mouse data." It combines trajectory forecasting, structured representation learning, multi-agent modeling, and representation interpretability in one testable research program.

---

# 22. Relevant references

- Vemula et al., **Social Attention: Modeling Attention in Human Crowds**, ICRA 2018 / arXiv:1710.04689.
- Sun et al., **MABe22: A Multi-Species Multi-Task Benchmark for Learned Representations of Behavior**, ICML 2023.
- MABe22 Mouse Triplets challenge documentation for the exact pose tensor format and public downstream labels.
- CalMS21 is a useful later transfer benchmark, but it is a two-mouse dataset with a different pose/keypoint setup and should not be mixed into the first experiment.

---

## Instruction to the coding agent

Work incrementally. Preserve the historical baseline. Do not silently repair architectural choices and then compare against a moving target. Every meaningful model change must have:

1. a stated hypothesis,
2. a controlled baseline,
3. an observable metric,
4. shape/gradient tests,
5. reproducible configuration and seed,
6. saved results sufficient to reproduce the comparison.

Correctness takes priority over speed. The first useful outcome is a trusted experiment, not a large model.
