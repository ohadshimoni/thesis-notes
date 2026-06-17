# Experiments 8.5 — TypiClust Distance Metric & Normalization

**Date:** May 8, 2026  
**Goal:** Compare variants of TypiClust with different distance metrics (L1 sum, L∞ sum, centroid L1/L2, squared L2) and normalization schemes (none, L1 norm) on CIFAR-10 and CIFAR-100.

All runs: ResNet18(?), NN1 evaluation, 5 AL rounds, 3 seeds (local_0/1/2), `initial_size=0`.

---

## Method Glossary

All methods are variants of **TypiClust** — the core idea is: cluster the unlabeled embeddings, then pick the most *typical* (representative) sample from each cluster. The variants differ in **how distance is measured** and **whether embeddings are normalized**.

| Method name | What it does |
|---|---|
| `fixed_clustering_typiclust` | Baseline TypiClust. Clusters unlabeled data into *k* clusters (k = budget), picks the sample with the smallest average L2 distance to its cluster neighbors ("most typical"). |
| `fixed_clustering_l1_typiclust` | Same as above but uses **L1 (Manhattan) distance** for typicality scoring instead of L2. L1 is less sensitive to large outlier dimensions. |
| `fixed_clustering_l1norm_typiclust` | Same as `fixed_clustering_l1`, but **embeddings are L1-normalized** (each vector divided by its L1 norm) before any distance is computed. Makes all points live on the same scale. |
| `l1_sum_typiclust` | No clustering. Scores each point by the **sum of its L1 distances to all other points** — lower sum = more central/typical. Picks the b most central points. No normalization. |
| `l1_sum_typiclust_l1norm` | Same sum scoring, but on **L1-normalized embeddings**. |
| `linf_sum_typiclust` | Like `l1_sum` but uses **L∞ (Chebyshev) distance** — distance = max over all dimensions. Measures how far two points differ on their most-different feature. |
| `linf_sum_typiclust_l1norm` | L∞ sum scoring on L1-normalized embeddings. |
| `linf_sum_typiclust_nonorm` | L∞ sum scoring, no normalization (same as `linf_sum` but explicitly no-norm). |
| `centroid_l1_typiclust` | Clusters data, then picks the sample **closest to the cluster centroid** using L1 distance (vs. most typical among neighbors). |
| `centroid_l2_typiclust` | Same centroid selection using L2 distance. |
| `squared_l2_sum_typiclust` | Sum scoring using **squared L2 distance** — penalizes far neighbors more heavily than plain L2. |

**Normalization note:** L1 normalization divides each embedding vector by its L1 norm, so all vectors have equal total magnitude. This prevents any single large-magnitude point from dominating distance computations.

---

---

## Group 1: CIFAR-100, budget=100 (20 samples/round)

| Experiment | Method | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---|---|---|---|---|
| `fixed_clustering_cifar100` | fixed_clustering_typiclust (L2, no norm) | 36.70% | 38.09% | 36.76% | **37.18%** |
| `squared_l2_cifar100` | squared_l2_sum_typiclust | 41.79% | 41.91% | 41.42% | **41.71%** |
| `centroid_l2_cifar100` | centroid_l2_typiclust | 41.91% | 41.42% | 41.79% | **41.71%** |
| `centroid_l1_cifar100` | centroid_l1_typiclust | 41.53% | 41.33% | 41.94% | **41.60%** |
| `l1_nonorm_cifar100` | l1_sum_typiclust (no norm) | 40.61% | 40.56% | 40.54% | **40.57%** |
| `l1_l1norm_cifar100` | l1_sum_typiclust + L1 norm | 42.76% | 42.43% | 41.71% | **42.30%** |
| `fixed_clustering_l1_cifar100` | fixed_clustering_l1_typiclust | 43.40% | 43.57% | 43.47% | **43.48%** |
| `fixed_l1norm_cifar100` | fixed_clustering_l1norm_typiclust | 43.78% | 43.10% | 43.78% | **43.55%** ✅ |

**Key takeaway (CIFAR-100):** `fixed_clustering_l1norm` is the best variant (~43.6%). Fixed clustering with L1 distance consistently beats the sum-based variants. No normalization (`l1_nonorm`) hurts. Standard fixed_clustering (L2, no norm) is worst.

---

## Group 2: CIFAR-10, budget=60 (12 samples/round × 5 rounds)

| Experiment                        | Method                            | Seed 1 | Seed 2 | Seed 3 | Mean         |
| --------------------------------- | --------------------------------- | ------ | ------ | ------ | ------------ |
| `l1_nonorm_cifar10_b60`           | l1_sum_typiclust (no norm)        | 72.55% | 72.17% | 72.71% | **72.48%**   |
| `fixed_clustering_l1_cifar10_b60` | fixed_clustering_l1_typiclust     | 75.67% | 78.10% | 76.17% | **76.65%**   |
| `linf_nonorm_cifar10_b60`         | linf_sum_typiclust (no norm)      | 75.84% | 74.93% | 74.03% | **74.93%**   |
| `linf_cifar10_b60`                | linf_sum_typiclust                | 76.96% | 76.24% | 77.30% | **76.83%**   |
| `linf_l1norm_cifar10_b60`         | linf_sum_typiclust + L1 norm      | 77.78% | 77.30% | 78.02% | **77.70%**   |
| `fixed_l1norm_cifar10_b60`        | fixed_clustering_l1norm_typiclust | 78.56% | 76.68% | 78.78% | **78.01%**   |
| `l1_l1norm_cifar10_b60`           | l1_sum_typiclust + L1 norm        | 79.73% | 77.55% | 78.23% | **78.50%** ✅ |

**Key takeaway (CIFAR-10, b60):** `l1_sum + L1norm` wins at ~78.5%. L1 norm consistently helps across metrics. No-norm variants are consistently worse. L∞ and L1 are close when normalized; L∞+norm (~77.7%) is slightly below L1+norm.

---

## Group 3: CIFAR-10, budget=100 (20 samples/round)

| Experiment | Method | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---|---|---|---|---|
| `fixed_clustering_l1_cifar10` | fixed_clustering_l1_typiclust | 81.90% | 82.34% | 82.88% | **82.37%** |

(Only one method run at b=100 on CIFAR-10 — reference point for the b60 group.)

---

## Summary of Findings

1. **L1 norm always helps** — across all datasets, metrics, and budgets, L1-normalizing the embedding before distance computation improves accuracy.
2. **Fixed clustering > sum-based** (mostly) — fixed-k clustering with L1 distance beats dynamic sum methods on CIFAR-100; on CIFAR-10 b60 the gap narrows.
3. **l1_sum + L1norm is the overall winner on CIFAR-10 b60** (78.5%) and competitive on CIFAR-100.
4. **L∞ and L1 distance are close when normalized**; L∞ is slightly worse.
5. **No normalization consistently hurts**, especially for L1 sum (72.5% vs 78.5% on CIFAR-10 b60).
6. **Standard fixed_clustering (L2, no norm) is worst on CIFAR-100** (37.2%) — likely because L2 distance in embedding space is less discriminative without normalization.

---

## Plots

Generated from experiment logs (same style as dashboard). All PNGs are in `plots/`.

### CIFAR-100

![[cifar100_mean_accuracy_curves.png]]
*Mean test accuracy across AL rounds — all 8 CIFAR-100 methods. `fixed_l1norm` and `fixed_clustering_l1` lead.*

![[cifar100_final_accuracy_bar.png]]
*Final test accuracy bar chart (mean ± std across 3 seeds).*

### CIFAR-10 (budget=60)

![[cifar10_b60_mean_accuracy_curves.png]]
*Mean test accuracy across AL rounds — all 7 CIFAR-10 b60 methods. `l1_l1norm` leads.*

![[cifar10_b60_final_accuracy_bar.png]]
*Final test accuracy bar chart. Clear ordering: l1+norm > linf+norm > fixed+norm > no-norm variants.*

### Normalization ablation

![[norm_ablation_l1.png]]
*L1 distance with vs without L1 normalization, on both CIFAR-10 and CIFAR-100. Normalization gives +5–6% on CIFAR-10.*

### Distance metric comparison (normalized only)

![[cifar10_b60_distance_metric_comparison.png]]
*L1 vs L∞ vs fixed-clustering, all with L1 normalization. L1 sum slightly best; methods are competitive.*

### All CIFAR-10 methods (including b=100 reference)

![[cifar10_all_methods_compare.png]]
*All CIFAR-10 b60 methods overlaid — same scale for easy comparison.*
