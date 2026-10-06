# ContextCrunch Diagnostic Report

**Split:** all  |  **Question variant:** v1
**Cache:** data\replay  |  **Entries:** 943

## 1. Per-Kind Signal Distributions

| Kind | Signal | n | Mean | P10 | Median | P90 |
|------|--------|---|------|-----|--------|-----|
| final | verdict_confidence | 30 | 0.434 | 0.355 | 0.428 | 0.566 |
| final | p_drop | 30 | 0.328 | 0.243 | 0.335 | 0.422 |
| final | p_keep | 30 | 0.397 | 0.267 | 0.406 | 0.566 |
| final | p_truncate | 30 | 0.275 | 0.210 | 0.286 | 0.332 |
| final | essential | 30 | 0.349 | 0.217 | 0.329 | 0.612 |
| final | consumed | 30 | 0.682 | 0.485 | 0.705 | 0.841 |
| final | superseded | 30 | 0.304 | 0.090 | 0.227 | 0.718 |
| final | relevance | 30 | 1.694 | 1.521 | 1.719 | 1.818 |
| noise | verdict_confidence | 43 | 0.421 | 0.392 | 0.424 | 0.449 |
| noise | p_drop | 43 | 0.307 | 0.236 | 0.320 | 0.353 |
| noise | p_keep | 43 | 0.419 | 0.392 | 0.424 | 0.449 |
| noise | p_truncate | 43 | 0.274 | 0.240 | 0.255 | 0.328 |
| noise | essential | 43 | 0.416 | 0.319 | 0.409 | 0.513 |
| noise | consumed | 43 | 0.707 | 0.630 | 0.705 | 0.804 |
| noise | superseded | 43 | 0.333 | 0.157 | 0.282 | 0.596 |
| noise | relevance | 43 | 1.582 | 1.383 | 1.598 | 1.759 |
| risky | verdict_confidence | 53 | 0.451 | 0.403 | 0.460 | 0.505 |
| risky | p_drop | 53 | 0.300 | 0.262 | 0.301 | 0.328 |
| risky | p_keep | 53 | 0.449 | 0.395 | 0.460 | 0.505 |
| risky | p_truncate | 53 | 0.251 | 0.212 | 0.249 | 0.308 |
| risky | essential | 53 | 0.425 | 0.323 | 0.444 | 0.496 |
| risky | consumed | 53 | 0.808 | 0.698 | 0.835 | 0.875 |
| risky | superseded | 53 | 0.394 | 0.115 | 0.390 | 0.681 |
| risky | relevance | 53 | 1.644 | 1.486 | 1.671 | 1.793 |
| superseded | verdict_confidence | 17 | 0.447 | 0.386 | 0.444 | 0.490 |
| superseded | p_drop | 17 | 0.344 | 0.257 | 0.309 | 0.484 |
| superseded | p_keep | 17 | 0.386 | 0.201 | 0.433 | 0.490 |
| superseded | p_truncate | 17 | 0.270 | 0.215 | 0.274 | 0.326 |
| superseded | essential | 17 | 0.373 | 0.240 | 0.351 | 0.669 |
| superseded | consumed | 17 | 0.629 | 0.380 | 0.673 | 0.803 |
| superseded | superseded | 17 | 0.271 | 0.085 | 0.159 | 0.841 |
| superseded | relevance | 17 | 1.638 | 1.472 | 1.701 | 1.759 |
| trap | verdict_confidence | 42 | 0.396 | 0.360 | 0.396 | 0.433 |
| trap | p_drop | 42 | 0.345 | 0.307 | 0.333 | 0.413 |
| trap | p_keep | 42 | 0.369 | 0.284 | 0.370 | 0.433 |
| trap | p_truncate | 42 | 0.286 | 0.257 | 0.277 | 0.331 |
| trap | essential | 42 | 0.438 | 0.270 | 0.358 | 0.689 |
| trap | consumed | 42 | 0.731 | 0.613 | 0.777 | 0.815 |
| trap | superseded | 42 | 0.457 | 0.236 | 0.301 | 0.825 |
| trap | relevance | 42 | 1.707 | 1.601 | 1.689 | 1.816 |
| used_up | verdict_confidence | 20 | 0.471 | 0.442 | 0.480 | 0.522 |
| used_up | p_drop | 20 | 0.380 | 0.252 | 0.337 | 0.510 |
| used_up | p_keep | 20 | 0.349 | 0.200 | 0.386 | 0.503 |
| used_up | p_truncate | 20 | 0.272 | 0.231 | 0.277 | 0.303 |
| used_up | essential | 20 | 0.369 | 0.255 | 0.348 | 0.500 |
| used_up | consumed | 20 | 0.581 | 0.306 | 0.685 | 0.817 |
| used_up | superseded | 20 | 0.299 | 0.202 | 0.310 | 0.435 |
| used_up | relevance | 20 | 1.672 | 1.433 | 1.706 | 1.875 |

## 2. Signal Separation (AUC: removable vs must_keep)

| Signal | AUC | Strength |
|--------|-----|----------|
| verdict_confidence | 0.580 | weak |
| p_drop | 0.528 | weak |
| p_keep | 0.533 | weak |
| p_truncate | 0.467 | weak |
| essential | 0.566 | weak |
| consumed | 0.228 | weak |
| superseded | 0.358 | weak |
| relevance | 0.608 | weak |

## 3. Veto Funnel (removable items, aggressive profile)

| Gate | First Failing | Any Failing |
|------|---------------|-------------|
| hard_rules | 80 | 80 |
| verdict_drop | 0 | 62 |
| confidence | 0 | 80 |
| essential | 0 | 15 |
| relevance | 0 | 57 |
| junk_reason | 0 | 66 |

## 4. Danger Check (must_keep items wrongly dropped)

**Must-keep items evaluated:** 95
**Wrongly dropped:** 0
**Danger rate:** 0.0%

## 5. Summary

**Strong signals (AUC >= 0.85):** none
**Usable signals (0.70 <= AUC < 0.85):** none
**Weak signals (AUC < 0.70):** verdict_confidence, p_drop, p_keep, p_truncate, essential, consumed, superseded, relevance

**Main veto gate:** hard_rules (80 items blocked first)

**Recommendation:** Both essential and relevance signals are weak. The model struggles to identify irreplaceable content. Consider improving question wording (v2) or adding more training examples.