# Model comparison

Macro-F1 on val (2022), test (>=2023) and test_strict (test without near-duplicates of train titles). Latency = end-to-end inference for 1000 test titles (batch 64, 8 CPU threads, median of 3). `train peak MB` = peak CUDA memory while training for neural models (as logged by the baseline module for TF-IDF models); `infer RAM MB` = RAM growth of a fresh process while loading the model and predicting on CPU. Family `posthoc` (ensembles, prior-shift adjustment): F1 val is measured on val half B only (half A estimated the class prior), and latency/size/params cover all member models. `*_trainval` = refit on train+val with the val-selected config: its F1 val is NOT a held-out score and it is never selected on val. Main table = the val-best variant of every group (baseline/tfidf, tiny transformer, e5, kaz-roberta, other pretrained, frozen e5, seq2seq, ensemble, posthoc) + fixed references (tiny_transformer_4L, majority floor, e5_small_ft, pruned models, mT5); every other variant is in `results/ablations.md`. CIs: paired bootstrap over test rows (1000 resamples); Δ = macro-F1 minus the val-best TF-IDF model of the task, `tie` when its 95% CI contains 0. Ensembles / prior-adjusted models are listed separately as reference: their only val score is on val half B, which is in-sample for their weights/λ, so they are never ranked against single models on validation.

## Task `area`: main models (val-best variant per group)

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kaz_roberta_llrd_rec | pretrained kaz-roberta | 0.6162 | – | 0.6285 | [0.613, 0.642] | +0.046 [+0.032, +0.058] better | 0.6080 | 0.7988 | 83.48M | 323.6 | 9,775.9 | 250.8 | 3,173.2 | 950.1 | 119.0 |
| kaz_roberta_ft | pretrained kaz-roberta | 0.6013 | – | 0.6223 | [0.609, 0.635] | +0.040 [+0.029, +0.051] better | 0.6011 | 0.7937 | 83.48M | 323.6 | 9,692.6 | 246.7 | 3,170.7 | 926.3 | 87.4 |
| kaz_roberta_kkprune_ft | pretrained kaz-roberta | 0.6000 | – | 0.6213 | [0.608, 0.634] | +0.038 [+0.028, +0.050] better | 0.6003 | 0.7936 | 61.61M | 240.4 | 9,631.3 | 247.5 | 2,843.0 | 936.5 | 65.2 |
| kazbert_ft | pretrained kazbert | 0.6239 | – | 0.6201 | [0.607, 0.632] | +0.037 [+0.027, +0.049] better | 0.5968 | 0.7993 | 110.64M | 422.9 | 23,375.2 | – | 5,217.7 | – | 212.9 |
| e5_small_ft | pretrained e5 | 0.5836 | – | 0.6010 | [0.588, 0.613] | +0.018 [+0.005, +0.030] better | 0.5792 | 0.7883 | 117.67M | 465.2 | 3,506.4 | 108.3 | 3,408.0 | 852.0 | 128.6 |
| e5_small_kkprune_ft | pretrained e5 | 0.5805 | – | 0.5997 | [0.587, 0.612] | +0.017 [+0.004, +0.029] better | 0.5790 | 0.7878 | 25.53M | 98.4 | 8,051.2 | – | 1,998.1 | – | 77.0 |
| tiny_transformer_4L_mlm_kdrob_a50_T2 | tiny + KD (kaz-roberta teacher) | 0.5967 | – | 0.5979 | [0.585, 0.610] | +0.015 [+0.004, +0.026] better | 0.5759 | 0.7903 | 5.24M | 20.8 | 515.5 | 50.2 | 1,100.3 | 124.4 | 34.0 |
| tiny_transformer_4L_mlm_kd_a50_T2 | tiny + MLM + KD (teacher e5) | 0.5922 | – | 0.5974 | [0.584, 0.610] | +0.015 [+0.002, +0.027] better | 0.5797 | 0.7904 | 5.24M | 20.8 | 502.3 | 49.9 | 1,100.3 | 124.5 | 33.9 |
| tiny_transformer_4L_mlm | tiny + MLM | 0.5857 | – | 0.5921 | [0.579, 0.605] | +0.009 [-0.002, +0.021] tie | 0.5695 | 0.7832 | 5.24M | 20.8 | 508.6 | 50.3 | 1,090.5 | 129.6 | 33.4 |
| tiny_transformer_4L | tiny (from scratch) | 0.5890 | – | 0.5836 | [0.570, 0.595] | +0.001 [-0.011, +0.013] tie | 0.5643 | 0.7776 | 5.24M | 20.8 | 502.5 | 49.9 | 1,070.5 | 126.6 | 34.7 |
| tiny_transformer_4L_recency_tau8 | tiny + recency weighting | 0.6003 | – | 0.5832 | [0.569, 0.596] | +0.001 [-0.012, +0.013] tie | 0.5580 | 0.7787 | 5.24M | 20.8 | 517.5 | 49.7 | 1,071.1 | 126.3 | 33.4 |
| tfidf_word_char_lr | baseline/tfidf | 0.5985 | – | 0.5827 | [0.570, 0.595] | +0.000 [+0.000, +0.000] baseline | 0.5555 | 0.7800 | 7.13M | 36.6 | 234.6 | – | 582.3 | 157.6 | 47.3 |
| tiny_transformer_4L_kd_a50_T2 | tiny + KD (teacher e5) | 0.5975 | – | 0.5810 | [0.568, 0.594] | -0.002 [-0.014, +0.010] tie | 0.5536 | 0.7805 | 5.24M | 20.8 | 514.2 | 49.7 | 1,080.4 | 127.1 | 24.2 |
| tiny_transformer_4L_do30 | tiny (from scratch) | 0.5907 | – | 0.5798 | [0.568, 0.592] | -0.003 [-0.015, +0.010] tie | 0.5565 | 0.7749 | 5.24M | 20.8 | 493.7 | – | 1,070.5 | 127.4 | 38.1 |
| e5_frozen_lr | pretrained_frozen | 0.5277 | – | 0.5324 | [0.519, 0.545] | -0.050 [-0.063, -0.038] worse | 0.5005 | 0.7485 | 117.67M | 448.9 | 3,520.5 | 114.4 | 1,230.1 | 871.6 | 69.8 |
| majority_dummy | baseline/tfidf | 0.0197 | – | 0.0186 | [0.018, 0.019] | -0.564 [-0.576, -0.551] worse | 0.0116 | 0.4411 | 0 | 0.0 | 0.3 | – | 80.9 | 5.9 | 0.7 |

### Task `area`: ensembles, post-hoc adjustment and LLM teacher - REFERENCE ONLY (ensembles/post-hoc: F1 val½B is in-sample for weights/λ; >120M rows exceed the budget; the LLM is a distillation teacher only)

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble_tfidf_tiny_e5_roberta (>120M, reference) | ensemble | – | 0.6407 | 0.6215 | [0.608, 0.633] | +0.039 [+0.030, +0.047] better | 0.6035 | 0.8019 | 213.51M | 846.2 | 13,960.2 | 660.5 | – | 1,519.4 | 328.3 |
| llm_teacher_qwen3_1p7b (teacher only, >120M, reference) | llm_teacher | 0.5486 | – | 0.5862 | [0.571, 0.599] | +0.004 [-0.011, +0.017] tie | 0.5691 | 0.7712 | 1727.13M | 3,317.5 | 2,425.4 | – | 15,615.2 | – | 528.2 |
| tiny_transformer_4L_recency_tau8_prioradj | posthoc | – | 0.6386 | 0.5836 | [0.570, 0.596] | +0.001 [-0.011, +0.013] tie | 0.5576 | 0.7792 | 5.24M | 20.8 | 536.2 | 51.2 | – | 123.1 | 33.4 |

### Task `area`: refits on train+val (no held-out val score, never selected on val)

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kaz_roberta_llrd_rec_trainval | pretrained kaz-roberta | 0.8102 | – | 0.6345 | [0.620, 0.646] | +0.052 [+0.041, +0.062] better | 0.6163 | 0.8076 | 83.48M | 323.6 | 9,841.1 | 248.2 | 3,177.2 | 926.5 | 101.7 |
| tiny_transformer_4L_recency_tau8_trainval | tiny + recency weighting | 0.7696 | – | 0.5880 | [0.574, 0.601] | +0.005 [-0.006, +0.016] tie | 0.5592 | 0.7886 | 5.24M | 20.8 | 523.4 | 49.7 | 1,072.6 | 124.1 | 21.0 |

## Task `form`: main models (val-best variant per group)

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kazbert_ft | pretrained kazbert | 0.8794 | – | 0.8601 | [0.832, 0.886] | +0.040 [+0.020, +0.062] better | 0.8662 | 0.9702 | 110.62M | 422.8 | 25,676.2 | 527.7 | 5,234.2 | 990.3 | 125.9 |
| kaz_roberta_ft | pretrained kaz-roberta | 0.8830 | – | 0.8528 | [0.822, 0.881] | +0.033 [+0.014, +0.053] better | 0.8675 | 0.9659 | 83.46M | 323.5 | 9,707.8 | 246.5 | 3,170.5 | 926.2 | 71.0 |
| kaz_roberta_kkprune_ft | pretrained kaz-roberta | 0.8843 | – | 0.8491 | [0.818, 0.878] | +0.029 [+0.011, +0.050] better | 0.8633 | 0.9656 | 61.59M | 240.3 | 9,755.5 | 248.9 | 2,836.6 | 889.6 | 66.2 |
| e5_small_kkprune_ft | pretrained e5 | 0.8628 | – | 0.8448 | [0.816, 0.874] | +0.025 [+0.004, +0.048] better | 0.8521 | 0.9653 | 25.52M | 98.3 | 8,554.8 | – | 1,991.0 | – | 82.8 |
| e5_small_ft | pretrained e5 | 0.8656 | – | 0.8415 | [0.813, 0.867] | +0.021 [+0.005, +0.040] better | 0.8462 | 0.9651 | 117.66M | 465.1 | 3,471.0 | 111.7 | 3,396.4 | 881.9 | 104.5 |
| kaz_roberta_llrd_rec | pretrained kaz-roberta | 0.8978 | – | 0.8414 | [0.813, 0.869] | +0.021 [+0.007, +0.037] better | 0.8478 | 0.9649 | 83.46M | 323.5 | 10,874.8 | 273.3 | 3,168.9 | 949.3 | 124.6 |
| tiny_transformer_4L_mlm_kdrob_a50_T2 | tiny + KD (kaz-roberta teacher) | 0.8683 | – | 0.8273 | [0.801, 0.854] | +0.007 [-0.012, +0.026] tie | 0.8286 | 0.9637 | 5.23M | 20.7 | 540.7 | 55.3 | 1,091.8 | 122.4 | 39.2 |
| tiny_transformer_4L_mlm_kd_a90_T2 | tiny + MLM + KD (teacher e5) | 0.8647 | – | 0.8252 | [0.799, 0.852] | +0.005 [-0.015, +0.026] tie | 0.8280 | 0.9655 | 5.23M | 20.7 | 538.3 | 52.1 | 1,092.5 | 125.0 | 53.3 |
| tfidf_word_char_lr | baseline/tfidf | 0.8576 | – | 0.8202 | [0.793, 0.847] | +0.000 [+0.000, +0.000] baseline | 0.8253 | 0.9620 | 1.51M | 15.2 | 189.6 | – | 561.0 | 135.6 | 27.4 |
| tiny_transformer_4L_mlm | tiny + MLM | 0.8514 | – | 0.8188 | [0.792, 0.844] | -0.002 [-0.022, +0.019] tie | 0.8173 | 0.9615 | 5.23M | 20.7 | 526.9 | 50.6 | 1,090.3 | 120.0 | 38.5 |
| tiny_transformer_2L_d128 | tiny (from scratch) | 0.8405 | – | 0.8099 | [0.782, 0.838] | -0.010 [-0.035, +0.015] tie | 0.8090 | 0.9582 | 1.43M | 6.2 | 146.3 | – | 347.2 | 71.1 | 15.4 |
| tiny_transformer_4L_kd_a90_T2 | tiny + KD (teacher e5) | 0.8716 | – | 0.7992 | [0.774, 0.825] | -0.021 [-0.047, +0.004] tie | 0.7939 | 0.9623 | 5.23M | 20.7 | 531.7 | 49.8 | 1,072.5 | 125.0 | 58.1 |
| tiny_transformer_4L | tiny (from scratch) | 0.8256 | – | 0.7931 | [0.765, 0.822] | -0.027 [-0.048, -0.008] worse | 0.7905 | 0.9564 | 5.23M | 20.7 | 520.1 | 49.9 | 1,070.4 | 125.7 | 39.6 |
| tiny_transformer_4L_recency_tau8 | tiny + recency weighting | 0.8443 | – | 0.7921 | [0.769, 0.815] | -0.028 [-0.050, -0.009] worse | 0.7824 | 0.9558 | 5.23M | 20.7 | 524.7 | 51.0 | 1,071.0 | 122.8 | 23.9 |
| mt5_small_kkprune_text | seq2seq | 0.8213 | – | 0.7910 | [0.761, 0.820] | -0.029 [-0.063, +0.004] tie | 0.7956 | 0.9575 | 58.73M | 225.0 | 20,552.0 | – | 3,560.9 | – | 179.3 |
| e5_frozen_lr | pretrained_frozen | 0.6634 | – | 0.6992 | [0.675, 0.725] | -0.121 [-0.149, -0.097] worse | 0.6981 | 0.9096 | 117.66M | 448.9 | 3,513.1 | 112.0 | 1,230.1 | 856.1 | 24.4 |
| majority_dummy | baseline/tfidf | 0.0471 | – | 0.0451 | [0.044, 0.046] | -0.775 [-0.802, -0.748] worse | 0.0625 | 0.1876 | 0 | 0.0 | 0.3 | – | 75.7 | 5.7 | 0.7 |

### Task `form`: ensembles, post-hoc adjustment and LLM teacher - REFERENCE ONLY (ensembles/post-hoc: F1 val½B is in-sample for weights/λ; >120M rows exceed the budget; the LLM is a distillation teacher only)

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble_tfidf_tiny_e5_roberta (>120M, reference) | ensemble | – | 0.9314 | 0.8497 | [0.819, 0.880] | +0.029 [+0.017, +0.045] better | 0.8607 | 0.9675 | 207.86M | 824.6 | 14,330.4 | 644.8 | – | 1,493.9 | 314.6 |
| llm_teacher_qwen3_1p7b (teacher only, >120M, reference) | llm_teacher | 0.7536 | – | 0.7747 | [0.743, 0.807] | -0.045 [-0.068, -0.026] worse | 0.7745 | 0.9482 | 1727.03M | 3,317.3 | 2,423.6 | – | 15,613.5 | – | 302.0 |

### Task `form`: refits on train+val (no held-out val score, never selected on val)

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny_transformer_4L_kd_a90_T2_trainval | tiny + KD (teacher e5) | 0.8658 | – | 0.8051 | [0.784, 0.829] | -0.015 [-0.040, +0.008] tie | 0.7992 | 0.9641 | 5.23M | 20.7 | 508.6 | 50.0 | 1,074.9 | 125.8 | 58.0 |

## Seed variance of the tiny transformer

Same data and hyper-parameters, seeds [1, 2, 3] (weight init + batch order). Differences between tiny-transformer variants smaller than about 2 std are within seed noise.

| task | arch | loss | f1_val mean ± std | f1_test mean ± std | f1_test_strict mean ± std |
|---|---|---|---|---|---|
| area | 2L | smooth | 0.5767 ± 0.0025 | 0.5789 ± 0.0093 | 0.5529 ± 0.0105 |
| area | 4L | smooth | 0.5782 ± 0.0022 | 0.5738 ± 0.0021 | 0.5437 ± 0.0053 |
| form | 2L | smooth | 0.8289 ± 0.0052 | 0.7912 ± 0.0137 | 0.7900 ± 0.0131 |
| form | 4L | smooth | 0.8268 ± 0.0097 | 0.7886 ± 0.0121 | 0.7834 ± 0.0188 |
