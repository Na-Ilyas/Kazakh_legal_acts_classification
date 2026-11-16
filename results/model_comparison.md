# Model comparison

Macro-F1 on val (2022), test (>=2023) and test_strict (test without near-duplicates of train titles). Latency = end-to-end inference for 1000 test titles (batch 64, 8 CPU threads, median of 3). `train peak MB` = peak CUDA memory while training for neural models (as logged by the baseline module for TF-IDF models); `infer RAM MB` = RAM growth of a fresh process while loading the model and predicting on CPU.

## Task `area`: main models

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_word_lr | tfidf | 0.5861 | 0.5915 | 0.5624 | 0.7793 | 3.81M | 20.8 | 30.7 | – | 61.2 | – | 6.7 |
| char_1nn | baseline | 0.5555 | 0.5363 | 0.4875 | 0.7379 | 47.53M | 362.9 | 2,801.2 | – | 2,275.5 | – | 9.9 |
| majority_dummy | baseline | 0.0197 | 0.0186 | 0.0116 | 0.4411 | 0 | 0.0 | 0.0 | – | 80.9 | – | 0.7 |

### Task `area`: ablations and other variants

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_word_char_lr | tfidf | 0.5816 | 0.5949 | 0.5658 | 0.7801 | 7.13M | 36.6 | 245.9 | – | 1,649.3 | 157.1 | 49.2 |
| tfidf_stem_lr | tfidf | 0.5773 | 0.5898 | 0.5604 | 0.7794 | 2.86M | 15.3 | 49.6 | – | 45.9 | 62.4 | 7.0 |
| tfidf_char_lr | tfidf | 0.5707 | 0.5834 | 0.5552 | 0.7754 | 3.32M | 15.9 | 173.7 | – | 527.5 | 76.2 | 41.9 |

## Task `form`: main models

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5_small_ft | pretrained | 0.8656 | 0.8415 | 0.8462 | 0.9651 | 117.66M | 465.1 | 3,824.5 | 118.3 | 3,396.4 | 815.0 | 104.5 |
| tfidf_word_char_lr | tfidf | 0.8405 | 0.8257 | 0.8201 | 0.9626 | 1.51M | 15.2 | 192.9 | – | 1,650.4 | 136.8 | 30.1 |
| char_1nn | baseline | 0.6747 | 0.6697 | 0.6322 | 0.9065 | 47.53M | 362.9 | 2,801.2 | – | 2,275.5 | – | 9.9 |
| majority_dummy | baseline | 0.0471 | 0.0451 | 0.0625 | 0.1876 | 0 | 0.0 | 0.1 | – | 75.7 | – | 0.7 |

### Task `form`: ablations and other variants

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_word_lr | tfidf | 0.8309 | 0.8030 | 0.8007 | 0.9575 | 0.81M | 9.3 | 23.0 | – | 49.7 | – | 4.5 |
| tfidf_stem_lr | tfidf | 0.8236 | 0.7943 | 0.7815 | 0.9584 | 0.61M | 6.7 | 33.7 | – | 37.4 | 49.4 | 4.2 |
| tfidf_char_lr | tfidf | 0.8006 | 0.7783 | 0.7745 | 0.9559 | 0.70M | 5.9 | 114.6 | – | 517.6 | – | 23.2 |

## Seed variance of the tiny transformer

Same data and hyper-parameters, seeds [1, 2, 3] (weight init + batch order). Differences between tiny-transformer variants smaller than about 2 std are within seed noise.

| task | arch | loss | f1_val mean ± std | f1_test mean ± std | f1_test_strict mean ± std |
|---|---|---|---|---|---|
| area | 2L | smooth | 0.5607 ± 0.0023 | 0.5802 ± 0.0099 | 0.5553 ± 0.0108 |
| area | 4L | smooth | 0.5624 ± 0.0022 | 0.5784 ± 0.0067 | 0.5500 ± 0.0089 |
| form | 2L | smooth | 0.8259 ± 0.0037 | 0.8099 ± 0.0050 | 0.8002 ± 0.0057 |
| form | 4L | smooth | 0.8332 ± 0.0032 | 0.8019 ± 0.0061 | 0.7926 ± 0.0037 |
