# Model comparison

Macro-F1 on val (2022), test (>=2023) and test_strict (test without near-duplicates of train titles). Latency = end-to-end inference for 1000 test titles (batch 64, 8 CPU threads, median of 3). `train peak MB` = peak CUDA memory while training for neural models (as logged by the baseline module for TF-IDF models); `infer RAM MB` = RAM growth of a fresh process while loading the model and predicting on CPU.

## Task `area`

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5_small_ft | pretrained | 0.5630 | 0.6001 | 0.5791 | 0.7877 | 117.67M | 465.2 | 3,945.3 | 132.7 | 3,407.6 | 869.2 | 117.4 |
| tiny_transformer_2L_do30 | tiny_transformer | 0.5685 | 0.5950 | 0.5702 | 0.7779 | 3.66M | 14.7 | 322.0 | 48.6 | 628.0 | 115.4 | 19.5 |
| tfidf_word_char_lr | tfidf | 0.5816 | 0.5949 | 0.5704 | 0.7801 | 7.13M | 36.6 | 334.7 | – | 0.1 | 157.2 | 76.5 |
| tfidf_word_lr | tfidf | 0.5711 | 0.5937 | 0.5684 | 0.7795 | 3.81M | 20.8 | 174.8 | – | 61.2 | 85.7 | 9.6 |
| tiny_transformer_4L_mlm | tiny_transformer | 0.5696 | 0.5929 | 0.5721 | 0.7831 | 5.24M | 20.8 | 790.4 | 68.3 | 1,091.5 | 126.4 | 35.4 |
| tfidf_stem_lr | tfidf | 0.5773 | 0.5898 | 0.5671 | 0.7794 | 2.86M | 15.3 | 94.7 | – | 0.0 | 62.1 | 6.8 |
| tiny_transformer_4L | tiny_transformer | 0.5725 | 0.5850 | 0.5638 | 0.7775 | 5.24M | 20.8 | 527.6 | 55.0 | 1,070.6 | 129.5 | 39.8 |
| tfidf_char_lr | tfidf | 0.5707 | 0.5834 | 0.5597 | 0.7754 | 3.32M | 15.9 | 353.0 | – | 0.1 | 76.4 | 68.7 |
| tiny_transformer_4L_do30 | tiny_transformer | 0.5740 | 0.5814 | 0.5592 | 0.7748 | 5.24M | 20.8 | 835.7 | 59.8 | 1,070.6 | 126.2 | 68.9 |
| tiny_transformer_2L | tiny_transformer | 0.5731 | 0.5777 | 0.5501 | 0.7746 | 3.66M | 14.7 | 301.2 | 48.1 | 628.0 | 114.7 | 28.9 |
| tiny_transformer_2L_d128 | tiny_transformer | 0.5709 | 0.5743 | 0.5457 | 0.7745 | 1.44M | 6.3 | 138.5 | 48.5 | 347.2 | 69.8 | 13.3 |
| tiny_transformer_4L_d128 | tiny_transformer | 0.5608 | 0.5697 | 0.5456 | 0.7732 | 1.83M | 7.8 | 340.0 | 63.9 | 566.0 | 72.9 | 25.2 |
| char_1nn | baseline | 0.5386 | 0.5375 | 0.5039 | 0.7393 | 49.05M | 374.5 | 3,672.9 | – | 2,190.3 | – | 12.1 |
| e5_frozen_lr | pretrained_frozen | 0.5123 | 0.5338 | 0.5073 | 0.7484 | 117.67M | 448.9 | 4,192.9 | 136.1 | 1,231.1 | 892.5 | 59.0 |
| majority_dummy | baseline | 0.0197 | 0.0185 | 0.0202 | 0.4398 | 0 | 0.0 | 0.3 | – | 80.9 | 6.0 | 0.7 |

## Task `form`

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5_small_ft | pretrained | 0.8648 | 0.8519 | 0.8497 | 0.9654 | 117.66M | 465.1 | 3,958.4 | 119.1 | 3,406.3 | 885.4 | 107.7 |
| tiny_transformer_4L_mlm | tiny_transformer | 0.8575 | 0.8294 | 0.8202 | 0.9605 | 5.23M | 20.7 | 526.0 | 51.8 | 1,091.3 | 125.9 | 35.7 |
| tfidf_word_char_lr | tfidf | 0.8405 | 0.8257 | 0.8189 | 0.9626 | 1.51M | 15.2 | 205.1 | – | 0.0 | 135.6 | 56.2 |
| tiny_transformer_4L | tiny_transformer | 0.8268 | 0.8256 | 0.8188 | 0.9516 | 5.23M | 20.7 | 592.3 | 55.9 | 1,070.5 | 130.7 | 31.6 |
| tiny_transformer_2L_do30 | tiny_transformer | 0.8242 | 0.8256 | 0.8172 | 0.9509 | 3.65M | 14.7 | 341.9 | 63.3 | 627.9 | 113.5 | 16.9 |
| tiny_transformer_4L_do30 | tiny_transformer | 0.8306 | 0.8143 | 0.8029 | 0.9529 | 5.23M | 20.7 | 536.9 | 52.0 | 1,070.5 | 126.7 | 30.8 |
| tfidf_word_lr | tfidf | 0.8326 | 0.8116 | 0.8051 | 0.9573 | 0.81M | 9.3 | 37.4 | – | 49.9 | 72.0 | 5.7 |
| tiny_transformer_2L | tiny_transformer | 0.8360 | 0.8094 | 0.7983 | 0.9510 | 3.65M | 14.7 | 407.0 | 58.4 | 627.9 | 121.7 | 16.7 |
| tiny_transformer_4L_d128 | tiny_transformer | 0.8144 | 0.8056 | 0.7936 | 0.9531 | 1.83M | 7.8 | 237.6 | 53.0 | 565.9 | 72.6 | 18.6 |
| tfidf_char_lr | tfidf | 0.8016 | 0.8021 | 0.7901 | 0.9547 | 0.70M | 5.9 | 118.9 | – | 517.9 | – | 28.9 |
| tiny_transformer_2L_d128 | tiny_transformer | 0.8318 | 0.8010 | 0.7905 | 0.9555 | 1.43M | 6.2 | 162.5 | 61.9 | 347.2 | 71.7 | 15.8 |
| tiny_transformer_4L_bpe4k | tiny_transformer | 0.8532 | 0.7945 | 0.7794 | 0.9528 | 4.21M | 16.4 | 679.3 | 56.3 | 1,255.1 | 137.9 | 53.8 |
| tfidf_stem_lr | tfidf | 0.8236 | 0.7943 | 0.7844 | 0.9584 | 0.61M | 6.7 | 38.2 | – | 0.0 | 48.5 | 4.4 |
| e5_frozen_lr | pretrained_frozen | 0.6666 | 0.7178 | 0.7163 | 0.9094 | 117.66M | 448.9 | 3,858.9 | 129.1 | 1,231.2 | 881.9 | 26.9 |
| char_1nn | baseline | 0.6751 | 0.6802 | 0.6548 | 0.9058 | 49.05M | 374.5 | 3,672.9 | – | 2,190.3 | – | 12.1 |
| majority_dummy | baseline | 0.0472 | 0.0453 | 0.0422 | 0.1882 | 0 | 0.0 | 0.3 | – | 75.7 | 5.7 | 0.7 |

## Seed variance of the tiny transformer

Same data and hyper-parameters, seeds [1, 2, 3] (weight init + batch order). Differences between tiny-transformer variants smaller than about 2 std are within seed noise.

| task | arch | loss | f1_val mean ± std | f1_test mean ± std | f1_test_strict mean ± std |
|---|---|---|---|---|---|
| area | 2L | smooth | 0.5607 ± 0.0023 | 0.5802 ± 0.0099 | 0.5553 ± 0.0108 |
| area | 4L | smooth | 0.5624 ± 0.0022 | 0.5784 ± 0.0067 | 0.5500 ± 0.0089 |
| form | 2L | smooth | 0.8259 ± 0.0037 | 0.8099 ± 0.0050 | 0.8002 ± 0.0057 |
| form | 4L | smooth | 0.8332 ± 0.0032 | 0.8019 ± 0.0061 | 0.7926 ± 0.0037 |
