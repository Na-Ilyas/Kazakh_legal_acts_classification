# Model comparison

Macro-F1 on val (2022), test (>=2023) and test_strict (test without near-duplicates of train titles). Latency = end-to-end inference for 1000 test titles (batch 64, 8 CPU threads, median of 3). `train peak MB` = peak CUDA memory while training for neural models (as logged by the baseline module for TF-IDF models); `infer RAM MB` = RAM growth of a fresh process while loading the model and predicting on CPU. Family `posthoc` (ensembles, prior-shift adjustment): F1 val is measured on val half B only (half A estimated the class prior), and latency/size/params cover all member models. `*_trainval` = refit on train+val with the val-selected config: its F1 val is NOT a held-out score and it is never selected on val. Main table = best model of each family on validation (+ the from-scratch tiny_transformer_4L and the majority floor); every other variant is listed in the ablation table.

## Task `area`: main models

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kaz_roberta_llrd_rec_trainval | pretrained | 0.8102 | 0.6345 | 0.6163 | 0.8076 | 83.48M | 323.6 | 10,018.5 | – | 3,177.2 | – | 101.7 |
| kaz_roberta_llrd_rec | pretrained | 0.6162 | 0.6285 | 0.6080 | 0.7988 | 83.48M | 323.6 | 10,018.5 | – | 3,173.2 | – | 119.0 |
| kaz_roberta_ft | pretrained | 0.6013 | 0.6223 | 0.6011 | 0.7937 | 83.48M | 323.6 | 9,848.9 | – | 3,170.7 | – | 87.4 |
| kaz_roberta_kkprune_ft | pretrained | 0.6000 | 0.6213 | 0.6003 | 0.7936 | 61.61M | 240.4 | 10,068.3 | – | 2,843.0 | – | 65.2 |
| e5_small_ft | pretrained | 0.5836 | 0.6010 | 0.5792 | 0.7883 | 117.67M | 465.2 | 3,880.5 | 139.9 | 3,408.0 | 731.7 | 128.6 |
| e5_small_kkprune_ft | pretrained | 0.5805 | 0.5997 | 0.5790 | 0.7878 | 25.53M | 98.4 | 8,051.2 | – | 1,998.1 | – | 77.0 |
| tiny_transformer_4L | tiny_transformer | 0.5890 | 0.5836 | 0.5643 | 0.7776 | 5.24M | – | – | – | 1,070.5 | – | 34.7 |
| tiny_transformer_4L_recency_tau8_prioradj | posthoc | 0.6299 | 0.5832 | 0.5570 | 0.7792 | 5.24M | 20.8 | 494.7 | 51.0 | – | – | 33.4 |
| tiny_transformer_4L_recency_tau8 | tiny_transformer | 0.6003 | 0.5832 | 0.5580 | 0.7787 | 5.24M | 20.8 | 494.7 | 51.0 | 1,071.1 | 124.7 | 33.4 |
| tfidf_word_char_lr | tfidf | 0.5985 | 0.5827 | 0.5555 | 0.7800 | 7.13M | 36.6 | 162.0 | – | 582.3 | – | 47.3 |
| tiny_transformer_4L_recency_tau8_trainval | tiny_transformer | 0.7116 | 0.5795 | 0.5550 | 0.7870 | 5.24M | 20.8 | 501.7 | 52.4 | 1,072.5 | 125.9 | 20.7 |
| char_1nn | baseline | 0.5555 | 0.5363 | 0.4875 | 0.7379 | 47.53M | 362.9 | 2,801.2 | – | 2,275.5 | – | 9.9 |
| e5_frozen_lr | pretrained_frozen | 0.5277 | 0.5324 | 0.5005 | 0.7485 | 117.67M | 448.9 | 5,533.2 | 153.9 | 1,230.1 | 877.5 | 69.8 |
| majority_dummy | baseline | 0.0197 | 0.0186 | 0.0116 | 0.4411 | 0 | 0.0 | 0.0 | – | 80.9 | – | 0.7 |

### Task `area`: ablations and other variants

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kaz_roberta_combo | pretrained | 0.6131 | 0.6330 | 0.6135 | 0.7975 | 83.48M | 323.6 | 10,018.5 | – | 3,173.2 | – | 116.2 |
| kaz_roberta_llrd | pretrained | 0.6065 | 0.6216 | 0.6000 | 0.7955 | 83.48M | 323.6 | 10,018.5 | – | 3,175.3 | – | 119.6 |
| kaz_roberta_llrd_cw | pretrained | 0.6066 | 0.6193 | 0.5980 | 0.7913 | 83.48M | 323.6 | 10,018.5 | – | 3,172.6 | – | 120.3 |
| kaz_roberta_llrd_ls | pretrained | 0.6046 | 0.6181 | 0.5946 | 0.7951 | 83.48M | 323.6 | 10,018.5 | – | 3,172.6 | – | 123.6 |
| kaz_roberta_ep5 | pretrained | 0.6025 | 0.6093 | 0.5861 | 0.7920 | 83.48M | 323.6 | 10,018.5 | – | 3,175.4 | – | 119.5 |
| ensemble_tfidf_tiny_e5 | posthoc | 0.6180 | 0.6040 | 0.5839 | 0.7916 | 130.04M | 522.6 | 4,537.2 | 190.9 | – | – | 209.3 |
| tiny_transformer_4L_mlm_kd_a50_T2 | tiny_transformer | 0.5922 | 0.5974 | 0.5797 | 0.7904 | 5.24M | 20.8 | 507.4 | 51.2 | 1,100.3 | 125.8 | 33.9 |
| tiny_transformer_2L_do30 | tiny_transformer | 0.5842 | 0.5937 | 0.5681 | 0.7780 | 3.66M | – | – | – | 628.0 | – | 26.6 |
| tiny_transformer_4L_kd_a90_T2 | tiny_transformer | 0.5786 | 0.5936 | 0.5671 | 0.7861 | 5.24M | 20.8 | 512.9 | 54.1 | 1,080.4 | 125.0 | 48.4 |
| tiny_transformer_4L_mlm | tiny_transformer | 0.5857 | 0.5921 | 0.5695 | 0.7832 | 5.24M | 20.8 | 530.2 | 54.1 | 1,090.5 | 124.1 | 33.4 |
| ablation_tfidf_word_fold_num | tfidf | 0.5861 | 0.5915 | 0.5624 | 0.7793 | 3.81M | – | – | – | – | – | 3.0 |
| tfidf_word_lr | tfidf | 0.5861 | 0.5915 | 0.5624 | 0.7793 | 3.81M | 20.8 | 30.7 | – | 61.2 | – | 6.7 |
| tfidf_stem_lr | tfidf | 0.5942 | 0.5905 | 0.5643 | 0.7806 | 2.86M | 15.3 | 32.6 | – | 45.9 | – | 8.3 |
| ensemble_tfidf_tiny_e5_prioradj | posthoc | 0.6248 | 0.5879 | 0.5615 | 0.7879 | 130.04M | 522.6 | 4,537.2 | 190.9 | – | – | 209.3 |
| ablation_tfidf_word_nofold_num | tfidf | 0.5860 | 0.5857 | 0.5558 | 0.7782 | 4.05M | – | – | – | – | – | 3.1 |
| ablation_tfidf_word_raw | tfidf | 0.5812 | 0.5821 | 0.5522 | 0.7754 | 4.79M | – | – | – | – | – | 3.1 |
| tiny_transformer_4L_kd_a50_T2 | tiny_transformer | 0.5975 | 0.5810 | 0.5536 | 0.7805 | 5.24M | 20.8 | 565.4 | 51.5 | 1,080.4 | 125.8 | 24.2 |
| ablation_tfidf_word_fold_digits | tfidf | 0.5784 | 0.5810 | 0.5537 | 0.7760 | 4.55M | – | – | – | – | – | 3.1 |
| tiny_transformer_4L_kd_a90_T4 | tiny_transformer | 0.5735 | 0.5808 | 0.5573 | 0.7835 | 5.24M | 20.8 | 535.0 | 52.6 | 1,080.4 | 131.2 | 53.3 |
| tiny_transformer_4L_do30 | tiny_transformer | 0.5907 | 0.5798 | 0.5565 | 0.7749 | 5.24M | – | – | – | 1,070.5 | – | 38.1 |
| tiny_transformer_2L | tiny_transformer | 0.5896 | 0.5764 | 0.5482 | 0.7746 | 3.66M | – | – | – | 628.0 | – | 24.7 |
| tiny_transformer_4L_recency_tau3 | tiny_transformer | 0.5968 | 0.5748 | 0.5478 | 0.7780 | 5.24M | 20.8 | 521.4 | 51.5 | 1,071.1 | 126.3 | 33.3 |
| tiny_transformer_2L_d128 | tiny_transformer | 0.5869 | 0.5727 | 0.5419 | 0.7745 | 1.44M | – | – | – | 347.2 | – | 12.2 |
| tiny_transformer_4L_d128 | tiny_transformer | 0.5768 | 0.5682 | 0.5433 | 0.7733 | 1.83M | – | – | – | 566.0 | – | 20.4 |
| tfidf_char_lr | tfidf | 0.5712 | 0.5569 | 0.5264 | 0.7613 | 3.32M | 15.9 | 128.4 | – | 527.6 | – | 47.8 |

## Task `form`: main models

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble_tfidf_tiny_e5_prioradj | posthoc | 0.9190 | 0.8552 | 0.8704 | 0.9681 | 124.40M | 501.1 | 4,494.3 | 170.6 | – | – | 190.0 |
| kaz_roberta_ft | pretrained | 0.8830 | 0.8528 | 0.8675 | 0.9659 | 83.46M | 323.5 | 10,217.7 | – | 3,170.5 | – | 71.0 |
| kaz_roberta_kkprune_ft | pretrained | 0.8843 | 0.8491 | 0.8633 | 0.9656 | 61.59M | 240.3 | 10,036.6 | – | 2,836.6 | – | 66.2 |
| e5_small_kkprune_ft | pretrained | 0.8628 | 0.8448 | 0.8521 | 0.9653 | 25.52M | 98.3 | 8,554.8 | – | 1,991.0 | – | 82.8 |
| e5_small_ft | pretrained | 0.8656 | 0.8415 | 0.8462 | 0.9651 | 117.66M | 465.1 | 3,824.5 | 118.3 | 3,396.4 | 815.0 | 104.5 |
| tfidf_word_char_lr | tfidf | 0.8576 | 0.8202 | 0.8253 | 0.9620 | 1.51M | 15.2 | 146.1 | – | 561.0 | – | 27.4 |
| tiny_transformer_4L_kd_a90_T2_trainval | tiny_transformer | 0.8658 | 0.8051 | 0.7992 | 0.9641 | 5.23M | 20.7 | 521.0 | 53.0 | 1,074.9 | 125.3 | 58.0 |
| tiny_transformer_4L_kd_a90_T2 | tiny_transformer | 0.8716 | 0.7992 | 0.7939 | 0.9623 | 5.23M | 20.7 | 523.7 | 52.3 | 1,072.5 | 125.0 | 58.1 |
| tiny_transformer_4L | tiny_transformer | 0.8256 | 0.7931 | 0.7905 | 0.9564 | 5.23M | – | – | – | 1,070.4 | – | 39.6 |
| mt5_small_kkprune_text | seq2seq | 0.8213 | 0.7910 | 0.7956 | 0.9575 | 58.73M | 225.0 | 20,552.0 | – | 3,560.9 | – | 179.3 |
| e5_frozen_lr | pretrained_frozen | 0.6634 | 0.6992 | 0.6981 | 0.9096 | 117.66M | 448.9 | 3,548.3 | 106.7 | 1,230.1 | 797.4 | 24.4 |
| char_1nn | baseline | 0.6747 | 0.6697 | 0.6322 | 0.9065 | 47.53M | 362.9 | 2,801.2 | – | 2,275.5 | – | 9.9 |
| majority_dummy | baseline | 0.0471 | 0.0451 | 0.0625 | 0.1876 | 0 | 0.0 | 0.1 | – | 75.7 | – | 0.7 |

### Task `form`: ablations and other variants

| model | family | F1 val | F1 test | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5_small_ft_prioradj | posthoc | 0.9173 | 0.8538 | 0.8663 | 0.9660 | 117.66M | 465.1 | 3,824.5 | 118.3 | – | – | 104.5 |
| ensemble_tfidf_tiny_e5 | posthoc | 0.9166 | 0.8531 | 0.8617 | 0.9679 | 124.40M | 501.1 | 4,494.3 | 170.6 | – | – | 190.0 |
| tiny_transformer_4L_mlm_kd_a90_T2 | tiny_transformer | 0.8647 | 0.8252 | 0.8280 | 0.9655 | 5.23M | 20.7 | 507.1 | 50.7 | 1,092.5 | 125.6 | 53.3 |
| tiny_transformer_4L_mlm | tiny_transformer | 0.8514 | 0.8188 | 0.8173 | 0.9615 | 5.23M | 20.7 | 508.5 | 50.6 | 1,090.3 | 125.7 | 38.5 |
| tiny_transformer_2L_do30 | tiny_transformer | 0.8292 | 0.8132 | 0.8153 | 0.9560 | 3.65M | – | – | – | 627.9 | – | 24.7 |
| tiny_transformer_4L_kd_a90_T4 | tiny_transformer | 0.8549 | 0.8102 | 0.8075 | 0.9619 | 5.23M | 20.7 | 517.2 | 51.2 | 1,072.5 | 126.3 | 53.2 |
| tiny_transformer_2L_d128 | tiny_transformer | 0.8405 | 0.8099 | 0.8090 | 0.9582 | 1.43M | – | – | – | 347.2 | – | 15.4 |
| tiny_transformer_4L_kd_a50_T2 | tiny_transformer | 0.8526 | 0.8098 | 0.8091 | 0.9635 | 5.23M | 20.7 | 523.9 | 53.6 | 1,071.8 | 123.8 | 43.8 |
| tiny_transformer_2L | tiny_transformer | 0.8294 | 0.8085 | 0.8100 | 0.9576 | 3.65M | – | – | – | 627.9 | – | 22.0 |
| ablation_tfidf_word_nofold_num | tfidf | 0.8362 | 0.8084 | 0.8068 | 0.9580 | 0.86M | – | – | – | – | – | 0.7 |
| ablation_tfidf_word_fold_num | tfidf | 0.8309 | 0.8030 | 0.8007 | 0.9575 | 0.81M | – | – | – | – | – | 0.8 |
| tfidf_word_lr | tfidf | 0.8309 | 0.8030 | 0.8007 | 0.9575 | 0.81M | 9.3 | 23.0 | – | 49.7 | – | 4.5 |
| ablation_tfidf_word_raw | tfidf | 0.8372 | 0.7982 | 0.7934 | 0.9578 | 1.02M | – | – | – | – | – | 0.7 |
| ablation_tfidf_word_fold_digits | tfidf | 0.8318 | 0.7928 | 0.7856 | 0.9567 | 0.96M | – | – | – | – | – | 0.8 |
| tiny_transformer_4L_recency_tau8 | tiny_transformer | 0.8443 | 0.7921 | 0.7824 | 0.9558 | 5.23M | 20.7 | 533.1 | 52.3 | 1,071.0 | 125.0 | 23.9 |
| tiny_transformer_4L_do30 | tiny_transformer | 0.8260 | 0.7872 | 0.7806 | 0.9552 | 5.23M | – | – | – | 1,070.4 | – | 34.8 |
| tfidf_stem_lr | tfidf | 0.8275 | 0.7829 | 0.7710 | 0.9553 | 0.61M | 6.7 | 25.4 | – | 37.4 | – | 6.3 |
| tiny_transformer_4L_bpe4k | tiny_transformer | 0.8511 | 0.7799 | 0.7722 | 0.9573 | 4.21M | – | – | – | 1,255.1 | – | 64.9 |
| tfidf_char_lr | tfidf | 0.8006 | 0.7783 | 0.7745 | 0.9559 | 0.70M | 5.9 | 114.6 | – | 517.6 | – | 23.2 |
| tiny_transformer_4L_recency_tau3 | tiny_transformer | 0.8435 | 0.7774 | 0.7659 | 0.9572 | 5.23M | 20.7 | 518.8 | 50.8 | 1,071.0 | 125.4 | 52.4 |
| tiny_transformer_4L_d128 | tiny_transformer | 0.8114 | 0.7545 | 0.7431 | 0.9471 | 1.83M | – | – | – | 565.9 | – | 11.7 |

## Seed variance of the tiny transformer

Same data and hyper-parameters, seeds [1, 2, 3] (weight init + batch order). Differences between tiny-transformer variants smaller than about 2 std are within seed noise.

| task | arch | loss | f1_val mean ± std | f1_test mean ± std | f1_test_strict mean ± std |
|---|---|---|---|---|---|
| area | 2L | smooth | 0.5767 ± 0.0025 | 0.5789 ± 0.0093 | 0.5529 ± 0.0105 |
| area | 4L | smooth | 0.5782 ± 0.0022 | 0.5738 ± 0.0021 | 0.5437 ± 0.0053 |
| form | 2L | smooth | 0.8289 ± 0.0052 | 0.7912 ± 0.0137 | 0.7900 ± 0.0131 |
| form | 4L | smooth | 0.8268 ± 0.0097 | 0.7886 ± 0.0121 | 0.7834 ± 0.0188 |
