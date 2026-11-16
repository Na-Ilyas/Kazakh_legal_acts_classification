# Ablations and other variants

Every logged variant that is not the val-best of its group (see `results/model_comparison.md` for the main table). Columns as in the main table; latency of ablations = measured at training time with the same protocol.

## Task `area`: ablations and other variants

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kaz_roberta_combo | pretrained kaz-roberta | 0.6131 | – | 0.6330 | – | – | 0.6135 | 0.7975 | 83.48M | 323.6 | 10,018.5 | – | 3,173.2 | – | 116.2 |
| kaz_roberta_llrd | pretrained kaz-roberta | 0.6065 | – | 0.6216 | – | – | 0.6000 | 0.7955 | 83.48M | 323.6 | 10,018.5 | – | 3,175.3 | – | 119.6 |
| kaz_roberta_llrd_cw | pretrained kaz-roberta | 0.6066 | – | 0.6193 | – | – | 0.5980 | 0.7913 | 83.48M | 323.6 | 10,018.5 | – | 3,172.6 | – | 120.3 |
| kaz_roberta_llrd_ls | pretrained kaz-roberta | 0.6046 | – | 0.6181 | – | – | 0.5946 | 0.7951 | 83.48M | 323.6 | 10,018.5 | – | 3,172.6 | – | 123.6 |
| kaz_roberta_llrd_rec_prioradj | posthoc | – | 0.6339 | 0.6097 | – | – | 0.5887 | 0.7950 | 83.48M | 323.6 | 9,775.9 | 250.8 | – | – | 119.0 |
| kaz_roberta_ep5 | pretrained kaz-roberta | 0.6025 | – | 0.6093 | – | – | 0.5861 | 0.7920 | 83.48M | 323.6 | 10,018.5 | – | 3,175.4 | – | 119.5 |
| ensemble_tfidf_tiny_e5 (>120M, reference) | ensemble | – | 0.6180 | 0.6040 | – | – | 0.5839 | 0.7916 | 130.04M | 522.6 | 4,258.5 | 158.0 | – | – | 209.3 |
| tiny_transformer_4L_mlm_kdrob_a90_T2 | tiny + KD (kaz-roberta teacher) | 0.5934 | – | 0.6032 | – | – | 0.5801 | 0.7886 | 5.24M | 20.8 | 547.7 | 65.1 | 1,100.3 | 105.9 | 44.0 |
| tiny_transformer_2L_do30 | tiny (from scratch) | 0.5842 | – | 0.5937 | – | – | 0.5681 | 0.7780 | 3.66M | 14.7 | 302.4 | – | 628.0 | 112.2 | 26.6 |
| tiny_transformer_4L_kd_a90_T2 | tiny + KD (teacher e5) | 0.5786 | – | 0.5936 | – | – | 0.5671 | 0.7861 | 5.24M | 20.8 | 512.9 | 54.1 | 1,080.4 | 125.0 | 48.4 |
| ablation_tfidf_word_fold_num | baseline/tfidf | 0.5861 | – | 0.5915 | – | – | 0.5624 | 0.7793 | 3.81M | – | – | – | – | – | 3.0 |
| tfidf_word_lr | baseline/tfidf | 0.5861 | – | 0.5915 | – | – | 0.5624 | 0.7793 | 3.81M | 20.8 | 30.7 | – | 61.2 | – | 6.7 |
| tfidf_stem_lr | baseline/tfidf | 0.5942 | – | 0.5905 | – | – | 0.5643 | 0.7806 | 2.86M | 15.3 | 32.6 | – | 45.9 | – | 8.3 |
| ablation_tfidf_word_nofold_num | baseline/tfidf | 0.5860 | – | 0.5857 | – | – | 0.5558 | 0.7782 | 4.05M | – | – | – | – | – | 3.1 |
| ablation_tfidf_word_raw | baseline/tfidf | 0.5812 | – | 0.5821 | – | – | 0.5522 | 0.7754 | 4.79M | – | – | – | – | – | 3.1 |
| ablation_tfidf_word_fold_digits | baseline/tfidf | 0.5784 | – | 0.5810 | – | – | 0.5537 | 0.7760 | 4.55M | – | – | – | – | – | 3.1 |
| tiny_transformer_4L_kd_a90_T4 | tiny + KD (teacher e5) | 0.5735 | – | 0.5808 | – | – | 0.5573 | 0.7835 | 5.24M | 20.8 | 535.0 | 52.6 | 1,080.4 | 131.2 | 53.3 |
| tiny_transformer_2L | tiny (from scratch) | 0.5896 | – | 0.5764 | – | – | 0.5482 | 0.7746 | 3.66M | 14.7 | 297.1 | – | 628.0 | 111.8 | 24.7 |
| tiny_transformer_4L_recency_tau3 | tiny + recency weighting | 0.5968 | – | 0.5748 | – | – | 0.5478 | 0.7780 | 5.24M | 20.8 | 521.4 | 51.5 | 1,071.1 | 126.3 | 33.3 |
| tiny_transformer_2L_d128 | tiny (from scratch) | 0.5869 | – | 0.5727 | – | – | 0.5419 | 0.7745 | 1.44M | 6.3 | 131.6 | – | 347.2 | 71.3 | 12.2 |
| tiny_transformer_4L_d128 | tiny (from scratch) | 0.5768 | – | 0.5682 | – | – | 0.5433 | 0.7733 | 1.83M | 7.8 | 209.8 | – | 566.0 | 73.0 | 20.4 |
| tfidf_char_lr | baseline/tfidf | 0.5712 | – | 0.5569 | – | – | 0.5264 | 0.7613 | 3.32M | 15.9 | 128.4 | – | 527.6 | – | 47.8 |
| char_1nn | baseline/tfidf | 0.5555 | – | 0.5363 | – | – | 0.4875 | 0.7379 | 47.53M | 362.9 | 2,801.2 | – | 2,275.5 | – | 9.9 |

## Task `form`: ablations and other variants

| model | group | F1 val | F1 val½B | F1 test | test 95% CI | Δ vs TF-IDF [95% CI] | F1 test_strict | acc test | params | size MB | CPU ms/1000 | GPU ms/1000 | train peak MB | infer RAM MB | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble_tfidf_tiny_e5_prioradj (>120M, reference) | ensemble | – | 0.9190 | 0.8624 | – | – | 0.8792 | 0.9687 | 124.40M | 501.1 | 4,192.3 | 161.5 | – | – | 190.0 |
| ensemble_tfidf_tiny_e5 (>120M, reference) | ensemble | – | 0.9166 | 0.8531 | – | – | 0.8617 | 0.9679 | 124.40M | 501.1 | 4,192.3 | 161.5 | – | – | 190.0 |
| tiny_transformer_4L_mlm_kdrob_a90_T2 | tiny + KD (kaz-roberta teacher) | 0.8569 | – | 0.8237 | – | – | 0.8252 | 0.9617 | 5.23M | 20.7 | 540.5 | 55.8 | 1,092.5 | 122.9 | 34.0 |
| tiny_transformer_2L_do30 | tiny (from scratch) | 0.8292 | – | 0.8132 | – | – | 0.8153 | 0.9560 | 3.65M | 14.7 | 304.7 | – | 627.9 | 116.1 | 24.7 |
| tiny_transformer_4L_kd_a90_T4 | tiny + KD (teacher e5) | 0.8549 | – | 0.8102 | – | – | 0.8075 | 0.9619 | 5.23M | 20.7 | 517.2 | 51.2 | 1,072.5 | 126.3 | 53.2 |
| tiny_transformer_4L_kd_a50_T2 | tiny + KD (teacher e5) | 0.8526 | – | 0.8098 | – | – | 0.8091 | 0.9635 | 5.23M | 20.7 | 523.9 | 53.6 | 1,071.8 | 123.8 | 43.8 |
| tiny_transformer_2L | tiny (from scratch) | 0.8294 | – | 0.8085 | – | – | 0.8100 | 0.9576 | 3.65M | 14.7 | 293.1 | – | 627.9 | 115.8 | 22.0 |
| ablation_tfidf_word_nofold_num | baseline/tfidf | 0.8362 | – | 0.8084 | – | – | 0.8068 | 0.9580 | 0.86M | – | – | – | – | – | 0.7 |
| ablation_tfidf_word_fold_num | baseline/tfidf | 0.8309 | – | 0.8030 | – | – | 0.8007 | 0.9575 | 0.81M | – | – | – | – | – | 0.8 |
| tfidf_word_lr | baseline/tfidf | 0.8309 | – | 0.8030 | – | – | 0.8007 | 0.9575 | 0.81M | 9.3 | 23.0 | – | 49.7 | – | 4.5 |
| ablation_tfidf_word_raw | baseline/tfidf | 0.8372 | – | 0.7982 | – | – | 0.7934 | 0.9578 | 1.02M | – | – | – | – | – | 0.7 |
| ablation_tfidf_word_fold_digits | baseline/tfidf | 0.8318 | – | 0.7928 | – | – | 0.7856 | 0.9567 | 0.96M | – | – | – | – | – | 0.8 |
| tiny_transformer_4L_do30 | tiny (from scratch) | 0.8260 | – | 0.7872 | – | – | 0.7806 | 0.9552 | 5.23M | 20.7 | 526.9 | – | 1,070.4 | 126.3 | 34.8 |
| tfidf_stem_lr | baseline/tfidf | 0.8275 | – | 0.7829 | – | – | 0.7710 | 0.9553 | 0.61M | 6.7 | 25.4 | – | 37.4 | – | 6.3 |
| tiny_transformer_4L_bpe4k | tiny tokenizer ablation | 0.8511 | – | 0.7799 | – | – | 0.7722 | 0.9573 | 4.21M | 16.4 | 641.1 | – | 1,255.1 | 141.5 | 64.9 |
| tfidf_char_lr | baseline/tfidf | 0.8006 | – | 0.7783 | – | – | 0.7745 | 0.9559 | 0.70M | 5.9 | 114.6 | – | 517.6 | – | 23.2 |
| tiny_transformer_4L_recency_tau3 | tiny + recency weighting | 0.8435 | – | 0.7774 | – | – | 0.7659 | 0.9572 | 5.23M | 20.7 | 518.8 | 50.8 | 1,071.0 | 125.4 | 52.4 |
| tiny_transformer_4L_d128 | tiny (from scratch) | 0.8114 | – | 0.7545 | – | – | 0.7431 | 0.9471 | 1.83M | 7.8 | 220.1 | – | 565.9 | 74.1 | 11.7 |
| char_1nn | baseline/tfidf | 0.6747 | – | 0.6697 | – | – | 0.6322 | 0.9065 | 47.53M | 362.9 | 2,801.2 | – | 2,275.5 | – | 9.9 |
