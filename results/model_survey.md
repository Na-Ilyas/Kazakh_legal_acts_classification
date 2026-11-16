# Compact Kazakh-capable encoders - survey

Tokenizer statistics on 3,000 seeded TRAIN titles (fertility = tokens per whitespace word; lower is better). Params from HF metadata. F1 = macro-F1 (val / test / test_strict, shared src.metrics) on the tasks `form` and `area` where the model was trained in this project (see SQLite `experiments`).

| model | params | <=120M | vocab | fertility | median tok/title | unk % | >128 tok % | note | form F1 val / test / strict | area F1 val / test / strict |
|---|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 117.7M | yes | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | multilingual MiniLM-L12, XLM-R SentencePiece 250k (project baseline) | 0.8656 / 0.8415 / 0.8462 | 0.5836 / 0.6010 / 0.5792 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 117.7M | yes | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | same architecture/tokenizer as e5-small |  |  |
| kz-transformers/kaz-roberta-conversational | 83.5M | yes | 52000 | 1.256 | 23.0 | 0.0 | 0.03 | monolingual Kazakh RoBERTa | 0.8830 / 0.8528 / 0.8675 | 0.6013 / 0.6223 / 0.6011 |
| Eraly-ml/KazBERT | 110.7M | yes | 32000 | 1.453 | 26.0 | 0.0 | 0.07 | Kazakh BERT | 0.8794 / 0.8601 / 0.8662 | 0.6239 / 0.6201 / 0.5968 |
| google/mt5-small | 300.2M | no | 250100 | 2.209 | 40.5 | 0.0 | 0.97 | seq2seq; >120M unless vocabulary-pruned (src.seq2seq_mt5) |  |  |
| FacebookAI/xlm-roberta-base | 278.9M | no | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | too large (>120M) without pruning |  |  |
| e5_small_kkprune_ft (this project) | 25.5M | yes | train vocab | = base | = base | 0 | = base | e5-small with Kazakh-pruned vocabulary (src.vocab_prune) | 0.8628 / 0.8448 / 0.8521 | 0.5805 / 0.5997 / 0.5790 |
| kaz_roberta_kkprune_ft (this project) | 61.6M | yes | train vocab | = base | = base | 0 | = base | kaz-roberta with Kazakh-train-pruned vocabulary (52,000 -> 23,532 tokens) | 0.8843 / 0.8491 / 0.8633 | 0.6000 / 0.6213 / 0.6003 |
| mt5_small_kkprune_text (this project) | 58.7M | yes | train vocab | = base | = base | 0 | = base | mT5-small seq2seq, pruned vocab, label-set scoring (src.seq2seq_mt5) | 0.8213 / 0.7910 / 0.7956 |  |

## Kazakh teachers: KazBERT fine-tune and kaz-roberta distillation (src.distill_roberta)

KazBERT fine-tuned with the kaz-roberta recipe (form: lr 5e-5, 3 epochs; area: LLRD 0.8 + recency weights). Distillation students use teacher probabilities on TRAIN titles from the val-selectable kaz-roberta models (`kaz_roberta_ft` form, `kaz_roberta_llrd_rec` area); `kdavg` = mean of the kaz-roberta and `kazbert_ft` train probabilities; `kd_` rows (e5 teacher) are shown for reference. Macro-F1 from SQLite; selection on val only. CPU ms per 1000 titles (8 threads, busy shared machine; `e5_small_kkprune_kdrob` has the same architecture as `e5_small_kkprune_ft`, its higher latency is load noise, and its MB includes the saved tokenizer files).

| task | model | val | test | test_strict | params | MB | CPU ms/1000 |
|---|---|---|---|---|---|---|---|
| area | kazbert_ft | 0.6239 | 0.6201 | 0.5968 | 110.6M | 423 | 23035 |
| area | kaz_roberta_llrd_rec | 0.6162 | 0.6285 | 0.6080 | 83.5M | 324 | 9776 |
| area | tiny_transformer_4L_mlm_kdavg_a100_T2 | 0.6054 | 0.6187 | 0.5976 | 5.2M | 21 | 526 |
| area | tiny_transformer_4L_mlm_kdavg_a90_T2 | 0.6049 | 0.6116 | 0.5927 | 5.2M | 21 | 513 |
| area | kaz_roberta_ft | 0.6013 | 0.6223 | 0.6011 | 83.5M | 324 | 9693 |
| area | tiny_transformer_4L_mlm_kdavg_a50_T2 | 0.6003 | 0.5962 | 0.5722 | 5.2M | 21 | 516 |
| area | tiny_transformer_4L_mlm_kdrob_a50_T2 | 0.5967 | 0.5979 | 0.5759 | 5.2M | 21 | 516 |
| area | tiny_transformer_4L_mlm_kdrob_a90_T2 | 0.5934 | 0.6032 | 0.5801 | 5.2M | 21 | 548 |
| area | tiny_transformer_4L_mlm_kd_a50_T2 | 0.5922 | 0.5974 | 0.5797 | 5.2M | 21 | 502 |
| area | e5_small_kkprune_ft | 0.5805 | 0.5997 | 0.5790 | 25.5M | 98 | 3726 |
| area | e5_small_kkprune_kdrob | 0.5769 | 0.6043 | 0.5840 | 25.5M | 115 | 8090 |
| form | kaz_roberta_llrd_rec | 0.8978 | 0.8414 | 0.8478 | 83.5M | 324 | 10875 |
| form | kaz_roberta_ft | 0.8830 | 0.8528 | 0.8675 | 83.5M | 324 | 9708 |
| form | kazbert_ft | 0.8794 | 0.8601 | 0.8662 | 110.6M | 423 | 25676 |
| form | tiny_transformer_4L_mlm_kdavg_a90_T2 | 0.8778 | 0.8387 | 0.8495 | 5.2M | 21 | 534 |
| form | tiny_transformer_4L_mlm_kdavg_a100_T2 | 0.8758 | 0.8267 | 0.8323 | 5.2M | 21 | 539 |
| form | tiny_transformer_4L_mlm_kdavg_a50_T2 | 0.8741 | 0.8284 | 0.8362 | 5.2M | 21 | 541 |
| form | tiny_transformer_4L_mlm_kdrob_a50_T2 | 0.8683 | 0.8273 | 0.8286 | 5.2M | 21 | 541 |
| form | tiny_transformer_4L_mlm_kd_a90_T2 | 0.8647 | 0.8252 | 0.8280 | 5.2M | 21 | 538 |
| form | e5_small_kkprune_ft | 0.8628 | 0.8448 | 0.8521 | 25.5M | 98 | 3557 |
| form | tiny_transformer_4L_mlm_kdrob_a90_T2 | 0.8569 | 0.8237 | 0.8252 | 5.2M | 21 | 540 |
| form | e5_small_kkprune_kdrob | 0.8527 | 0.8450 | 0.8543 | 25.5M | 115 | 7767 |

## kaz-roberta improvements (src.roberta_tune)

All rows from SQLite `experiments` (macro-F1, shared `src.metrics`). Selection is on **val** only; `*_trainval` rows are refits on train+val with the selected epoch count, so their val column is in-sample (marked *) and was never used for selection. Latency: CPU, 8 threads, batch 64, ms per 1000 titles on a busy shared machine.

| task | model | val | test | test_strict | params | MB | CPU ms/1000 |
|---|---|---|---|---|---|---|---|
| area | kaz_roberta_llrd_rec | 0.6162 | 0.6285 | 0.6080 | 83.5M | 324 | 10018 |
| area | kaz_roberta_combo | 0.6131 | 0.6330 | 0.6135 | 83.5M | 324 | 10018 |
| area | kaz_roberta_llrd_cw | 0.6066 | 0.6193 | 0.5980 | 83.5M | 324 | 10018 |
| area | kaz_roberta_llrd | 0.6065 | 0.6216 | 0.6000 | 83.5M | 324 | 10018 |
| area | kaz_roberta_llrd_ls | 0.6046 | 0.6181 | 0.5946 | 83.5M | 324 | 10018 |
| area | kaz_roberta_ep5 | 0.6025 | 0.6093 | 0.5861 | 83.5M | 324 | 10018 |
| area | kaz_roberta_ft | 0.6013 | 0.6223 | 0.6011 | 83.5M | 324 | 9849 |
| area | kaz_roberta_kkprune_ft | 0.6000 | 0.6213 | 0.6003 | 61.6M | 240 | 10068 |
| area | kaz_roberta_llrd_rec_trainval | 0.8102* | 0.6345 | 0.6163 | 83.5M | 324 | 10018 |
| form | kaz_roberta_llrd_rec | 0.8978 | 0.8414 | 0.8478 | 83.5M | 324 | 11950 |
| form | kaz_roberta_kkprune_ft | 0.8843 | 0.8491 | 0.8633 | 61.6M | 240 | 10037 |
| form | kaz_roberta_ft | 0.8830 | 0.8528 | 0.8675 | 83.5M | 324 | 10218 |

- `area`: val-selected **kaz_roberta_llrd_rec**; vs `kaz_roberta_ft` (lr 5e-5, 3 epochs): val +0.0148, test +0.0063, test_strict +0.0069.
- `form`: val-selected **kaz_roberta_llrd_rec**; vs `kaz_roberta_ft` (lr 5e-5, 3 epochs): val +0.0148, test -0.0113, test_strict -0.0198. The val gain does NOT transfer to test (2022 val vs >=2023 test shift); reported as selected, test was not used to change the choice.
