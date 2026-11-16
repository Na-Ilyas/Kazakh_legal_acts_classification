# Compact Kazakh-capable encoders - survey

Tokenizer statistics on 3,000 seeded TRAIN titles (fertility = tokens per whitespace word; lower is better). Params from HF metadata. F1 = macro-F1 (val / test / test_strict, shared src.metrics) on the tasks `form` and `area` where the model was trained in this project (see SQLite `experiments`).

| model | params | <=120M | vocab | fertility | median tok/title | unk % | >128 tok % | note | form F1 val / test / strict | area F1 val / test / strict |
|---|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 117.7M | yes | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | multilingual MiniLM-L12, XLM-R SentencePiece 250k (project baseline) | 0.8656 / 0.8415 / 0.8462 | 0.5836 / 0.6010 / 0.5792 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 117.7M | yes | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | same architecture/tokenizer as e5-small |  |  |
| kz-transformers/kaz-roberta-conversational | 83.5M | yes | 52000 | 1.256 | 23.0 | 0.0 | 0.03 | monolingual Kazakh RoBERTa | 0.8830 / 0.8528 / 0.8675 | 0.6013 / 0.6223 / 0.6011 |
| Eraly-ml/KazBERT | 110.7M | yes | 32000 | 1.453 | 26.0 | 0.0 | 0.07 | Kazakh BERT |  |  |
| google/mt5-small | 300.2M | no | 250100 | 2.209 | 40.5 | 0.0 | 0.97 | seq2seq; >120M unless vocabulary-pruned (src.seq2seq_mt5) |  |  |
| FacebookAI/xlm-roberta-base | 278.9M | no | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | too large (>120M) without pruning |  |  |
| e5_small_kkprune_ft (this project) | 25.5M | yes | train vocab | = base | = base | 0 | = base | e5-small with Kazakh-pruned vocabulary (src.vocab_prune) | 0.8628 / 0.8448 / 0.8521 | 0.5805 / 0.5997 / 0.5790 |
| kaz_roberta_kkprune_ft (this project) | 61.6M | yes | train vocab | = base | = base | 0 | = base | kaz-roberta with Kazakh-train-pruned vocabulary (52,000 -> 23,532 tokens) | 0.8843 / 0.8491 / 0.8633 | 0.6000 / 0.6213 / 0.6003 |
| mt5_small_kkprune_text (this project) | 58.7M | yes | train vocab | = base | = base | 0 | = base | mT5-small seq2seq, pruned vocab, label-set scoring (src.seq2seq_mt5) | 0.8213 / 0.7910 / 0.7956 |  |

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
