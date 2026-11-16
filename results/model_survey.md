# Compact Kazakh-capable encoders - survey

Tokenizer statistics on 3,000 seeded TRAIN titles (fertility = tokens per whitespace word; lower is better). Params from HF metadata. F1 = macro-F1 on task `form` where the model was trained in this project (see SQLite `experiments`).

| model | params | <=120M | vocab | fertility | median tok/title | unk % | >128 tok % | note | form F1 val / test / strict |
|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 117.7M | yes | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | multilingual MiniLM-L12, XLM-R SentencePiece 250k (project baseline) | 0.8656 / 0.8415 / 0.8462 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 117.7M | yes | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | same architecture/tokenizer as e5-small |  |
| kz-transformers/kaz-roberta-conversational | 83.5M | yes | 52000 | 1.256 | 23.0 | 0.0 | 0.03 | monolingual Kazakh RoBERTa | 0.8830 / 0.8528 / 0.8675 |
| Eraly-ml/KazBERT | 110.7M | yes | 32000 | 1.453 | 26.0 | 0.0 | 0.07 | Kazakh BERT |  |
| google/mt5-small | 300.2M | no | 250100 | 2.209 | 40.5 | 0.0 | 0.97 | seq2seq; >120M unless vocabulary-pruned (src.seq2seq_mt5) | 0.8213 / 0.7910 / 0.7956 |
| FacebookAI/xlm-roberta-base | 278.9M | no | 250002 | 1.732 | 30.5 | 0.0 | 0.3 | too large (>120M) without pruning |  |

Fine-tuned `kz-transformers/kaz-roberta-conversational` on `form`: 83,456,263 params, 323.5 MB, 10217.7 ms / 1000 titles on CPU, best epoch 2, train 70.95 s.
