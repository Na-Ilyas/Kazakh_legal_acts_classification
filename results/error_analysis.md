# Error analysis

All numbers come from the test predictions stored in SQLite (`predictions` table, test split >= 2023). Figures: `reports/figures/confusion_*.png`, `per_class_f1_*.png`, `error_by_year_length.png`, `error_by_similarity.png`.

## Task `form`

Models analysed (best of each family, chosen on validation macro-F1):

| model_family | model_name | f1_val | f1_test |
|---|---|---|---|
| baseline | char_1nn | 0.6747 | 0.6697 |
| tfidf | tfidf_word_char_lr | 0.8576 | 0.8202 |
| pretrained_frozen | e5_frozen_lr | 0.6634 | 0.6992 |
| tiny_transformer | tiny_transformer_4L_kd_a90_T2 | 0.8716 | 0.7992 |
| pretrained | kaz_roberta_llrd_rec | 0.8978 | 0.8414 |
| posthoc | kaz_roberta_llrd_rec_prioradj | 0.9342 | 0.8434 |
| seq2seq | mt5_small_kkprune_text | 0.8213 | 0.7910 |

Overall best on validation: **kaz_roberta_llrd_rec_prioradj** (val 0.9342, test 0.8434).

### Per-class quality (best model)

Hardest classes for kaz_roberta_llrd_rec_prioradj: **Басқа** F1 0.40 (P 0.57, R 0.31, n=26); **Жарлық** F1 0.83 (P 0.87, R 0.79, n=319); **Өкім** F1 0.84 (P 0.81, R 0.87, n=262).
Easiest: **Шешім** F1 0.99 (n=24668); **Заң** F1 0.98 (n=329); **Бұйрық** F1 0.95 (n=5648).
Full table: `results/per_class_metrics.csv`; figure `reports/figures/per_class_f1_form.png`.

### Most confused class pairs (best model, test)

- **Шешім → Қаулы**: 400 errors. Examples (highest confidence):
    - "Катонқарағай ауданы бойынша коммуналдық қалдықтардың түзілу және жинақталу нормаларын есептеудің қағидаларын бекіту туралы" (conf 1.0)
    - "Жергілікті бюджеттен қаржыландырылатын ұйымдар жұмыскерлерінің лауазымдық айлықақыларына ынталандыру үстемақыларын белгілеу тәртібі мен шарттарын айқындау туралы" (conf 1.0)
    - "2024 жылға жұмыс орындарына квота белгілеу туралы" (conf 1.0)
- **Бұйрық → Қаулы**: 303 errors. Examples (highest confidence):
    - "Қазақы ит тұқымдарын сақтау және өсімін молайту жөніндегі қызметті жүзеге асыруға арналған мамандандырылған ұйымды айқындау туралы" (conf 1.0)
    - "Қазақстан Республикасының Үкіметі белгілеген шекті шама шеңберінде "Әлеуметтік медициналық сақтандыру қоры" коммерциялық емес акционерлік қоғамының 2025 жылға арналған комиссиялық сыйақысының пайыздық мөлшерлемесінің шам" (conf 1.0)
    - "Энергетикалық және коммуналдық секторларды жаңғырту бойынша ұлттық жобаны іске асыру шеңберіндегі сатып алулардың электрондық платформасының әкімшісін айқындау туралы" (conf 1.0)
- **Қаулы → Шешім**: 176 errors. Examples (highest confidence):
    - "Кентау қалалық мәслихатының 2022 жылғы 27 желтоқсандағы № 209 "Кентау қаласының Ащысай ауылының 2023-2025 жылдарға арналған бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 1.0)
    - "Павлодар облыстық мәслихатының 2021 жылғы 22 сәуірдегі № 22/3 "Павлодар облысының елді мекендерінде ауыл шаруашылығы жануарларын асырау қағидаларын бекіту туралы" шешіміне өзгеріс енгізу туралы" (conf 1.0)
    - "Солтүстік Қазақстан облысы Тайынша ауданы мәслихатының 2024 жылғы 19 сәуірдегі № 151/13 "Солтүстік Қазақстан облысы Тайынша ауданында тұрғын-үй көмегін көрсетудің мөлшері мен тәртібін айқындау туралы" шешіміне өзгерістер" (conf 1.0)
- **Қаулы → Бұйрық**: 168 errors. Examples (highest confidence):
    - "Ведомстволық статистикалық байқауларды жүргізу үшін статистикалық нысандарды және оларды толтыру нұсқаулықтарын бекіту туралы" (conf 1.0)
    - ""Атмосфералық ауаға және су объектілеріне ластағыш заттар эмиссияларының мониторингі", "Негізгі органикалық химиялық заттар және полимер өндірісі", "Елді мекендердің орталықтандырылған су бұру жүйелерінің сарқынды сулары" (conf 1.0)
    - "Ең үздік қолжетімді техникалар бойынша "Алюминий өндірісі", "Мұнай және газ өндіру", "Қара металды одан әрі қайта өңдеу бұйымдарының өндірісі", "Көмір өндіру және байыту", "Шойын және болат өндірісі", "Шаруашылық және (н" (conf 1.0)
- **Жарлық → Қаулы**: 57 errors. Examples (highest confidence):
    - "Қазақстан Республикасы Мәдениет және ақпарат министрлігі Мәдениет комитетінің "Әбілхан Қастеев атындағы Қазақстан Республикасының мемлекеттік өнер музейі" және "Қазақстан Республикасының мемлекеттік орталық музейі" респу" (conf 1.0)
    - "Мемлекеттік (муниципалдық) сатып алуды жүзеге асыру кезінде банк кепілдіктерін өзара тану туралы келісімге қол қою туралы" (conf 0.996)
    - "Есірткі құралдарының, психотроптық заттар мен олардың прекурсорларының заңсыз айналымына қарсы күрес жөніндегі Орталық Азия өңірлік ақпараттық үйлестіру орталығы шеңберінде құпия ақпаратты алмасу және қорғау туралы келіс" (conf 0.995)

### Error rate by year and title length (best model)

| year | n | errors | error_rate |
|---|---|---|---|
| 2023 | 12627 | 343 | 0.0272 |
| 2024 | 12535 | 356 | 0.0284 |
| 2025 | 12023 | 532 | 0.0442 |
| 2026 | 1282 | 90 | 0.0702 |

| length_bucket | n | errors | error_rate |
|---|---|---|---|
| 01-10 | 5025 | 448 | 0.0892 |
| 11-20 | 10238 | 555 | 0.0542 |
| 21-30 | 16603 | 214 | 0.0129 |
| 31-50 | 5519 | 81 | 0.0147 |
| 51+ | 1082 | 23 | 0.0213 |

Error rate ranges from 0.013 (21-30 words) to 0.089 (01-10 words); by year from 0.027 (2023) to 0.070 (2026).

Error rate by adopting-body type (SQL JOIN predictions-acts-bodies, groups with n>=30):

| body_type | n | errors | error_rate |
|---|---|---|---|
| Президент | 346 | 72 | 0.2081 |
| Әкім | 1414 | 277 | 0.1959 |
| Сот / прокуратура | 210 | 24 | 0.1143 |
| Үкімет | 1545 | 121 | 0.0783 |
| Халықаралық / ЕАЭО | 278 | 17 | 0.0612 |
| Министрлік | 4596 | 255 | 0.0555 |

### Error rate vs similarity to the nearest train title (SQL JOIN predictions-nn_sim)

`max_cos` = highest char-ngram TF-IDF cosine between a test title and any train title (data/processed/nn_similarity.parquet; >= 0.9 means a near-duplicate, excluded from test_strict).

| sim_bucket | n | char_1nn | e5_frozen_lr | kaz_roberta_llrd_rec | kaz_roberta_llrd_rec_prioradj | mt5_small_kkprune_text | tfidf_word_char_lr | tiny_transformer_4L_kd_a90_T2 |
|---|---|---|---|---|---|---|---|---|
| 1: <0.60 | 3388 | 0.4560 | 0.2774 | 0.1712 | 0.1677 | 0.1948 | 0.1809 | 0.1939 |
| 2: 0.60-0.75 | 5128 | 0.2034 | 0.1488 | 0.0612 | 0.0597 | 0.0647 | 0.0632 | 0.0595 |
| 3: 0.75-0.85 | 4914 | 0.1048 | 0.1079 | 0.0368 | 0.0350 | 0.0458 | 0.0446 | 0.0397 |
| 4: 0.85-0.90 | 4211 | 0.0484 | 0.0850 | 0.0221 | 0.0216 | 0.0302 | 0.0249 | 0.0256 |
| 5: 0.90-0.95 | 6790 | 0.0227 | 0.0493 | 0.0102 | 0.0105 | 0.0158 | 0.0130 | 0.0099 |
| 6: >=0.95 | 14036 | 0.0097 | 0.0393 | 0.0081 | 0.0081 | 0.0130 | 0.0081 | 0.0085 |

The label of the nearest train title equals the true label for 90.6% of test titles (a 1-NN 'copy' rule). kaz_roberta_llrd_rec_prioradj predicts the neighbour's label for 91.7% of titles, and 52.9% of its errors are exactly the neighbour's label (errors inherited from a similar training title).

| model_name | n | errors | neighbour_acc | copy_rate | errors_copying_share |
|---|---|---|---|---|---|
| char_1nn | 38467 | 3597 | 0.9063 | 0.9998 | 1.0000 |
| tfidf_word_char_lr | 38467 | 1462 | 0.9063 | 0.9192 | 0.5793 |
| e5_frozen_lr | 38467 | 3478 | 0.9063 | 0.8721 | 0.2510 |
| tiny_transformer_4L_kd_a90_T2 | 38467 | 1452 | 0.9063 | 0.9131 | 0.5034 |
| kaz_roberta_llrd_rec | 38467 | 1350 | 0.9063 | 0.9167 | 0.5274 |
| kaz_roberta_llrd_rec_prioradj | 38467 | 1321 | 0.9063 | 0.9169 | 0.5291 |
| mt5_small_kkprune_text | 38467 | 1634 | 0.9063 | 0.9045 | 0.3960 |

Confidence: 96.9% of test predictions have confidence >= 0.9 with accuracy 0.978, vs 0.5746 below 0.9; 62.0% of all errors are confident (>= 0.9).

### Agreement between model families (test, SQL self-join on predictions)

| A | B | both right | only A right | only B right | both wrong | same prediction |
|---|---|---|---|---|---|---|
| char_1nn | tfidf_word_char_lr | 34518 | 352 | 2487 | 1110 | 0.919 |
| char_1nn | e5_frozen_lr | 32681 | 2189 | 2308 | 1289 | 0.872 |
| char_1nn | tiny_transformer_4L_kd_a90_T2 | 34399 | 471 | 2616 | 981 | 0.913 |
| char_1nn | kaz_roberta_llrd_rec | 34558 | 312 | 2559 | 1038 | 0.917 |
| char_1nn | kaz_roberta_llrd_rec_prioradj | 34578 | 292 | 2568 | 1029 | 0.917 |
| char_1nn | mt5_small_kkprune_text | 34151 | 719 | 2682 | 915 | 0.905 |
| tfidf_word_char_lr | e5_frozen_lr | 34543 | 2462 | 446 | 1016 | 0.921 |
| tfidf_word_char_lr | tiny_transformer_4L_kd_a90_T2 | 36535 | 470 | 480 | 982 | 0.973 |
| tfidf_word_char_lr | kaz_roberta_llrd_rec | 36628 | 377 | 489 | 973 | 0.976 |
| tfidf_word_char_lr | kaz_roberta_llrd_rec_prioradj | 36646 | 359 | 500 | 962 | 0.976 |
| tfidf_word_char_lr | mt5_small_kkprune_text | 36280 | 725 | 553 | 909 | 0.964 |
| e5_frozen_lr | tiny_transformer_4L_kd_a90_T2 | 34606 | 383 | 2409 | 1069 | 0.923 |
| e5_frozen_lr | kaz_roberta_llrd_rec | 34626 | 363 | 2491 | 987 | 0.923 |
| e5_frozen_lr | kaz_roberta_llrd_rec_prioradj | 34623 | 366 | 2523 | 955 | 0.922 |
| e5_frozen_lr | mt5_small_kkprune_text | 34483 | 506 | 2350 | 1128 | 0.922 |
| tiny_transformer_4L_kd_a90_T2 | kaz_roberta_llrd_rec | 36638 | 377 | 479 | 973 | 0.975 |
| tiny_transformer_4L_kd_a90_T2 | kaz_roberta_llrd_rec_prioradj | 36656 | 359 | 490 | 962 | 0.975 |
| tiny_transformer_4L_kd_a90_T2 | mt5_small_kkprune_text | 36410 | 605 | 423 | 1029 | 0.971 |
| kaz_roberta_llrd_rec | kaz_roberta_llrd_rec_prioradj | 37090 | 27 | 56 | 1294 | 0.998 |
| kaz_roberta_llrd_rec | mt5_small_kkprune_text | 36357 | 760 | 476 | 874 | 0.966 |
| kaz_roberta_llrd_rec_prioradj | mt5_small_kkprune_text | 36374 | 772 | 459 | 862 | 0.966 |

Oracle (at least one of the 7 models right): 0.9860 accuracy; all models wrong on 537 of 38467 test titles.

Examples where **tfidf_word_char_lr** is right and **tiny_transformer_4L_kd_a90_T2** is wrong:

- "Төлеби ауданы бойынша қалдықтарды басқару бағдарламасын бекіту туралы" — true Қаулы, tiny_transformer_4L_kd_a90_T2 predicted Шешім (conf 0.997)
- "Шортанды ауданының ішкі саясат, мәдениет және тілдерді дамыту бөліміненің "Жастар ресурстық орталығы" және "Тілдерді оқыту орталығы" коммуналдық мемлекеттік мекемелері қызметкерлерінің лауазымдық айлы" — true Қаулы, tiny_transformer_4L_kd_a90_T2 predicted Шешім (conf 0.997)
- "Казталов ауданы бойынша арнаулы әлеуметтік қызметтер көрсетуге арналған тарифтерді бекіту туралы" — true Қаулы, tiny_transformer_4L_kd_a90_T2 predicted Шешім (conf 0.997)
- "2006 жылғы 4 шiлдедегі Қазақстан Республикасының Үкiметi мен Қырғыз Республикасының Үкiметi арасындағы Қырғыз Республикасы Ыстықкөл облысының аумағында орналасқан курорттық-рекреациялық шаруашылық объ" — true Заң, tiny_transformer_4L_kd_a90_T2 predicted Қаулы (conf 0.997)

Examples where **tiny_transformer_4L_kd_a90_T2** is right and **tfidf_word_char_lr** is wrong:

- "Қаумдық сервитутты орнату туралы" — true Шешім, tfidf_word_char_lr predicted Қаулы (conf 0.996)
- "Жергілікті бюджеттен қаржыландырылатын ішкі саясат бөлімінің жанындағы "Астрахан ауданының Жастар ресурстық орталығы" КММ мамандарының лауазымдық айлықақыларына ынталандырушы үстемеақылар белгілеу тур" — true Шешім, tfidf_word_char_lr predicted Қаулы (conf 0.995)
- "Солтүстік Қазақстан облысы Айыртау ауданының жергілікті бюджетінен қаржыландырылатын ұйымдар қызметкерлерінің лауазымдық айлықақыларына ынталандырушы үстемеақылар белгілеудің тәртібі мен шарттарын айқ" — true Шешім, tfidf_word_char_lr predicted Қаулы (conf 0.992)
- "Солтүстік Қазақстан облысы Жамбыл ауданының жергілікті бюджеттен қаржыландырылатын ұйымдар жұмыскерлерінің лауазымдық айлықақыларына ынталандыру үстемеақыларын белгілеу туралы" — true Шешім, tfidf_word_char_lr predicted Қаулы (conf 0.987)

## Task `area`

Models analysed (best of each family, chosen on validation macro-F1):

| model_family | model_name | f1_val | f1_test |
|---|---|---|---|
| baseline | char_1nn | 0.5555 | 0.5363 |
| tfidf | tfidf_word_char_lr | 0.5985 | 0.5827 |
| pretrained_frozen | e5_frozen_lr | 0.5277 | 0.5324 |
| tiny_transformer | tiny_transformer_4L_recency_tau8 | 0.6003 | 0.5832 |
| pretrained | kaz_roberta_llrd_rec | 0.6162 | 0.6285 |
| posthoc | ensemble_tfidf_tiny_e5_roberta | 0.6407 | 0.6215 |

Overall best on validation: **ensemble_tfidf_tiny_e5_roberta** (val 0.6407, test 0.6215).

### Per-class quality (best model)

Hardest classes for ensemble_tfidf_tiny_e5_roberta: **Сыртқы экономикалық қызмет** F1 0.00 (P 0.00, R 0.00, n=8); **Басқа** F1 0.15 (P 0.14, R 0.15, n=26); **Қылмыстық құқық** F1 0.19 (P 0.22, R 0.17, n=29).
Easiest: **ЕурАзЭҚ. Кедендік одақ. ЕАЭО** F1 0.95 (n=283); **Қаржы** F1 0.94 (n=16928); **Халықаралық қатынастар** F1 0.90 (n=477).
Full table: `results/per_class_metrics.csv`; figure `reports/figures/per_class_f1_area.png`.

### Most confused class pairs (best model, test)

- **Конституциялық құрылыс және мемлекеттік басқару негіздері → Қаржы**: 1519 errors. Examples (highest confidence):
    - "Көкпекті аудандық мәслихатының 2023 жылғы 26 желтоқсандағы № 10-9 "2024-2026 жылдарға арналған Үлкебөкен ауылдық округінің бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 0.983)
    - "Көкпекті аудандық мәслихатының 2023 жылғы 26 желтоқсандағы № 10-7 "2024-2026 жылдарға арналған Тассай ауылдық округінің бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 0.981)
    - "Бесқарағай аудандық мәслихатының 2023 жылғы 28 желтоқсандағы № 12/2-VIII "2024-2026 жылдарға арналған Баскөл ауылдық округінің бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 0.981)
- **Конституциялық құрылыс және мемлекеттік басқару негіздері → Еңбек**: 332 errors. Examples (highest confidence):
    - ""Б" корпусындағы аудандық бюджеттен қаржыландырылатын атқарушы органдардың мемлекеттік әкімшілік қызметшілері мен Түлкібас ауданы әкімі аппаратының мемлекеттік әкімшілік қызметшілерінің қызметін бағалаудың әдістемесін бе" (conf 0.95)
    - "Бәйдібек ауданы әкімдігінің 2023 жылғы 4 қаңтардағы № 1 ""Б" корпусындағы аудандық бюджеттен қаржыландырылатын атқарушы органдардың мемлекеттік әкімшілік қызметшілері мен Бәйдібек ауданы әкімі аппаратының мемлекеттік әкі" (conf 0.947)
    - "Түркістан облысы Сайрам ауданы әкiмдiгiнiң 2023 жылғы 10 қазандағы № 391 "Б" корпусындағы аудандық бюджеттен қаржыландырылатын атқарушы органдардың мемлекеттік әкімшілік қызметшілері мен Сайрам ауданы әкімі аппаратының м" (conf 0.941)
- **Еңбек → Конституциялық құрылыс және мемлекеттік басқару негіздері**: 240 errors. Examples (highest confidence):
    - "Аудандық мәслихаттың 2022 жылғы 28 қыркүйектегі № 176 "Махамбет аудандық мәслихатының аппараты" мемлекеттік мекемесінің Ережесін бекіту туралы" шешіміне өзгеріс енгізу туралы" (conf 0.959)
    - ""Құрманғазы аудандық Маслихатының аппараты" мемлекеттік мекемесінің Ережесін бекіту туралы" (conf 0.935)
    - "Мақат ауданының Мақат, Доссор кенттері және Бәйге төбе ауылдық округінің жергілікті қоғамдастық жиналысының регламенттерін бекіту туралы" (conf 0.928)
- **Конституциялық құрылыс және мемлекеттік басқару негіздері → Шаруашылық қызмет**: 233 errors. Examples (highest confidence):
    - ""Жасанды интеллект туралы" және "Қазақстан Республикасының кейбір заңнамалық актілеріне жасанды интеллект және цифрландыру мәселелері бойынша өзгерістер мен толықтырулар енгізу туралы" 2025 жылғы 17 қарашадағы Қазақстан " (conf 0.805)
    - "Қазақстан Республикасының кейбір заңнамалық актілеріне биржа саудасы және кәсіпкерлік мәселелері бойынша өзгерістер мен толықтырулар енгізу туралы" (conf 0.787)
    - ""Қазақстан Республикасының кейбір заңнамалық актілеріне инновациялық қызмет саласын мемлекеттік қолдау жүйесін жетілдіру және мемлекеттік статистика мәселелері бойынша өзгерістер мен толықтырулар енгізу туралы" 2025 жылғ" (conf 0.785)
- **Шаруашылық қызмет → Конституциялық құрылыс және мемлекеттік басқару негіздері**: 227 errors. Examples (highest confidence):
    - "Қарақия ауданы әкімінің 2015 жылғы 14 тамыздағы № 61 "Сайлау учаскелерін құру туралы" шешіміне өзгеріс енгізу туралы" (conf 0.963)
    - "Бейнеу ауданының аумағында жергілікті қоғамдастықтың бөлек жиындарын өткізудің қағидасын және жергілікті қоғамдастық жиынына қатысу үшін ауылдар тұрғындары өкілдерінің сандық құрамын бекіту туралы" (conf 0.949)
    - ""Сайлау учаскелерін құру туралы" Жаңаөзен қаласы әкімінің 2019 жылғы 10 қаңтардағы № 1 шешіміне өзгеріс енгізу туралы" (conf 0.945)

### Error rate by year and title length (best model)

| year | n | errors | error_rate |
|---|---|---|---|
| 2023 | 12612 | 2638 | 0.2092 |
| 2024 | 12495 | 2366 | 0.1894 |
| 2025 | 11990 | 2282 | 0.1903 |
| 2026 | 1280 | 316 | 0.2469 |

| length_bucket | n | errors | error_rate |
|---|---|---|---|
| 01-10 | 5014 | 634 | 0.1264 |
| 11-20 | 10219 | 1978 | 0.1936 |
| 21-30 | 16558 | 3131 | 0.1891 |
| 31-50 | 5505 | 1548 | 0.2812 |
| 51+ | 1081 | 311 | 0.2877 |

Error rate ranges from 0.126 (01-10 words) to 0.288 (51+ words); by year from 0.189 (2024) to 0.247 (2026).

Error rate by adopting-body type (SQL JOIN predictions-acts-bodies, groups with n>=30):

| body_type | n | errors | error_rate |
|---|---|---|---|
| Президент | 345 | 166 | 0.4812 |
| Сот / прокуратура | 210 | 87 | 0.4143 |
| unknown | 32 | 12 | 0.3750 |
| Үкімет | 1545 | 549 | 0.3553 |
| Бірлескен (joint) | 941 | 330 | 0.3507 |
| Әкімдік | 5145 | 1548 | 0.3009 |

### Error rate vs similarity to the nearest train title (SQL JOIN predictions-nn_sim)

`max_cos` = highest char-ngram TF-IDF cosine between a test title and any train title (data/processed/nn_similarity.parquet; >= 0.9 means a near-duplicate, excluded from test_strict).

| sim_bucket | n | char_1nn | e5_frozen_lr | ensemble_tfidf_tiny_e5_roberta | kaz_roberta_llrd_rec | tfidf_word_char_lr | tiny_transformer_4L_recency_tau8 |
|---|---|---|---|---|---|---|---|
| 1: <0.60 | 3378 | 0.5595 | 0.4663 | 0.3372 | 0.3505 | 0.4121 | 0.3999 |
| 2: 0.60-0.75 | 5117 | 0.3991 | 0.3541 | 0.2595 | 0.2644 | 0.2972 | 0.2982 |
| 3: 0.75-0.85 | 4899 | 0.3254 | 0.3131 | 0.2384 | 0.2382 | 0.2672 | 0.2733 |
| 4: 0.85-0.90 | 4191 | 0.2947 | 0.3080 | 0.2434 | 0.2484 | 0.2651 | 0.2677 |
| 5: 0.90-0.95 | 6781 | 0.2843 | 0.2901 | 0.2647 | 0.2657 | 0.2712 | 0.2770 |
| 6: >=0.95 | 14011 | 0.0976 | 0.1051 | 0.0822 | 0.0837 | 0.0908 | 0.0911 |

The label of the nearest train title equals the true label for 73.8% of test titles (a 1-NN 'copy' rule). ensemble_tfidf_tiny_e5_roberta predicts the neighbour's label for 84.9% of titles, and 68.8% of its errors are exactly the neighbour's label (errors inherited from a similar training title).

| model_name | n | errors | neighbour_acc | copy_rate | errors_copying_share |
|---|---|---|---|---|---|
| char_1nn | 38377 | 10057 | 0.7376 | 0.9990 | 0.9977 |
| tfidf_word_char_lr | 38377 | 8444 | 0.7376 | 0.8434 | 0.6440 |
| e5_frozen_lr | 38377 | 9651 | 0.7376 | 0.7714 | 0.4409 |
| tiny_transformer_4L_recency_tau8 | 38377 | 8492 | 0.7376 | 0.8294 | 0.5950 |
| kaz_roberta_llrd_rec | 38377 | 7720 | 0.7376 | 0.8358 | 0.6407 |
| ensemble_tfidf_tiny_e5_roberta | 38377 | 7602 | 0.7376 | 0.8491 | 0.6880 |

Confidence: 51.7% of test predictions have confidence >= 0.9 with accuracy 0.9259, vs 0.6691 below 0.9; 19.4% of all errors are confident (>= 0.9).

### Agreement between model families (test, SQL self-join on predictions)

| A | B | both right | only A right | only B right | both wrong | same prediction |
|---|---|---|---|---|---|---|
| char_1nn | tfidf_word_char_lr | 26941 | 1379 | 2992 | 7065 | 0.844 |
| char_1nn | e5_frozen_lr | 25359 | 2961 | 3367 | 6690 | 0.772 |
| char_1nn | tiny_transformer_4L_recency_tau8 | 26783 | 1537 | 3102 | 6955 | 0.830 |
| char_1nn | kaz_roberta_llrd_rec | 27142 | 1178 | 3515 | 6542 | 0.836 |
| char_1nn | ensemble_tfidf_tiny_e5_roberta | 27370 | 950 | 3405 | 6652 | 0.850 |
| tfidf_word_char_lr | e5_frozen_lr | 27040 | 2893 | 1686 | 6758 | 0.833 |
| tfidf_word_char_lr | tiny_transformer_4L_recency_tau8 | 28468 | 1465 | 1417 | 7027 | 0.896 |
| tfidf_word_char_lr | kaz_roberta_llrd_rec | 28882 | 1051 | 1775 | 6669 | 0.900 |
| tfidf_word_char_lr | ensemble_tfidf_tiny_e5_roberta | 29441 | 492 | 1334 | 7110 | 0.937 |
| e5_frozen_lr | tiny_transformer_4L_recency_tau8 | 26926 | 1800 | 2959 | 6692 | 0.829 |
| e5_frozen_lr | kaz_roberta_llrd_rec | 27270 | 1456 | 3387 | 6264 | 0.834 |
| e5_frozen_lr | ensemble_tfidf_tiny_e5_roberta | 27513 | 1213 | 3262 | 6389 | 0.846 |
| tiny_transformer_4L_recency_tau8 | kaz_roberta_llrd_rec | 28841 | 1044 | 1816 | 6676 | 0.898 |
| tiny_transformer_4L_recency_tau8 | ensemble_tfidf_tiny_e5_roberta | 29178 | 707 | 1597 | 6895 | 0.919 |
| kaz_roberta_llrd_rec | ensemble_tfidf_tiny_e5_roberta | 30108 | 549 | 667 | 7053 | 0.957 |

Oracle (at least one of the 6 models right): 0.8691 accuracy; all models wrong on 5024 of 38377 test titles.

Examples where **tfidf_word_char_lr** is right and **tiny_transformer_4L_recency_tau8** is wrong:

- "Қызметтік тұрғынжайларды ұстау жəне орталықтандырылған жылыту жүйесі бюджет қаражаты есебінен қамтамасыз етілетін, сондай-ақ қызметтік тұрғынжайы жекешелендіруге жатпайтын жабық жəне оқшауланған əскер" — true Тұрғын үй-коммуналдық заңдар. Халыққа тұрмыстық қызмет көрсету, tiny_transformer_4L_recency_tau8 predicted Құрылыс (conf 0.928)
- "Сарыағаш ауданының құрылыс, сәулет және қала құрылысы бөлімінің Ережесін бекітудің кейбір мәселелері туралы" — true Конституциялық құрылыс және мемлекеттік басқару негіздері, tiny_transformer_4L_recency_tau8 predicted Құрылыс (conf 0.915)
- ""Ұлытау облысы бойынша тексеру комиссиясы" мемлекеттік мекемесінің "Б" корпусы мемлекеттік әкімшілік қызметшілерінің қызметін бағалаудың әдістемесін бекіту туралы" — true Еңбек, tiny_transformer_4L_recency_tau8 predicted Конституциялық құрылыс және мемлекеттік басқару негіздері (conf 0.913)
- "Е.К. Жамаубаевты тағайындау туралы" — true Конституциялық құрылыс және мемлекеттік басқару негіздері, tiny_transformer_4L_recency_tau8 predicted Еңбек (conf 0.909)

Examples where **tiny_transformer_4L_recency_tau8** is right and **tfidf_word_char_lr** is wrong:

- "Білім беру саласындағы пилоттық ұлттық жоба шеңберінде бюджеттік инвестицияларды жоспарлау және іске асыру қағидаларын және Білім беру саласындағы пилоттық ұлттық жобаны іске асыру кезінде бюджеттің а" — true Білім, tfidf_word_char_lr predicted Қаржы (conf 0.979)
- ""Жоғары және жоғары оқу орнынан кейінгі медициналық және фармацевтикалық білімі бар мамандарды даярлауға 2024-2025 оқу жылына арналған мемлекеттік білім беру тапсырысын орналастыру туралы" Қазақстан Р" — true Білім, tfidf_word_char_lr predicted Денсаулық сақтау (conf 0.978)
- ""Шектеу шараларының, оның ішінде карантиннің, әлеуметтік, табиғи және техногендік сипаттағы төтенше жағдайлардың қолданылу кезеңіне Нұр-Сұлтан қаласының мектепке дейінгі ұйымдарына мемлекеттік білім б" — true Конституциялық құрылыс және мемлекеттік басқару негіздері, tfidf_word_char_lr predicted Білім (conf 0.962)
- ""Мемлекеттік реттеуге жататын өлшемдер тізбесін бекіту туралы" Қазақстан Республикасы Сауда және интеграция министрінің 2021 жылғы 20 қазандағы № 566-НҚ бұйрығына өзгеріс енгізу туралы"" — true Шаруашылық қызмет, tfidf_word_char_lr predicted Конституциялық құрылыс және мемлекеттік басқару негіздері (conf 0.953)

