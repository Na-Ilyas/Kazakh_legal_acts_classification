# Error analysis

All numbers come from the test predictions stored in SQLite (`predictions` table, test split >= 2023). Figures: `reports/figures/confusion_*.png`, `per_class_f1_*.png`, `error_by_year_length.png`, `error_by_similarity.png`.

## Task `form`

Models analysed (val-best model of each comparison group, as in the main table):

| group | model_name | f1_val | f1_test |
|---|---|---|---|
| baseline/tfidf | tfidf_word_char_lr | 0.8576 | 0.8202 |
| pretrained_frozen | e5_frozen_lr | 0.6634 | 0.6992 |
| tiny_transformer | tiny_transformer_4L_kd_a90_T2 | 0.8716 | 0.7992 |
| pretrained e5 | e5_small_ft | 0.8656 | 0.8415 |
| pretrained kaz-roberta | kaz_roberta_llrd_rec | 0.8978 | 0.8414 |
| ensemble | ensemble_tfidf_tiny_e5_roberta | 0.9314 | 0.8497 |

Overall best single model on validation (ensembles excluded: their val score is on val half B): **kaz_roberta_llrd_rec** (val 0.8978, test 0.8414).

### Per-class quality (best model)

Hardest classes for kaz_roberta_llrd_rec: **Басқа** F1 0.39 (P 0.53, R 0.31, n=26); **Жарлық** F1 0.83 (P 0.88, R 0.79, n=319); **Өкім** F1 0.84 (P 0.81, R 0.87, n=262).
Easiest: **Шешім** F1 0.99 (n=24668); **Заң** F1 0.98 (n=329); **Бұйрық** F1 0.95 (n=5648).
Full table: `results/per_class_metrics.csv`; figure `reports/figures/per_class_f1_form.png`.

### Most confused class pairs (best model, test)

- **Шешім → Қаулы**: 423 errors. Examples (highest confidence):
    - "2024 жылға жұмыс орындарына квота белгілеу туралы" (conf 1.0)
    - "Катонқарағай ауданы бойынша коммуналдық қалдықтардың түзілу және жинақталу нормаларын есептеудің қағидаларын бекіту туралы" (conf 1.0)
    - "Жергілікті бюджеттен қаржыландырылатын ұйымдар жұмыскерлерінің лауазымдық айлықақыларына ынталандыру үстемақыларын белгілеу тәртібі мен шарттарын айқындау туралы" (conf 1.0)
- **Бұйрық → Қаулы**: 334 errors. Examples (highest confidence):
    - "Энергетикалық және коммуналдық секторларды жаңғырту бойынша ұлттық жобаны іске асыру шеңберіндегі сатып алулардың электрондық платформасының әкімшісін айқындау туралы" (conf 1.0)
    - "Қазақстан Республикасының су қоймаларын, каналдары мен гидротехникалық құрылысжайларын дамытуға мемлекеттік кепілдікпен тартылған қарыздар бойынша өтеу, қызмет көрсету және қоса қаржыландыру жөніндегі шығыстарды жабуға б" (conf 1.0)
    - "Қазақы ит тұқымдарын сақтау және өсімін молайту жөніндегі қызметті жүзеге асыруға арналған мамандандырылған ұйымды айқындау туралы" (conf 1.0)
- **Қаулы → Бұйрық**: 160 errors. Examples (highest confidence):
    - "Ведомстволық статистикалық байқауларды жүргізу үшін статистикалық нысандарды және оларды толтыру нұсқаулықтарын бекіту туралы" (conf 1.0)
    - ""Атмосфералық ауаға және су объектілеріне ластағыш заттар эмиссияларының мониторингі", "Негізгі органикалық химиялық заттар және полимер өндірісі", "Елді мекендердің орталықтандырылған су бұру жүйелерінің сарқынды сулары" (conf 1.0)
    - "Жеке және заңды тұлғалардың Қазақстан Республикасы елордасының аумағында сауда, ойын-сауық, қонақ үй, медициналық және өзге де қызметтер көрсету қағидаларын бекіту туралы" (conf 0.999)
- **Қаулы → Шешім**: 159 errors. Examples (highest confidence):
    - "Кентау қалалық мәслихатының 2022 жылғы 27 желтоқсандағы № 209 "Кентау қаласының Ащысай ауылының 2023-2025 жылдарға арналған бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 1.0)
    - "Павлодар облыстық мәслихатының 2021 жылғы 22 сәуірдегі № 22/3 "Павлодар облысының елді мекендерінде ауыл шаруашылығы жануарларын асырау қағидаларын бекіту туралы" шешіміне өзгеріс енгізу туралы" (conf 1.0)
    - "Солтүстік Қазақстан облысы Тайынша ауданы мәслихатының 2024 жылғы 19 сәуірдегі № 151/13 "Солтүстік Қазақстан облысы Тайынша ауданында тұрғын-үй көмегін көрсетудің мөлшері мен тәртібін айқындау туралы" шешіміне өзгерістер" (conf 1.0)
- **Жарлық → Қаулы**: 57 errors. Examples (highest confidence):
    - "Қазақстан Республикасы Мәдениет және ақпарат министрлігі Мәдениет комитетінің "Әбілхан Қастеев атындағы Қазақстан Республикасының мемлекеттік өнер музейі" және "Қазақстан Республикасының мемлекеттік орталық музейі" респу" (conf 1.0)
    - "Есірткі құралдарының, психотроптық заттар мен олардың прекурсорларының заңсыз айналымына қарсы күрес жөніндегі Орталық Азия өңірлік ақпараттық үйлестіру орталығы шеңберінде құпия ақпаратты алмасу және қорғау туралы келіс" (conf 0.996)
    - "Тәуелсіз Мемлекеттер Достастығына қатысушы мемлекеттердің Сот-сараптама қызметі саласындағы үйлестіру кеңесін құру туралы келісімге қол қою туралы" (conf 0.996)

### Error rate by year and title length (best model)

| year | n | errors | error_rate |
|---|---|---|---|
| 2023 | 12627 | 356 | 0.0282 |
| 2024 | 12535 | 359 | 0.0286 |
| 2025 | 12023 | 548 | 0.0456 |
| 2026 | 1282 | 87 | 0.0679 |

| length_bucket | n | errors | error_rate |
|---|---|---|---|
| 01-10 | 5025 | 457 | 0.0909 |
| 11-20 | 10238 | 569 | 0.0556 |
| 21-30 | 16603 | 220 | 0.0133 |
| 31-50 | 5519 | 79 | 0.0143 |
| 51+ | 1082 | 25 | 0.0231 |

Error rate ranges from 0.013 (21-30 words) to 0.091 (01-10 words); by year from 0.028 (2023) to 0.068 (2026).

Error rate by adopting-body type (SQL JOIN predictions-acts-bodies, groups with n>=30):

| body_type | n | errors | error_rate |
|---|---|---|---|
| Президент | 346 | 73 | 0.2110 |
| Әкім | 1414 | 291 | 0.2058 |
| unknown | 32 | 4 | 0.1250 |
| Сот / прокуратура | 210 | 24 | 0.1143 |
| Үкімет | 1545 | 116 | 0.0751 |
| Халықаралық / ЕАЭО | 278 | 18 | 0.0647 |

### Error rate vs similarity to the nearest train title (SQL JOIN predictions-nn_sim)

`max_cos` = highest char-ngram TF-IDF cosine between a test title and any train title (data/processed/nn_similarity.parquet; >= 0.9 means a near-duplicate, excluded from test_strict).

| sim_bucket | n | e5_frozen_lr | e5_small_ft | ensemble_tfidf_tiny_e5_roberta | kaz_roberta_llrd_rec | tfidf_word_char_lr | tiny_transformer_4L_kd_a90_T2 |
|---|---|---|---|---|---|---|---|
| 1: <0.60 | 3388 | 0.2774 | 0.1756 | 0.1626 | 0.1712 | 0.1809 | 0.1939 |
| 2: 0.60-0.75 | 5128 | 0.1488 | 0.0525 | 0.0521 | 0.0612 | 0.0632 | 0.0595 |
| 3: 0.75-0.85 | 4914 | 0.1079 | 0.0391 | 0.0352 | 0.0368 | 0.0446 | 0.0397 |
| 4: 0.85-0.90 | 4211 | 0.0850 | 0.0247 | 0.0228 | 0.0221 | 0.0249 | 0.0256 |
| 5: 0.90-0.95 | 6790 | 0.0493 | 0.0100 | 0.0090 | 0.0102 | 0.0130 | 0.0099 |
| 6: >=0.95 | 14036 | 0.0393 | 0.0081 | 0.0074 | 0.0081 | 0.0081 | 0.0085 |

The label of the nearest train title equals the true label for 90.6% of test titles (a 1-NN 'copy' rule). kaz_roberta_llrd_rec predicts the neighbour's label for 91.7% of titles, and 52.7% of its errors are exactly the neighbour's label (errors inherited from a similar training title).

| model_name | n | errors | neighbour_acc | copy_rate | errors_copying_share |
|---|---|---|---|---|---|
| tfidf_word_char_lr | 38467 | 1462 | 0.9063 | 0.9192 | 0.5793 |
| e5_frozen_lr | 38467 | 3478 | 0.9063 | 0.8721 | 0.2510 |
| tiny_transformer_4L_kd_a90_T2 | 38467 | 1452 | 0.9063 | 0.9131 | 0.5034 |
| e5_small_ft | 38467 | 1342 | 0.9063 | 0.9129 | 0.5104 |
| kaz_roberta_llrd_rec | 38467 | 1350 | 0.9063 | 0.9167 | 0.5274 |
| ensemble_tfidf_tiny_e5_roberta | 38467 | 1252 | 0.9063 | 0.9176 | 0.5599 |

Confidence: 97.0% of test predictions have confidence >= 0.9 with accuracy 0.9768, vs 0.5754 below 0.9; 64.1% of all errors are confident (>= 0.9).

### Agreement between model families (test, SQL self-join on predictions)

| A | B | both right | only A right | only B right | both wrong | same prediction |
|---|---|---|---|---|---|---|
| tfidf_word_char_lr | e5_frozen_lr | 34543 | 2462 | 446 | 1016 | 0.921 |
| tfidf_word_char_lr | tiny_transformer_4L_kd_a90_T2 | 36535 | 470 | 480 | 982 | 0.973 |
| tfidf_word_char_lr | e5_small_ft | 36589 | 416 | 536 | 926 | 0.973 |
| tfidf_word_char_lr | kaz_roberta_llrd_rec | 36628 | 377 | 489 | 973 | 0.976 |
| tfidf_word_char_lr | ensemble_tfidf_tiny_e5_roberta | 36788 | 217 | 427 | 1035 | 0.982 |
| e5_frozen_lr | tiny_transformer_4L_kd_a90_T2 | 34606 | 383 | 2409 | 1069 | 0.923 |
| e5_frozen_lr | e5_small_ft | 34657 | 332 | 2468 | 1010 | 0.924 |
| e5_frozen_lr | kaz_roberta_llrd_rec | 34626 | 363 | 2491 | 987 | 0.923 |
| e5_frozen_lr | ensemble_tfidf_tiny_e5_roberta | 34680 | 309 | 2535 | 943 | 0.924 |
| tiny_transformer_4L_kd_a90_T2 | e5_small_ft | 36727 | 288 | 398 | 1054 | 0.980 |
| tiny_transformer_4L_kd_a90_T2 | kaz_roberta_llrd_rec | 36638 | 377 | 479 | 973 | 0.975 |
| tiny_transformer_4L_kd_a90_T2 | ensemble_tfidf_tiny_e5_roberta | 36765 | 250 | 450 | 1002 | 0.979 |
| e5_small_ft | kaz_roberta_llrd_rec | 36738 | 387 | 379 | 963 | 0.978 |
| e5_small_ft | ensemble_tfidf_tiny_e5_roberta | 36851 | 274 | 364 | 978 | 0.982 |
| kaz_roberta_llrd_rec | ensemble_tfidf_tiny_e5_roberta | 37048 | 69 | 167 | 1183 | 0.994 |

Oracle (at least one of the 6 models right): 0.9836 accuracy; all models wrong on 632 of 38467 test titles.

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

Models analysed (val-best model of each comparison group, as in the main table):

| group | model_name | f1_val | f1_test |
|---|---|---|---|
| baseline/tfidf | tfidf_word_char_lr | 0.5985 | 0.5827 |
| pretrained_frozen | e5_frozen_lr | 0.5277 | 0.5324 |
| tiny_transformer | tiny_transformer_4L_recency_tau8 | 0.6003 | 0.5832 |
| pretrained e5 | e5_small_ft | 0.5836 | 0.6010 |
| pretrained kaz-roberta | kaz_roberta_llrd_rec | 0.6162 | 0.6285 |
| ensemble | ensemble_tfidf_tiny_e5_roberta | 0.6407 | 0.6215 |

Overall best single model on validation (ensembles excluded: their val score is on val half B): **kaz_roberta_llrd_rec** (val 0.6162, test 0.6285).

### Per-class quality (best model)

Hardest classes for kaz_roberta_llrd_rec: **Басқа** F1 0.20 (P 0.18, R 0.23, n=26); **Қылмыстық құқық** F1 0.23 (P 0.25, R 0.21, n=29); **Сыртқы экономикалық қызмет** F1 0.27 (P 0.21, R 0.38, n=8).
Easiest: **ЕурАзЭҚ. Кедендік одақ. ЕАЭО** F1 0.95 (n=283); **Қаржы** F1 0.94 (n=16928); **Халықаралық қатынастар** F1 0.89 (n=477).
Full table: `results/per_class_metrics.csv`; figure `reports/figures/per_class_f1_area.png`.

### Most confused class pairs (best model, test)

- **Конституциялық құрылыс және мемлекеттік басқару негіздері → Қаржы**: 1525 errors. Examples (highest confidence):
    - "Көкпекті аудандық мәслихатының 2022 жылғы 27 желтоқсандағы № 28-5/7 "2023-2025 жылдарға арналған Үлкенбөкен ауылдық округінің бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 1.0)
    - "Көкпекті аудандық мәслихатының 2022 жылғы 27 желтоқсандағы № 28-5/5 "2023-2025 жылдарға арналған Тассай ауылдық округінің бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 1.0)
    - "Көкпекті аудандық мәслихатының 2022 жылғы 27 желтоқсандағы № 28-5/5 "2023-2025 жылдарға арналған Тассай ауылдық округінің бюджеті туралы" шешіміне өзгерістер енгізу туралы" (conf 1.0)
- **Конституциялық құрылыс және мемлекеттік басқару негіздері → Еңбек**: 326 errors. Examples (highest confidence):
    - ""Б" корпусындағы аудандық бюджеттен қаржыландырылатын атқарушы органдардың мемлекеттік әкімшілік қызметшілері мен Түлкібас ауданы әкімі аппаратының мемлекеттік әкімшілік қызметшілерінің қызметін бағалаудың әдістемесін бе" (conf 0.999)
    - "Түркістан облысы Сайрам ауданы әкiмдiгiнiң 2023 жылғы 10 қазандағы № 391 "Б" корпусындағы аудандық бюджеттен қаржыландырылатын атқарушы органдардың мемлекеттік әкімшілік қызметшілері мен Сайрам ауданы әкімі аппаратының м" (conf 0.999)
    - "Түркістан облысы Кентау қаласы әкімдігінің 2023 жылғы 3 шілдедегі № 191 ""Б" корпусындағы қалалық бюджеттен қаржыландырылатын атқарушы органдардың мемлекеттік әкімшілік қызметшілері мен Кентау қаласы әкімі аппаратының ме" (conf 0.999)
- **Еңбек → Конституциялық құрылыс және мемлекеттік басқару негіздері**: 258 errors. Examples (highest confidence):
    - ""Құрманғазы аудандық Маслихатының аппараты" мемлекеттік мекемесінің Ережесін бекіту туралы" (conf 0.999)
    - "Аудандық мәслихаттың 2022 жылғы 28 қыркүйектегі № 176 "Махамбет аудандық мәслихатының аппараты" мемлекеттік мекемесінің Ережесін бекіту туралы" шешіміне өзгеріс енгізу туралы" (conf 0.999)
    - "Мәслихаттың 2023 жылғы 21 сәуірдегі № 29 "Қамысты аудандық мәслихатының аппараты" мемлекеттік мекемесінің "Б" корпусы мемлекеттік әкімшілік қызметшілерінің қызметін бағалау әдістемесін бекіту туралы" шешіміне өзгеріс енг" (conf 0.999)
- **Конституциялық құрылыс және мемлекеттік басқару негіздері → Шаруашылық қызмет**: 254 errors. Examples (highest confidence):
    - "Қазақстан Республикасы Ұлттық экономика министрінің "Реттеліп көрсетілетін қызметтерді ұсынудың үлгілік шарттарын бекіту туралы" 2019 жылғы 24 маусымдағы № 58 және "Табиғи монополиялар субъектілері қызметкерлерінің нақты" (conf 0.987)
    - "Жеке кәсіпкерлік субъектілерін креативті индустриялар субъектілеріне жатқызу өлшемшарттарын бекіту туралы" (conf 0.981)
    - "Павлодар облыстық мәслихатының 2019 жылғы 27 қыркүйектегі № 381/34 "Павлодар облысында жерлеудің және зираттарды күтіп ұстау ісін ұйымдастырудың қағидаларын бекіту туралы" шешімінің күші жойылды деп тану туралы" (conf 0.98)
- **Шаруашылық қызмет → Конституциялық құрылыс және мемлекеттік басқару негіздері**: 254 errors. Examples (highest confidence):
    - "Қарақия ауданы әкімінің 2015 жылғы 14 тамыздағы № 61 "Сайлау учаскелерін құру туралы" шешіміне өзгеріс енгізу туралы" (conf 0.999)
    - ""Солтүстік Қазақстан облысы Мамлют ауданы әкімдігінің экономика және қаржы бөлімі" коммуналдық мемлекеттік мекемесінің ережесін бекіту туралы" (conf 0.999)
    - "Бейнеу ауданының аумағында жергілікті қоғамдастықтың бөлек жиындарын өткізудің қағидасын және жергілікті қоғамдастық жиынына қатысу үшін ауылдар тұрғындары өкілдерінің сандық құрамын бекіту туралы" (conf 0.999)

### Error rate by year and title length (best model)

| year | n | errors | error_rate |
|---|---|---|---|
| 2023 | 12612 | 2671 | 0.2118 |
| 2024 | 12495 | 2390 | 0.1913 |
| 2025 | 11990 | 2341 | 0.1952 |
| 2026 | 1280 | 318 | 0.2484 |

| length_bucket | n | errors | error_rate |
|---|---|---|---|
| 01-10 | 5014 | 667 | 0.1330 |
| 11-20 | 10219 | 2055 | 0.2011 |
| 21-30 | 16558 | 3117 | 0.1882 |
| 31-50 | 5505 | 1549 | 0.2814 |
| 51+ | 1081 | 332 | 0.3071 |

Error rate ranges from 0.133 (01-10 words) to 0.307 (51+ words); by year from 0.191 (2024) to 0.248 (2026).

Error rate by adopting-body type (SQL JOIN predictions-acts-bodies, groups with n>=30):

| body_type | n | errors | error_rate |
|---|---|---|---|
| Президент | 345 | 164 | 0.4754 |
| unknown | 32 | 13 | 0.4063 |
| Сот / прокуратура | 210 | 80 | 0.3810 |
| Бірлескен (joint) | 941 | 356 | 0.3783 |
| Үкімет | 1545 | 560 | 0.3625 |
| Әкімдік | 5145 | 1573 | 0.3057 |

### Error rate vs similarity to the nearest train title (SQL JOIN predictions-nn_sim)

`max_cos` = highest char-ngram TF-IDF cosine between a test title and any train title (data/processed/nn_similarity.parquet; >= 0.9 means a near-duplicate, excluded from test_strict).

| sim_bucket | n | e5_frozen_lr | e5_small_ft | ensemble_tfidf_tiny_e5_roberta | kaz_roberta_llrd_rec | tfidf_word_char_lr | tiny_transformer_4L_recency_tau8 |
|---|---|---|---|---|---|---|---|
| 1: <0.60 | 3378 | 0.4663 | 0.3727 | 0.3372 | 0.3505 | 0.4121 | 0.3999 |
| 2: 0.60-0.75 | 5117 | 0.3541 | 0.2720 | 0.2595 | 0.2644 | 0.2972 | 0.2982 |
| 3: 0.75-0.85 | 4899 | 0.3131 | 0.2533 | 0.2384 | 0.2382 | 0.2672 | 0.2733 |
| 4: 0.85-0.90 | 4191 | 0.3080 | 0.2701 | 0.2434 | 0.2484 | 0.2651 | 0.2677 |
| 5: 0.90-0.95 | 6781 | 0.2901 | 0.2759 | 0.2647 | 0.2657 | 0.2712 | 0.2770 |
| 6: >=0.95 | 14011 | 0.1051 | 0.0879 | 0.0822 | 0.0837 | 0.0908 | 0.0911 |

The label of the nearest train title equals the true label for 73.8% of test titles (a 1-NN 'copy' rule). kaz_roberta_llrd_rec predicts the neighbour's label for 83.6% of titles, and 64.1% of its errors are exactly the neighbour's label (errors inherited from a similar training title).

| model_name | n | errors | neighbour_acc | copy_rate | errors_copying_share |
|---|---|---|---|---|---|
| tfidf_word_char_lr | 38377 | 8444 | 0.7376 | 0.8434 | 0.6440 |
| e5_frozen_lr | 38377 | 9651 | 0.7376 | 0.7714 | 0.4409 |
| tiny_transformer_4L_recency_tau8 | 38377 | 8492 | 0.7376 | 0.8294 | 0.5950 |
| e5_small_ft | 38377 | 8126 | 0.7376 | 0.8184 | 0.5845 |
| kaz_roberta_llrd_rec | 38377 | 7720 | 0.7376 | 0.8358 | 0.6407 |
| ensemble_tfidf_tiny_e5_roberta | 38377 | 7602 | 0.7376 | 0.8491 | 0.6880 |

Confidence: 74.5% of test predictions have confidence >= 0.9 with accuracy 0.8906, vs 0.5316 below 0.9; 40.5% of all errors are confident (>= 0.9).

### Agreement between model families (test, SQL self-join on predictions)

| A | B | both right | only A right | only B right | both wrong | same prediction |
|---|---|---|---|---|---|---|
| tfidf_word_char_lr | e5_frozen_lr | 27040 | 2893 | 1686 | 6758 | 0.833 |
| tfidf_word_char_lr | tiny_transformer_4L_recency_tau8 | 28468 | 1465 | 1417 | 7027 | 0.896 |
| tfidf_word_char_lr | e5_small_ft | 28454 | 1479 | 1797 | 6647 | 0.885 |
| tfidf_word_char_lr | kaz_roberta_llrd_rec | 28882 | 1051 | 1775 | 6669 | 0.900 |
| tfidf_word_char_lr | ensemble_tfidf_tiny_e5_roberta | 29441 | 492 | 1334 | 7110 | 0.937 |
| e5_frozen_lr | tiny_transformer_4L_recency_tau8 | 26926 | 1800 | 2959 | 6692 | 0.829 |
| e5_frozen_lr | e5_small_ft | 27439 | 1287 | 2812 | 6839 | 0.854 |
| e5_frozen_lr | kaz_roberta_llrd_rec | 27270 | 1456 | 3387 | 6264 | 0.834 |
| e5_frozen_lr | ensemble_tfidf_tiny_e5_roberta | 27513 | 1213 | 3262 | 6389 | 0.846 |
| tiny_transformer_4L_recency_tau8 | e5_small_ft | 28464 | 1421 | 1787 | 6705 | 0.886 |
| tiny_transformer_4L_recency_tau8 | kaz_roberta_llrd_rec | 28841 | 1044 | 1816 | 6676 | 0.898 |
| tiny_transformer_4L_recency_tau8 | ensemble_tfidf_tiny_e5_roberta | 29178 | 707 | 1597 | 6895 | 0.919 |
| e5_small_ft | kaz_roberta_llrd_rec | 29028 | 1223 | 1629 | 6497 | 0.901 |
| e5_small_ft | ensemble_tfidf_tiny_e5_roberta | 29343 | 908 | 1432 | 6694 | 0.919 |
| kaz_roberta_llrd_rec | ensemble_tfidf_tiny_e5_roberta | 30108 | 549 | 667 | 7053 | 0.957 |

Oracle (at least one of the 6 models right): 0.8656 accuracy; all models wrong on 5157 of 38377 test titles.

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

