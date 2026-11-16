-- ===========================================================================
-- Analytical queries over the normalised corpus + experiment tables.
-- Format: every query starts with "-- name: <id>" followed by "-- answers:"
-- and "-- uses:" comment lines. src/database.py:run_queries() parses this
-- file, runs every query into pandas and saves results/sql/<id>.csv.
-- ===========================================================================

-- name: acts_per_year_by_form
-- answers: How many acts of each primary legal form were adopted per year? (input of EDA figure "acts per year by form")
-- uses: GROUP BY on two keys
SELECT year,
       form_label,
       COUNT(*) AS n_acts
FROM acts
WHERE year IS NOT NULL
GROUP BY year, form_label
ORDER BY year, form_label;

-- name: top_adopting_bodies
-- answers: Which 15 bodies adopted the most acts, what type are they, when were they active and how many of their acts are repealed? (input of EDA figure "top bodies")
-- uses: JOIN acts->bodies, GROUP BY, aggregates, ORDER BY + LIMIT
SELECT b.name                              AS body,
       b.body_type,
       COUNT(*)                            AS n_acts,
       ROUND(100.0 * AVG(a.is_repealed), 1) AS pct_repealed,
       MIN(a.year)                         AS first_year,
       MAX(a.year)                         AS last_year
FROM acts a
JOIN bodies b ON b.body_id = a.body_id
GROUP BY b.body_id
ORDER BY n_acts DESC
LIMIT 15;

-- name: body_type_form_having
-- answers: Which (body type, legal form) combinations are frequent (>= 500 acts)? Shows that form is largely determined by who adopts the act.
-- uses: JOIN, GROUP BY + HAVING
SELECT b.body_type,
       a.form_label,
       COUNT(*) AS n_acts
FROM acts a
JOIN bodies b ON b.body_id = a.body_id
GROUP BY b.body_type, a.form_label
HAVING COUNT(*) >= 500
ORDER BY b.body_type, n_acts DESC;

-- name: yoy_change_by_form
-- answers: Year-over-year change of the number of acts per form (2000-2026): when did a form become more / less common?
-- uses: CTE + window function LAG() partitioned by form
WITH yearly AS (
    SELECT form_label, year, COUNT(*) AS n_acts
    FROM acts
    WHERE year BETWEEN 2000 AND 2026
    GROUP BY form_label, year
)
SELECT form_label,
       year,
       n_acts,
       LAG(n_acts) OVER (PARTITION BY form_label ORDER BY year) AS prev_year_acts,
       ROUND(100.0 * (n_acts - LAG(n_acts) OVER (PARTITION BY form_label ORDER BY year))
             / LAG(n_acts) OVER (PARTITION BY form_label ORDER BY year), 1) AS yoy_change_pct
FROM yearly
ORDER BY form_label, year;

-- name: running_total_by_year
-- answers: Cumulative growth of the corpus over time and the repeal rate per adoption year (how much of the corpus is "old law").
-- uses: CTE + running total SUM() OVER (ORDER BY ... ROWS UNBOUNDED PRECEDING) + grand total SUM() OVER ()
WITH yearly AS (
    SELECT year, COUNT(*) AS n_acts, SUM(is_repealed) AS n_repealed
    FROM acts
    WHERE year IS NOT NULL
    GROUP BY year
)
SELECT year,
       n_acts,
       SUM(n_acts) OVER (ORDER BY year ROWS UNBOUNDED PRECEDING) AS cumulative_acts,
       ROUND(100.0 * SUM(n_acts) OVER (ORDER BY year ROWS UNBOUNDED PRECEDING)
             / SUM(n_acts) OVER (), 2)                         AS cumulative_pct,
       ROUND(100.0 * n_repealed / n_acts, 1)                    AS pct_repealed
FROM yearly
ORDER BY year;

-- name: form_share_within_split
-- answers: Share of each form inside train / val / test - quantifies the label shift created by the temporal split.
-- uses: GROUP BY + share-within-partition window SUM(COUNT(*)) OVER (PARTITION BY split)
SELECT split,
       form_label,
       COUNT(*) AS n_acts,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split), 2) AS pct_within_split
FROM acts
WHERE split IS NOT NULL
GROUP BY split, form_label
ORDER BY split, n_acts DESC;

-- name: top_areas_per_form
-- answers: What are the 3 most frequent legal-relation areas for every legal form?
-- uses: 3-table JOIN through the M2M table, CTEs, RANK() OVER (PARTITION BY form)
WITH counts AS (
    SELECT a.form_label, ar.name AS area, COUNT(*) AS n_acts
    FROM acts a
    JOIN act_area aa ON aa.law_id = a.law_id
    JOIN areas ar    ON ar.area_id = aa.area_id
    GROUP BY a.form_label, ar.area_id
),
ranked AS (
    SELECT form_label, area, n_acts,
           RANK() OVER (PARTITION BY form_label ORDER BY n_acts DESC) AS area_rank
    FROM counts
)
SELECT form_label, area_rank, area, n_acts
FROM ranked
WHERE area_rank <= 3
ORDER BY form_label, area_rank;

-- name: repeal_rate_by_section
-- answers: Which curated database sections (>= 1000 acts) have the highest share of repealed acts?
-- uses: JOIN acts->sections, GROUP BY + HAVING, conditional aggregates
SELECT s.name                                AS section,
       s.is_area_class,
       COUNT(*)                              AS n_acts,
       SUM(a.is_repealed)                    AS n_repealed,
       ROUND(100.0 * AVG(a.is_repealed), 1)  AS pct_repealed
FROM acts a
JOIN sections s ON s.section_id = a.section_id
GROUP BY s.section_id
HAVING COUNT(*) >= 1000
ORDER BY pct_repealed DESC;

-- name: multi_area_acts_by_form
-- answers: Which forms have the most acts tagged with more legal-relation areas than the corpus average (i.e. "cross-cutting" acts)?
-- uses: derived-table subquery in FROM + scalar subquery in WHERE
SELECT a.form_label,
       COUNT(*)                AS n_acts_above_avg,
       ROUND(AVG(t.n_areas), 2) AS avg_areas_of_these,
       (SELECT ROUND(AVG(cnt), 2) FROM (SELECT COUNT(*) AS cnt FROM act_area GROUP BY law_id)) AS corpus_avg_areas
FROM acts a
JOIN (SELECT law_id, COUNT(*) AS n_areas FROM act_area GROUP BY law_id) t ON t.law_id = a.law_id
WHERE t.n_areas > (SELECT AVG(cnt) FROM (SELECT COUNT(*) AS cnt FROM act_area GROUP BY law_id))
GROUP BY a.form_label
ORDER BY n_acts_above_avg DESC;

-- name: form_tag_cooccurrence
-- answers: Which act_form tags co-occur most often on the same act (e.g. a decision that also amends)? Justifies treating act_form as an unordered tag set.
-- uses: self-JOIN of the M2M table + JOIN to the tag dictionary twice
SELECT t1.name   AS tag_a,
       t2.name   AS tag_b,
       COUNT(*)  AS n_acts
FROM act_act_form x
JOIN act_act_form y   ON y.law_id = x.law_id AND x.tag_id < y.tag_id
JOIN act_form_tags t1 ON t1.tag_id = x.tag_id
JOIN act_form_tags t2 ON t2.tag_id = y.tag_id
GROUP BY x.tag_id, y.tag_id
ORDER BY n_acts DESC
LIMIT 20;

-- name: title_length_by_form
-- answers: How long are titles of each form (chars / words)? Informs the max sequence length of the transformers.
-- uses: GROUP BY with several aggregates
SELECT form_label,
       COUNT(*)                         AS n_acts,
       ROUND(AVG(title_len_chars), 1)   AS avg_chars,
       MAX(title_len_chars)             AS max_chars,
       ROUND(AVG(title_len_words), 1)   AS avg_words,
       MAX(title_len_words)             AS max_words
FROM acts
GROUP BY form_label
ORDER BY avg_chars DESC;

-- name: area_form_matrix
-- answers: Joint distribution of the two targets (area_label x form_label) for acts in the model dataset (input of EDA heatmap).
-- uses: GROUP BY on two keys with a filter
SELECT area_label,
       form_label,
       COUNT(*) AS n_acts
FROM acts
WHERE area_label IS NOT NULL AND split IS NOT NULL
GROUP BY area_label, form_label
ORDER BY area_label, form_label;

-- name: top_regions_recent_rank
-- answers: Which regions produced the most acts in 2019-2026 and how does each rank inside its leading body type?
-- uses: JOINs, CTE, DENSE_RANK() global and RANK() partitioned by body type
WITH recent AS (
    SELECT r.name AS region, b.body_type, COUNT(*) AS n_acts
    FROM acts a
    JOIN regions r ON r.region_id = a.region_id
    JOIN bodies b  ON b.body_id = a.body_id
    WHERE a.year >= 2019
    GROUP BY r.region_id, b.body_type
)
SELECT region, body_type, n_acts,
       DENSE_RANK() OVER (ORDER BY n_acts DESC)                       AS overall_rank,
       RANK()       OVER (PARTITION BY body_type ORDER BY n_acts DESC) AS rank_in_body_type
FROM recent
ORDER BY n_acts DESC
LIMIT 25;

-- name: model_ranking_per_task
-- answers: Ranking of all logged models by macro-F1 within every (task, eval split), with their size / latency and the gap to the best model.
-- uses: window functions RANK() and MAX() OVER (PARTITION BY task, eval_split) on the experiments table
SELECT task,
       eval_split,
       model_name,
       model_family,
       ROUND(f1_macro, 4)                                                       AS f1_macro,
       RANK() OVER (PARTITION BY task, eval_split ORDER BY f1_macro DESC)        AS f1_rank,
       ROUND(f1_macro - MAX(f1_macro) OVER (PARTITION BY task, eval_split), 4)   AS gap_to_best,
       n_params,
       size_mb,
       latency_ms_per_1000,
       latency_device
FROM experiments
ORDER BY task, eval_split, f1_rank;

-- name: per_class_error_rate
-- answers: Per model and true class: test error rate, mean confidence, and whether errors have longer titles than correct predictions.
-- uses: 3-table JOIN predictions->experiments->acts, GROUP BY, conditional aggregation (CASE)
SELECT e.task,
       e.model_name,
       p.y_true,
       COUNT(*)                                                         AS n_test,
       ROUND(100.0 * SUM(p.y_true <> p.y_pred) / COUNT(*), 2)           AS error_rate_pct,
       ROUND(AVG(p.confidence), 3)                                       AS avg_confidence,
       ROUND(AVG(CASE WHEN p.y_true <> p.y_pred THEN a.title_len_chars END), 1) AS avg_len_errors,
       ROUND(AVG(CASE WHEN p.y_true =  p.y_pred THEN a.title_len_chars END), 1) AS avg_len_correct
FROM predictions p
JOIN experiments e ON e.exp_id = p.exp_id
JOIN acts a        ON a.law_id = p.law_id
WHERE e.eval_split = 'test'
GROUP BY e.task, e.model_name, p.y_true
ORDER BY e.task, e.model_name, error_rate_pct DESC;

-- name: error_rate_by_year
-- answers: Does test accuracy degrade for acts adopted later (temporal drift)? Error rate per model and adoption year.
-- uses: JOIN predictions->experiments->acts, GROUP BY, LAG() for the change vs previous year
WITH per_year AS (
    SELECT e.task, e.model_name, a.year,
           COUNT(*) AS n_test,
           ROUND(100.0 * SUM(p.y_true <> p.y_pred) / COUNT(*), 2) AS error_rate_pct
    FROM predictions p
    JOIN experiments e ON e.exp_id = p.exp_id
    JOIN acts a        ON a.law_id = p.law_id
    WHERE e.eval_split = 'test'
    GROUP BY e.task, e.model_name, a.year
)
SELECT task, model_name, year, n_test, error_rate_pct,
       ROUND(error_rate_pct - LAG(error_rate_pct) OVER (PARTITION BY task, model_name ORDER BY year), 2) AS change_vs_prev_year
FROM per_year
ORDER BY task, model_name, year;
