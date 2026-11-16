-- Indexes created AFTER the bulk load (faster inserts), used by sql/queries.sql.
CREATE INDEX IF NOT EXISTS idx_acts_year        ON acts(year);
CREATE INDEX IF NOT EXISTS idx_acts_form        ON acts(form_label);
CREATE INDEX IF NOT EXISTS idx_acts_split       ON acts(split);
CREATE INDEX IF NOT EXISTS idx_acts_body        ON acts(body_id);
CREATE INDEX IF NOT EXISTS idx_acts_region      ON acts(region_id);
CREATE INDEX IF NOT EXISTS idx_acts_section     ON acts(section_id);
CREATE INDEX IF NOT EXISTS idx_acts_force       ON acts(legal_force_id);
CREATE INDEX IF NOT EXISTS idx_act_form_tag     ON act_act_form(tag_id);
CREATE INDEX IF NOT EXISTS idx_act_area_area    ON act_area(area_id);
CREATE INDEX IF NOT EXISTS idx_predictions_law  ON predictions(law_id);
