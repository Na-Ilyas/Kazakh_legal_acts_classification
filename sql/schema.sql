-- ===========================================================================
-- Normalised corpus schema (3NF) for KazakhLawCorpus, SQLite.
-- Lives in the same file as the `experiments` / `predictions` tables
-- (src/experiments_db.py), so model results can be joined to act metadata.
--
-- Design:
--   * acts               - one row per legal act (law_id is the natural key)
--   * bodies, regions,   - lookup tables for repeated low-cardinality text;
--     sections,            acts reference them through integer FKs
--     legal_forces
--   * act_form_tags +    - `act_form` is an UNORDERED multi-valued tag list
--     act_act_form         ("Шешім, Өзгерту" == "Өзгерту, Шешім") -> M2M
--   * areas + act_area   - `legal_relation_area` is multi-valued (names may
--                          themselves contain commas) -> M2M
-- Derived ML labels (form_label, area_label, split) are stored on acts for
-- convenient analysis; they are re-computable from the base columns.
-- ===========================================================================
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS act_area;
DROP TABLE IF EXISTS act_act_form;
DROP TABLE IF EXISTS acts;
DROP TABLE IF EXISTS areas;
DROP TABLE IF EXISTS act_form_tags;
DROP TABLE IF EXISTS legal_forces;
DROP TABLE IF EXISTS sections;
DROP TABLE IF EXISTS regions;
DROP TABLE IF EXISTS bodies;

CREATE TABLE bodies (
    body_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- adopting body as recorded (joint acts keep the combined string)
    body_type   TEXT NOT NULL                  -- rule-based type: Мәслихат, Әкімдік, Министрлік, Үкімет, ...
);

CREATE TABLE regions (
    region_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE sections (
    section_id    INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,        -- database_section (curated top-level area)
    is_area_class INTEGER NOT NULL DEFAULT 0   -- 1 if kept as its own class in task `area`
);

CREATE TABLE legal_forces (
    legal_force_id INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE
);

CREATE TABLE act_form_tags (
    tag_id      INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- e.g. 'Шешім', 'Өзгерту', 'Қағида'
    base_form   TEXT                           -- base legal form if the tag is one (Заң..Өкім), else NULL
);

CREATE TABLE areas (
    area_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE           -- legal_relation_area item
);

CREATE TABLE acts (
    law_id          TEXT PRIMARY KEY,
    source_id       INTEGER,
    title           TEXT NOT NULL,
    title_len_chars INTEGER NOT NULL,
    title_len_words INTEGER NOT NULL,
    date_adopted    TEXT,                      -- ISO yyyy-mm-dd
    date_modified   TEXT,
    date_published  TEXT,
    year            INTEGER,
    is_repealed     INTEGER NOT NULL CHECK (is_repealed IN (0, 1)),
    form_label      TEXT,                      -- task `form` label (primary legal form; NULL if act_form missing)
    area_label      TEXT,                      -- task `area` label (NULL if section missing / not in dataset)
    split           TEXT CHECK (split IN ('train', 'val', 'test') OR split IS NULL),  -- NULL = removed by leakage guard
    body_id         INTEGER REFERENCES bodies(body_id),
    region_id       INTEGER REFERENCES regions(region_id),
    section_id      INTEGER REFERENCES sections(section_id),
    legal_force_id  INTEGER REFERENCES legal_forces(legal_force_id)
);

CREATE TABLE act_act_form (
    law_id  TEXT    NOT NULL REFERENCES acts(law_id) ON DELETE CASCADE,
    tag_id  INTEGER NOT NULL REFERENCES act_form_tags(tag_id),
    PRIMARY KEY (law_id, tag_id)
);

CREATE TABLE act_area (
    law_id  TEXT    NOT NULL REFERENCES acts(law_id) ON DELETE CASCADE,
    area_id INTEGER NOT NULL REFERENCES areas(area_id),
    PRIMARY KEY (law_id, area_id)
);
