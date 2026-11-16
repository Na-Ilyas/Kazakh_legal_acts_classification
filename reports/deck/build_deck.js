// Build reports/presentation.pptx from reports/deck/deck_data.json + reports/figures/*.png
// Usage (from project root):
//   .venv/bin/python reports/deck/deck_data.py && NODE_PATH=<dir with pptxgenjs> node reports/deck/build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = (f) => path.join(ROOT, "reports", "figures", f);
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(ROOT, "reports", "presentation.pptx");

const THEME = {
  name: "Kazakh Sky", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2430", lt1: "FFFFFF", dk2: "0E2A3B", lt2: "EEF5F8",
            accent1: "00AFCA", accent2: "FEC50C", accent3: "5C7A8A", accent4: "E4572E",
            accent5: "3E8E5E", accent6: "9AA5AE", hlink: "0077A3", folHlink: "5C7A8A" } };
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Tiny Transformer for Kazakh Legal Act Classification";
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "DARK", background: { color: HEX.dk2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.2, w: 11.9, h: 1.6,
    fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, valign: "middle" },
    text: "" } }],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
      fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.text2, valign: "middle" }, text: "" } },
    { text: { text: "Topic #24 · Programming for AI · AAI-2603M", options: { x: 0.6, y: 7.0, w: 8, h: 0.3,
      fontSize: 10, color: C.accent3 } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent3 },
});

const S = (sectionTitle, layout = "CONTENT") => pres.addSlide({ masterName: layout, sectionTitle });
const txt = (slide, text, opts) => slide.addText(text, { isTextBox: true, fontSize: 15, color: C.text1,
  valign: "top", margin: 0, ...opts });
const bullets = (items, size = 15) => items.map((t, i) => ({ text: t,
  options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size } }));
const card = (slide, x, y, w, h, fill = C.background2, name = "card") => slide.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
const stat = (slide, x, y, w, big, small, color = C.accent1) => {
  txt(slide, big, { x, y, w, h: 0.75, fontSize: 30, bold: true, color, fontFace: THEME.headFontFace, fit: "shrink" });
  txt(slide, small, { x, y: y + 0.8, w, h: 0.9, fontSize: 13, color: C.text1 });
};
const img = (slide, file, x, y, w, h) => {
  if (fs.existsSync(FIG(file))) slide.addImage({ path: FIG(file), x, y, w, h,
    sizing: { type: "contain", w, h }, altText: file.replace(/[_.]/g, " ") });
  else txt(slide, `[missing figure ${file}]`, { x, y, w, h: 0.4, color: C.accent4 });
};
const pct = (v) => (v == null ? "–" : v.toFixed(3));
const fmtParams = (n) => (n == null || n === 0 ? "–" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : (n / 1e3).toFixed(0) + "k");
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };

// ---------- 1. title ----------
pres.addSection({ title: "Intro" });
let s = S("Intro", "DARK");
s.addText("Tiny Transformer for Kazakh Legal Act Classification", { placeholder: "title" });
txt(s, "Can a 5M-parameter model trained from scratch compete with pretrained encoders on Kazakh legal titles?",
  { x: 0.7, y: 3.9, w: 11.5, h: 0.6, fontSize: 18, color: C.accent1, italic: true });
txt(s, "Ilyas Nagiyev · Beksultan Aliyev · Yernur Baimukhambetov  |  AAI-2603M\nProgramming for AI · Topic #24 · Astana IT University",
  { x: 0.7, y: 5.6, w: 11.5, h: 0.9, fontSize: 14, color: C.background1 });
s.addNotes("Introduce the team and the topic: classify Kazakh legal acts from their titles with a tiny transformer.");

// ---------- 2. task ----------
s = S("Intro");
s.addText("Two tasks, one input: the act's title", { placeholder: "title" });
card(s, 0.6, 1.45, 7.3, 2.3);
txt(s, "Example title (Kazakh)", { x: 0.85, y: 1.6, w: 6.8, h: 0.3, fontSize: 12, color: C.accent3, bold: true });
txt(s, "«Қазақстан Республикасы Ішкі істер министрінің кейбір бұйрықтарының күшін жою туралы»",
  { x: 0.85, y: 1.95, w: 6.8, h: 0.9, fontSize: 17, italic: true, color: C.text2 });
txt(s, [{ text: "form → ", options: { bold: true, color: C.accent1 } }, { text: "Бұйрық (order)     " },
        { text: "area → ", options: { bold: true, color: C.accent1 } }, { text: "Constitutional order & public administration" }],
  { x: 0.85, y: 3.0, w: 6.8, h: 0.6, fontSize: 14 });
stat(s, 8.4, 1.45, 2.1, "7", "act forms\nЗаң · Жарлық · Қаулы · Шешім · Бұйрық · Өкім · Other");
stat(s, 10.8, 1.45, 2.0, "33", "regulation areas\n(database_section)");
txt(s, "Research questions", { x: 0.6, y: 4.15, w: 6, h: 0.4, fontSize: 18, bold: true, color: C.text2,
  fontFace: THEME.headFontFace });
txt(s, bullets([
  "How close does a tiny transformer from scratch (own BPE, ≤10M params) get to TF-IDF + LR and to fine-tuned compact pretrained encoders (≤120M)?",
  "What does the quality cost in parameters, megabytes and CPU latency per 1000 titles?",
  "How much do scores depend on the evaluation protocol: temporal split, templated near-duplicates?",
]), { x: 0.6, y: 4.6, w: 12.1, h: 2.2 });

// ---------- 3. data ----------
pres.addSection({ title: "Data" });
s = S("Data");
s.addText("215,889 legal acts, handled within a 3 GB RAM budget", { placeholder: "title" });
const io = D.io_benchmark || {};
const mem = io.memory || {};
stat(s, 0.6, 1.5, 3.9, `${Math.round(mem.object_dtype_mb ?? 637)} → ${Math.round(mem.optimized_full_frame_mb ?? 201)} MB`,
  `in-memory size after dtype optimisation (−${mem.reduction_vs_object_pct ?? 68.4}%): categories, bools, datetimes, Int32`);
stat(s, 4.75, 1.5, 3.9, `${Math.round(io.csv_size_mb ?? 321)} → ${Math.round(io.parquet_size_mb ?? 26)} MB`,
  "CSV → Parquet (zstd), written chunk by chunk; reading 5 columns is 7× faster");
stat(s, 8.9, 1.5, 3.9, "43.2 %", "of regulatory_body values are the placeholder «ЖОҚ»; a naive isna() reports 0.01 %", C.accent4);
card(s, 0.6, 3.5, 12.1, 3.2);
txt(s, "Audit and cleaning decisions", { x: 0.85, y: 3.65, w: 11.5, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
txt(s, bullets([
  "Read with chunksize = 20,000; placeholders (ЖОҚ, НЕТ, undefined, numeric junk codes) are mapped to NA and counted separately.",
  "act_form is an unordered tag set (863 combinations of 76 tags) → the primary legal instrument is taken by hierarchy; modifiers (Өзгерту, Жоба …) go to a M2M table.",
  "legal_relation_area: names contain commas → a vocabulary-aware splitter recovers 336 areas (a naive split gives 450 fragments).",
  "Titles are highly templated: 76,460 rows share their exact title with another act.",
], 14), { x: 0.85, y: 4.1, w: 11.6, h: 2.5 });

// ---------- 4. SQL ----------
s = S("Data");
s.addText("Normalised SQLite database with 17 analytic queries", { placeholder: "title" });
img(s, "er_diagram.png", 0.5, 1.35, 7.4, 5.5);
card(s, 8.2, 1.4, 4.5, 5.4);
txt(s, "Schema", { x: 8.45, y: 1.55, w: 4, h: 0.35, fontSize: 16, bold: true, color: C.text2 });
txt(s, bullets(["acts (fact) + bodies, regions, sections, legal_forces",
  "M2M: act_act_form, act_area", "experiments + predictions: every model logs here"], 13),
  { x: 8.45, y: 1.95, w: 4.1, h: 1.6 });
txt(s, "Queries use", { x: 8.45, y: 3.6, w: 4, h: 0.35, fontSize: 16, bold: true, color: C.text2 });
txt(s, bullets(["JOIN (incl. self-join), GROUP BY / HAVING", "CTEs and subqueries",
  "Window functions: LAG, RANK, DENSE_RANK, running SUM", "Results → pandas → EDA figures, leaderboard, error analysis"], 13),
  { x: 8.45, y: 4.0, w: 4.1, h: 2.6 });

// ---------- 5. label shift ----------
s = S("Data");
s.addText("Temporal split: train ≤2021 · val 2022 · test ≥2023", { placeholder: "title" });
const cd = ((D.split_stats || {}).class_distribution || {}).form || {};
const forms = Object.keys(cd);
const share = (split) => { const tot = forms.reduce((a, f) => a + cd[f][split], 0);
  return forms.map((f) => +(100 * cd[f][split] / tot).toFixed(1)); };
if (forms.length) {
  s.addChart(pres.charts.BAR, [
    { name: "train (≤2021)", labels: forms, values: share("train") },
    { name: "test (≥2023)", labels: forms, values: share("test") }],
    { x: 0.5, y: 1.4, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent3, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFormatCode: "0.0",
      showTitle: true, title: "Share of each act form, %", titleFontSize: 14, titleColor: HEX.dk2,
      showLegend: true, legendPos: "t", legendFontSize: 12, catAxisLabelFontSize: 12, valAxisHidden: true,
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: HEX.dk1, ...chartText });
}
card(s, 8.5, 1.4, 4.2, 5.3);
stat(s, 8.75, 1.6, 3.8, "36 % → 64 %", "share of Шешім (local decisions) from train to test");
txt(s, bullets(["Models must predict the future: no act from 2022+ is seen in training.",
  "Accuracy is dominated by one class → macro-F1 is the headline metric.",
  "Area shifts too: finance 26.4 % → 44.1 %."], 14), { x: 8.75, y: 3.4, w: 3.8, h: 3.2 });

// ---------- 6. leakage ----------
s = S("Data");
s.addText("Templated titles leak → a near-duplicate-free test set", { placeholder: "title" });
const lk = D.leakage_report || {};
const nd = ((lk.near_duplicates || {}).test) || {};
stat(s, 0.6, 1.5, 3.0, `${nd.pct_ge_0_9 ?? nd["pct_ge_0.9"] ?? 54.2} %`, "of test titles have a train title with char-n-gram cosine ≥ 0.9 (digits masked)", C.accent4);
stat(s, 0.6, 3.1, 3.0, "20.6 %", "are identical to a train title up to the digits (years, numbers)", C.accent4);
stat(s, 0.6, 4.7, 3.0, "17,629", "test_strict titles (45.8 %): no digit-masked match, no neighbour ≥ 0.9", C.accent5);
img(s, "leakage_max_cosine_hist.png", 3.9, 1.35, 8.8, 5.4);

// ---------- 7. models ----------
pres.addSection({ title: "Models" });
s = S("Models");
s.addText("Model line-up: from 0 to 118M parameters", { placeholder: "title" });
const cols = [
  ["Baselines", ["Majority class", "char 1-NN (template matching)", "TF-IDF word / char / stem / word+char + logistic regression"]],
  ["Tiny transformer (scratch)", ["Own byte-level BPE (8k, train titles only)", "Own multi-head attention, pre-LN, 2–4 layers", "1.4M–5.2M params", "+ MLM pretraining, + distillation"]],
  ["Compact pretrained", ["multilingual-e5-small (118M)", "kaz-roberta (83.5M, Kazakh)", "KazBERT (110.7M, Kazakh)", "frozen e5 embeddings + LR"]],
  ["Compression & shift", ["Kazakh vocab pruning: e5 118M → 25.5M", "mT5-small seq2seq pruned to 58.7M", "Prior-shift adjustment, ensembles", "LLM teacher experiment"]],
];
cols.forEach(([head, items], i) => {
  const x = 0.6 + i * 3.08;
  card(s, x, 1.45, 2.85, 5.3, i === 1 ? "D9F3F8" : C.background2);
  txt(s, head, { x: x + 0.2, y: 1.6, w: 2.5, h: 0.75, fontSize: 16, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  txt(s, bullets(items, 13), { x: x + 0.2, y: 2.45, w: 2.5, h: 4.1 });
});

// ---------- 8. tiny transformer ----------
s = S("Models");
s.addText("The tiny transformer, written from scratch", { placeholder: "title" });
const blocks = ["Own BPE tokens (8k vocab, max_len 80)", "Token + learned position embeddings",
  "N × [ LayerNorm → own multi-head self-attention → + ]", "N × [ LayerNorm → GELU feed-forward → + ]",
  "Masked mean pooling", "Linear head (7 or 33 classes)"];
blocks.forEach((b, i) => {
  const y = 1.45 + i * 0.88;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y, w: 5.6, h: 0.66, rectRadius: 0.08,
    fill: { color: i === 2 || i === 3 ? C.accent1 : C.background2 }, line: { color: C.accent1 }, objectName: `arch${i}` });
  txt(s, b, { x: 0.95, y: y + 0.15, w: 5.3, h: 0.4, fontSize: 14, align: "center",
    color: i === 2 || i === 3 ? C.background1 : C.text2, bold: i === 2 || i === 3 });
});
card(s, 7.0, 1.45, 5.7, 5.3);
txt(s, "Training recipe (all choices on val)", { x: 7.25, y: 1.6, w: 5.2, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
txt(s, bullets(["AdamW, warm-up + cosine, bf16, gradient clipping, early stopping on val macro-F1",
  "Loss chosen on val: label smoothing 0.1 beat class weights and plain CE",
  "Ablations: 2 vs 4 layers, d 256 vs 128, dropout 0.1 vs 0.3, BPE 8k vs 4k",
  "MLM pretraining on train-period titles + descriptions",
  "Knowledge distillation: α·T²·KL(teacher‖student) + (1−α)·CE",
  "Largest model: 5.2M params (assert ≤ 10M), 21 MB"], 13), { x: 7.25, y: 2.1, w: 5.3, h: 4.5 });

// ---------- 9/10. results charts ----------
pres.addSection({ title: "Results" });
for (const task of ["form", "area"]) {
  const rows = (D.tasks[task] || []).filter((r) => r.model !== "majority_dummy");
  s = S("Results");
  s.addText(task === "form" ? "Results: act form (7 classes)" : "Results: regulation area (33 classes)", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "test macro-F1", labels: rows.map((r) => r.label), values: rows.map((r) => r.test) },
    { name: "test_strict macro-F1", labels: rows.map((r) => r.label), values: rows.map((r) => r.test_strict) }],
    { x: 0.5, y: 1.3, w: 8.4, h: 5.55, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.accent1, HEX.accent3],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 9,
      valAxisMinVal: task === "form" ? 0.6 : 0.45, valAxisMaxVal: task === "form" ? 0.9 : 0.66, valAxisHidden: true,
      catAxisLabelFontSize: 11, catAxisOrientation: "maxMin", showLegend: true, legendPos: "t", legendFontSize: 11,
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: HEX.dk1, ...chartText });
  const best = rows.filter((r) => !r.model.includes("trainval") && !r.model.startsWith("ensemble")).reduce((a, r) => (r.test_strict > (a?.test_strict ?? 0) ? r : a), null);
  const tiny = rows.find((r) => r.label.startsWith("Tiny 4L + MLM + KD"));
  const tfidf = rows.find((r) => r.model === "tfidf_word_char_lr");
  card(s, 9.2, 1.4, 3.5, 5.4);
  if (best) stat(s, 9.4, 1.55, 3.2, pct(best.test_strict), `best single model ≤120M (test_strict): ${best.label}, ${fmtParams(best.n_params)} params`);
  if (tiny) stat(s, 9.4, 3.3, 3.2, pct(tiny.test_strict), `tiny transformer (${fmtParams(tiny.n_params)}) vs TF-IDF ${tfidf ? pct(tfidf.test_strict) : "–"}`, C.accent5);
  txt(s, "Axis truncated to show differences; majority-class baseline: " + pct((D.tasks[task].find((r) => r.model === "majority_dummy") || {}).test),
    { x: 9.4, y: 5.3, w: 3.2, h: 1.2, fontSize: 11, color: C.accent3 });
}

// ---------- 11. efficiency ----------
s = S("Results");
s.addText("Quality vs cost", { placeholder: "title" });
const effModels = D.efficiency || [];
const head = ["model", "params", "size, MB", "CPU ms / 1000 titles", "form F1 (strict)", "area F1 (strict)"].map((t) => ({ text: t,
  options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 13 } }));
s.addTable([head, ...effModels.map((r) => [r.label, fmtParams(r.n_params), r.size_mb == null ? "–" : Math.round(r.size_mb).toString(),
  r.latency == null ? "–" : Math.round(r.latency).toString(), pct(r.form), pct(r.area)].map((t, j) => ({ text: t,
  options: { fontSize: 13, bold: j === 0, fill: { color: r.highlight ? "D9F3F8" : HEX.lt1 } } })))],
  { x: 0.6, y: 1.45, w: 12.1, colW: [3.9, 1.3, 1.3, 2.1, 1.75, 1.75], rowH: 0.45,
    border: { type: "solid", color: "D5DEE3", pt: 0.5 }, color: HEX.dk1 });
txt(s, D.efficiency_note || "", { x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 12, color: C.accent3 });

// ---------- 12. what helped ----------
s = S("Results");
s.addText("What helped, and what did not", { placeholder: "title" });
card(s, 0.6, 1.45, 5.95, 5.3, "E3F2E8");
card(s, 6.75, 1.45, 5.95, 5.3, "FBE6E0");
txt(s, "Helped", { x: 0.85, y: 1.6, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.accent5 });
txt(s, "Did not help", { x: 7.0, y: 1.6, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.accent4 });
txt(s, bullets(D.helped || ["(filled from final results)"], 13), { x: 0.85, y: 2.1, w: 5.5, h: 4.5 });
txt(s, bullets(D.not_helped || ["(filled from final results)"], 13), { x: 7.0, y: 2.1, w: 5.5, h: 4.5 });

// ---------- 13. errors ----------
s = S("Results");
s.addText("Error analysis", { placeholder: "title" });
img(s, D.confusion_figure || "confusion_form_kaz_roberta_llrd_rec.png", 0.5, 1.3, 6.4, 5.5);
txt(s, bullets(D.error_points || ["(filled from error analysis)"], 14), { x: 7.2, y: 1.5, w: 5.5, h: 5.2 });

// ---------- 14. demo ----------
pres.addSection({ title: "Wrap-up" });
s = S("Wrap-up");
s.addText("Demo: python main.py → 11. Classify your own title", { placeholder: "title" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.4, w: 12.1, h: 5.35, rectRadius: 0.08,
  fill: { color: "11161C" }, line: { color: "11161C" }, objectName: "terminal" });
txt(s, D.demo_text || "(demo output)", { x: 0.85, y: 1.6, w: 11.6, h: 5.0, fontFace: "Courier New", fontSize: 12,
  color: "E8EEF2" });

// ---------- 15. conclusions ----------
s = S("Wrap-up", "DARK");
txt(s, "Conclusions", { x: 0.7, y: 0.5, w: 11.9, h: 0.9, fontSize: 36, bold: true, color: C.background1, fontFace: THEME.headFontFace });
txt(s, bullets(D.conclusions || ["(filled from final results)"], 16), { x: 0.7, y: 1.7, w: 7.6, h: 5.2, color: C.background1 });
txt(s, "Team contributions", { x: 8.8, y: 1.7, w: 4, h: 0.4, fontSize: 16, bold: true, color: C.accent2 });
txt(s, bullets(D.team || ["Ilyas Nagiyev", "Beksultan Aliyev", "Yernur Baimukhambetov"], 13),
  { x: 8.8, y: 2.2, w: 4, h: 4.5, color: C.background1 });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();
