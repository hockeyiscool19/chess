// Ladder view: the ordered rungs and each model's best result against them.

import { api } from "./api.js";
import { el, setTitle } from "./ui.js";

const MATRIX_MODELS = 15;

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function rungsTable(ladder) {
  const rows = ladder.rungs.map((rung, index) =>
    el(
      "tr",
      {},
      el("td", { class: "num" }, String(index + 1)),
      el("th", { scope: "row" }, rung.label),
      el("td", {}, rung.description),
      el("td", { class: "num" }, String(rung.nominal_elo)),
    ),
  );
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      {},
      el("caption", {}, `${ladder.ladder_id} · ${ladder.games_per_rung} games per rung · beat a rung with ${pct(ladder.pass_score)} of the points`),
      el("thead", {}, el("tr", {}, el("th", { scope: "col", class: "num" }, "#"), el("th", { scope: "col" }, "Opponent"), el("th", { scope: "col" }, "How it plays"), el("th", { scope: "col", class: "num" }, "Nominal Elo"))),
      el("tbody", {}, rows),
    ),
  );
}

function resultCell(result) {
  if (!result) return el("td", { class: "muted" }, "·");
  const record = `+${result.wins} =${result.draws} −${result.losses}`;
  const score = result.wins + result.draws + result.losses ? (result.wins + 0.5 * result.draws) / (result.wins + result.draws + result.losses) : 0;
  return el(
    "td",
    { class: "score-cell" },
    el("span", { class: `pill ${result.passed ? "good" : "bad"}` }, result.passed ? "✓ beaten" : "✗ not beaten"),
    el("div", {}, `${pct(score)} `, el("span", { class: "muted" }, record)),
    el("span", { class: "score-bar", "aria-hidden": "true" }, el("span", { style: `width:${Math.round(score * 100)}%` })),
  );
}

function matrix(ladder, models, reports) {
  const shown = models.filter((item) => reports.has(item.card.model_id)).slice(-MATRIX_MODELS).reverse();
  if (!shown.length) {
    return el("p", { class: "muted" }, "No model has finished a ladder run yet. The chess institute publishes models and queues runs here.");
  }
  const rungs = ladder.rungs.slice(0, Math.max(...shown.map((item) => reports.get(item.card.model_id).results.length)) + 1);
  const head = el("tr", {}, el("th", { scope: "col" }, "Model"), el("th", { scope: "col", class: "num" }, "Score"), rungs.map((rung) => el("th", { scope: "col" }, rung.label)));
  const body = shown.map((item) => {
    const report = reports.get(item.card.model_id);
    const byRung = new Map(report.results.map((result) => [result.rung_id, result]));
    return el(
      "tr",
      {},
      el("th", { scope: "row" }, el("a", { href: `#/play?opponent=model:${encodeURIComponent(item.card.model_id)}` }, item.card.label)),
      el("td", { class: "num" }, report.ladder_score.toFixed(2)),
      rungs.map((rung) => resultCell(byRung.get(rung.rung_id))),
    );
  });
  return el("div", { class: "table-wrap" }, el("table", {}, el("caption", {}, "Best ladder run per model (newest models first)"), el("thead", {}, head), el("tbody", {}, body)));
}

export async function renderLadder(view) {
  setTitle("Ladder");
  const [ladders, models, runs] = await Promise.all([api.ladders(), api.models(), api.ladderRuns()]);
  const ladder = ladders.find((item) => item.ladder_id === "ladder-v1") ?? ladders[0];
  const reports = new Map();
  for (const run of runs) {
    const report = run.report;
    if (!report || !report.complete) continue;
    const id = report.player.player_id;
    const best = reports.get(id);
    if (!best || report.ladder_score > best.ladder_score) reports.set(id, report);
  }
  const active = runs.filter((run) => run.status === "running" || run.status === "queued");
  view.replaceChildren(
    el("h1", {}, "Engine ladder"),
    el("p", { class: "lede" }, "A model climbs rung by rung, from a random mover to full-strength Stockfish. It must beat each rung before it may face the next. Its ladder score is the number of rungs beaten plus its share of the points on the first rung it could not beat."),
    el("p", { role: "status", class: "muted" }, active.length ? `${active.length} ladder run${active.length === 1 ? "" : "s"} in progress.` : ""),
    el("section", { class: "card stack", "aria-labelledby": "results-title" }, el("h2", { id: "results-title" }, "Results"), matrix(ladder, models, reports)),
    el("section", { class: "card stack", "aria-labelledby": "rungs-title" }, el("h2", { id: "rungs-title" }, "Rungs"), rungsTable(ladder)),
  );
}
