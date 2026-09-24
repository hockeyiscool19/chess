// Models view: the climb chart (ladder score by training order) and the model table.

import { api } from "./api.js";
import { el, formatDate, setTitle } from "./ui.js";

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 760;
const HEIGHT = 300;
const PAD = { left: 132, right: 16, top: 12, bottom: 32 };

function svg(tag, attributes = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  return node;
}

function climbChart(models, rungs) {
  const scored = models
    .map((item, index) => ({ item, index, score: item.best_ladder_score }))
    .filter((point) => point.score !== null && point.score !== undefined);
  if (!scored.length) return el("p", { class: "muted" }, "No evaluated models yet.");
  let best = -Infinity;
  const champions = scored.map((point) => {
    best = Math.max(best, point.score);
    return { ...point, best };
  });
  const top = Math.min(rungs.length, Math.max(3, Math.ceil(best) + 1));
  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;
  const last = Math.max(1, models.length - 1);
  const x = (index) => PAD.left + (models.length === 1 ? plotWidth / 2 : (index / last) * plotWidth);
  const y = (score) => PAD.top + plotHeight - (score / top) * plotHeight;

  const root = svg("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img", "aria-labelledby": "climb-title climb-desc" });
  const title = svg("title", { id: "climb-title" });
  title.textContent = "Ladder score by model, in training order";
  const desc = svg("desc", { id: "climb-desc" });
  const champion = champions[champions.length - 1];
  desc.textContent = `${scored.length} evaluated models. Best score ${best.toFixed(2)} by ${champion.item.card.label}. The table below lists every value.`;
  root.append(title, desc);

  const grid = svg("g", { class: "grid" });
  const axis = svg("g", { class: "axis" });
  for (let level = 0; level <= top; level += 1) {
    grid.append(svg("line", { x1: PAD.left, x2: WIDTH - PAD.right, y1: y(level), y2: y(level) }));
    const label = svg("text", { x: PAD.left - 8, y: y(level) + 4, "text-anchor": "end" });
    label.textContent = level === 0 ? "0 · none beaten" : `${level} · ${rungs[level - 1].label}`;
    axis.append(label);
  }
  const xLabel = svg("text", { x: PAD.left + plotWidth / 2, y: HEIGHT - 6, "text-anchor": "middle" });
  xLabel.textContent = "Models, oldest to newest";
  axis.append(xLabel);
  root.append(grid, axis);

  const path = champions.map((point, index) => `${index ? "L" : "M"}${x(point.index)},${y(point.best)}`).join(" ");
  root.append(svg("path", { class: "series-line", d: path }));

  const wrap = el("div", { class: "chart-wrap" });
  const tooltip = el("div", { class: "tooltip", hidden: true });
  for (const point of champions) {
    const isChampion = point.score === point.best;
    const cx = x(point.index);
    const cy = y(point.score);
    root.append(svg("circle", { class: `dot${isChampion ? " champion" : ""}`, cx, cy, r: 5 }));
    const hit = svg("circle", { class: "hit", cx, cy, r: 14, tabindex: 0, "aria-label": `${point.item.card.label}: score ${point.score.toFixed(2)}` });
    const showTip = () => {
      const card = point.item.card;
      tooltip.replaceChildren(
        el("strong", {}, card.label),
        el("div", {}, `Ladder score ${point.score.toFixed(2)}`),
        el("div", {}, `Beat up to: ${point.item.highest_beaten ?? "none"}`),
        el("div", {}, `Elo estimate ${point.item.best_elo_estimate ?? "–"}`),
      );
      tooltip.hidden = false;
      const box = root.getBoundingClientRect();
      const scale = box.width / WIDTH;
      tooltip.style.left = `${Math.min(cx * scale + 12, box.width - 180)}px`;
      tooltip.style.top = `${Math.max(0, cy * scale - 70)}px`;
    };
    hit.addEventListener("pointerenter", showTip);
    hit.addEventListener("focus", showTip);
    hit.addEventListener("pointerleave", () => { tooltip.hidden = true; });
    hit.addEventListener("blur", () => { tooltip.hidden = true; });
    root.append(hit);
  }
  wrap.append(root, tooltip);
  const legend = el(
    "div",
    { class: "legend" },
    el("span", { class: "key" }, el("span", { class: "swatch-line", "aria-hidden": "true" }), "Best so far (champion)"),
    el("span", { class: "key" }, el("span", { class: "swatch-dot", "aria-hidden": "true" }), "Each model's best run"),
  );
  return el(
    "figure",
    { class: "chart-figure chart" },
    legend,
    wrap,
    el("figcaption", {}, "Each whole number is one more rung beaten; the fraction is the share of points taken on the next rung."),
  );
}

function recipeSummary(recipe) {
  const data = recipe.data;
  const games = data.random_games + data.bot_games + data.stockfish_games + data.self_play_games;
  const hidden = recipe.network.hidden.join("×");
  if (!games) return `Hand-written evaluation, depth ${recipe.search.depth}`;
  return `${hidden} net · ${games} games · teacher ${recipe.labels.teacher} · depth ${recipe.search.depth} · blend ${recipe.search.material_blend}`;
}

function modelsTable(models) {
  const rows = [...models].reverse().map((item) => {
    const card = item.card;
    return el(
      "tr",
      {},
      el("th", { scope: "row" }, card.label, el("div", { class: "muted" }, card.model_id)),
      el("td", {}, recipeSummary(card.recipe)),
      el("td", {}, card.parent_id ?? "—"),
      el("td", { class: "num" }, item.best_ladder_score === null || item.best_ladder_score === undefined ? "not run" : item.best_ladder_score.toFixed(2)),
      el("td", {}, item.highest_beaten ?? "—"),
      el("td", { class: "num" }, item.best_elo_estimate ?? "—"),
      el("td", {}, formatDate(card.created_at)),
      el("td", {}, el("a", { href: `#/play?opponent=model:${encodeURIComponent(card.model_id)}` }, `Play ${card.label}`)),
    );
  });
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      {},
      el("caption", {}, "Every published model, newest first"),
      el(
        "thead",
        {},
        el("tr", {}, ["Model", "Recipe", "Parent", "Ladder score", "Highest rung beaten", "Elo estimate", "Created", "Play"].map((name, index) => el("th", { scope: "col", class: index === 3 || index === 5 ? "num" : null }, name))),
      ),
      el("tbody", {}, rows),
    ),
  );
}

export async function renderModels(view) {
  setTitle("Models");
  const [models, ladders] = await Promise.all([api.models(), api.ladders()]);
  const ladder = ladders.find((item) => item.ladder_id === "ladder-v1") ?? ladders[0];
  const scored = models.filter((item) => item.best_ladder_score !== null && item.best_ladder_score !== undefined);
  const champion = scored.reduce((best, item) => (!best || item.best_ladder_score > best.best_ladder_score ? item : best), null);
  const kpis = el(
    "div",
    { class: "kpis" },
    el("div", { class: "card kpi" }, el("div", { class: "value" }, String(models.length)), el("div", { class: "label" }, "models published")),
    el("div", { class: "card kpi" }, el("div", { class: "value" }, champion ? champion.best_ladder_score.toFixed(2) : "–"), el("div", { class: "label" }, champion ? `best score (${champion.card.label})` : "best score")),
    el("div", { class: "card kpi" }, el("div", { class: "value" }, champion?.highest_beaten ?? "–"), el("div", { class: "label" }, "highest rung beaten")),
    el("div", { class: "card kpi" }, el("div", { class: "value" }, champion?.best_elo_estimate ?? "–"), el("div", { class: "label" }, "Elo estimate of the champion")),
  );
  view.replaceChildren(
    el("h1", {}, "Models"),
    el("p", { class: "lede" }, "Models are value networks trained by the chess institute and played through a small alpha-beta search. Each one records its recipe and parent, so the lineage from the hand-written baseline to the current champion stays visible."),
    kpis,
    el("section", { class: "card stack", "aria-labelledby": "climb-heading" }, el("h2", { id: "climb-heading" }, "The climb"), climbChart(models, ladder.rungs)),
    el("section", { class: "card stack", "aria-labelledby": "models-heading" }, el("h2", { id: "models-heading" }, "All models"), models.length ? modelsTable(models) : el("p", { class: "muted" }, "No models yet.")),
  );
}
