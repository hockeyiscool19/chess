// Hash router and the engine-availability chip.

import { api } from "./api.js";
import { renderGames } from "./games.js";
import { renderLadder } from "./ladder.js";
import { renderModels } from "./models.js";
import { renderPlay } from "./play.js";
import { clearError, el, showError } from "./ui.js";

const VIEWS = { play: renderPlay, ladder: renderLadder, models: renderModels, games: renderGames };

function parseRoute() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = hash.split("?");
  const [name = "play", id = null] = path.split("/").filter(Boolean);
  return { name: VIEWS[name] ? name : "play", id: id ? decodeURIComponent(id) : null, query: new URLSearchParams(query) };
}

async function route() {
  const current = parseRoute();
  for (const link of document.querySelectorAll("[data-nav]")) {
    if (link.dataset.nav === current.name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  const view = document.getElementById("view");
  view.replaceChildren(el("p", { class: "muted" }, "Loading…"));
  clearError();
  try {
    await VIEWS[current.name](view, current);
  } catch (error) {
    view.replaceChildren();
    showError(error);
  }
}

async function engineChip() {
  const chip = document.getElementById("engine-chip");
  try {
    const health = await api.health();
    chip.textContent = health.stockfish
      ? `Stockfish ready · ${health.models} model${health.models === 1 ? "" : "s"}`
      : `Stockfish not installed · ${health.models} models`;
  } catch {
    chip.textContent = "Server unreachable";
  }
}

window.addEventListener("hashchange", () => {
  route();
  engineChip();
  document.getElementById("main").focus({ preventScroll: true });
});
route();
engineChip();
