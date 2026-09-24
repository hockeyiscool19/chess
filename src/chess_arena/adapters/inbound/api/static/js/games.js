// Games view: recent games with filters, and a step-by-step replay viewer.

import { api } from "./api.js";
import { Board } from "./board.js";
import { el, formatDate, playerLabel, setTitle, terminationText } from "./ui.js";

const KINDS = [
  { value: "", label: "All games" },
  { value: "human", label: "Human games" },
  { value: "ladder", label: "Ladder games" },
  { value: "match", label: "Matches" },
];

function gamesTable(games, opponents) {
  if (!games.length) return el("p", { class: "muted" }, "No games yet.");
  const rows = games.map((game) =>
    el(
      "tr",
      {},
      el("td", {}, formatDate(game.created_at)),
      el("td", {}, playerLabel(game.white, opponents)),
      el("td", {}, playerLabel(game.black, opponents)),
      el("td", {}, game.result === "*" ? "in progress" : game.result),
      el("td", {}, terminationText(game.termination)),
      el("td", {}, game.context.opening ?? "—"),
      el("td", { class: "num" }, String(game.moves.length)),
      el("td", {}, el("a", { href: `#/games/${encodeURIComponent(game.game_id)}` }, "Replay")),
    ),
  );
  return el(
    "div",
    { class: "table-wrap" },
    el(
      "table",
      {},
      el("caption", { class: "visually-hidden" }, "Recent games"),
      el("thead", {}, el("tr", {}, ["Date", "White", "Black", "Result", "Ended by", "Opening", "Plies", ""].map((name, index) => el("th", { scope: "col", class: index === 6 ? "num" : null }, name || el("span", { class: "visually-hidden" }, "Replay link"))))),
      el("tbody", {}, rows),
    ),
  );
}

export async function renderGames(view, route) {
  if (route.id) return renderReplay(view, route.id);
  setTitle("Games");
  const opponents = await api.opponents();
  const select = el("select", { id: "kind" }, KINDS.map((kind) => el("option", { value: kind.value }, kind.label)));
  select.value = route.query.get("kind") ?? "";
  const host = el("div", {});
  const refresh = async () => {
    host.replaceChildren(el("p", { class: "muted" }, "Loading…"));
    host.replaceChildren(gamesTable(await api.games(select.value || null), opponents));
  };
  select.addEventListener("change", () => {
    history.replaceState(null, "", select.value ? `#/games?kind=${select.value}` : "#/games");
    refresh();
  });
  view.replaceChildren(
    el("h1", {}, "Games"),
    el("section", { class: "card stack" }, el("div", { class: "field" }, el("label", { for: "kind" }, "Show"), select), host),
  );
  await refresh();
}

async function renderReplay(view, gameId) {
  const [replay, opponents] = await Promise.all([api.replay(gameId), api.opponents()]);
  const game = replay.game;
  const white = playerLabel(game.white, opponents);
  const black = playerLabel(game.black, opponents);
  setTitle(`${white} vs ${black}`);
  const boardHost = el("div", {});
  const board = new Board(boardHost, { label: "Replay board", interactive: false });
  const position = el("p", { class: "status-line", role: "status" });
  const frames = replay.frames;
  let ply = frames.length - 1;
  const moveButtons = [];
  const list = el("ol", { class: "moves", "aria-label": "Moves" });
  for (let index = 1; index < frames.length; index += 2) {
    const make = (frameIndex) => {
      if (frameIndex >= frames.length) return el("span", {});
      const button = el("button", { type: "button", class: "ply", onclick: () => go(frameIndex) }, frames[frameIndex].san);
      moveButtons[frameIndex] = button;
      return button;
    };
    list.append(el("li", {}, el("span", { class: "num" }, `${(index + 1) / 2}.`), make(index), make(index + 1)));
  }

  function go(target) {
    ply = Math.max(0, Math.min(frames.length - 1, target));
    const frame = frames[ply];
    board.setPosition({ fen: frame.fen, lastMove: frame.uci });
    for (const [index, button] of moveButtons.entries()) {
      if (button) button.setAttribute("aria-current", index === ply ? "step" : "false");
    }
    position.textContent = ply === 0 ? "Start position" : `After ${Math.ceil(ply / 2)}${ply % 2 ? "." : "..."} ${frame.san} (ply ${ply} of ${frames.length - 1})`;
  }

  const controls = el(
    "div",
    { class: "replay-controls" },
    el("button", { type: "button", onclick: () => go(0) }, "⏮ First"),
    el("button", { type: "button", onclick: () => go(ply - 1) }, "◀ Previous"),
    el("button", { type: "button", onclick: () => go(ply + 1) }, "Next ▶"),
    el("button", { type: "button", onclick: () => go(frames.length - 1) }, "Last ⏭"),
    el("button", { type: "button", onclick: () => board.flip() }, "Flip board"),
  );
  controls.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") go(ply - 1);
    if (event.key === "ArrowRight") go(ply + 1);
  });
  const copy = el("button", { type: "button" }, "Copy PGN");
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(replay.pgn);
    copy.textContent = "Copied";
  });
  const result = game.result === "*" ? "in progress" : `${game.result} by ${terminationText(game.termination)}`;
  view.replaceChildren(
    el("p", {}, el("a", { href: "#/games" }, "← All games")),
    el("h1", {}, `${white} vs ${black}`),
    el("p", { class: "muted" }, `${formatDate(game.created_at)} · ${game.context.kind} game${game.context.opening ? ` · ${game.context.opening}` : ""} · ${result}`),
    el(
      "div",
      { class: "play-layout" },
      el("section", { class: "board-column", "aria-label": "Replay" }, boardHost, controls, position),
      el(
        "div",
        { class: "stack" },
        el("section", { class: "card stack", "aria-labelledby": "moves-title" }, el("h2", { id: "moves-title" }, "Moves"), list),
        el("section", { class: "card stack", "aria-labelledby": "pgn-title" }, el("h2", { id: "pgn-title" }, "PGN"), el("pre", { class: "pgn" }, replay.pgn), copy),
      ),
    ),
  );
  go(ply);
}
