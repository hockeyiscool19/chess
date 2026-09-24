// Play view: pick an opponent and color, then play on the accessible board.

import { api, parsePlayerKey, playerKey } from "./api.js";
import { Board } from "./board.js";
import {
  announce,
  clearError,
  el,
  playerLabel,
  resultFor,
  setTitle,
  showError,
  terminationText,
} from "./ui.js";

const GROUPS = [
  { label: "Built-in bots", test: (item) => item.player.kind === "engine" && !item.player.player_id.startsWith("stockfish") },
  { label: "Stockfish", test: (item) => item.player.player_id.startsWith("stockfish") },
  { label: "Institute models", test: (item) => item.player.kind === "model" },
];

function opponentSelect(opponents, preselect) {
  const select = el("select", { id: "opponent", name: "opponent", "aria-describedby": "opponent-help" });
  for (const group of GROUPS) {
    const members = opponents.filter(group.test);
    if (!members.length) continue;
    const optgroup = el("optgroup", { label: group.label });
    for (const item of members) {
      const elo = item.nominal_elo ? ` (about ${item.nominal_elo})` : "";
      const missing = item.available ? "" : " (not installed)";
      optgroup.append(
        el("option", { value: playerKey(item.player), disabled: !item.available }, `${item.label}${elo}${missing}`),
      );
    }
    select.append(optgroup);
  }
  const fallback = opponents.find((item) => item.player.player_id === "minimax-2");
  select.value = preselect || (fallback ? playerKey(fallback.player) : select.value);
  return select;
}

function moveList(sans) {
  const list = el("ol", { class: "moves", id: "moves", "aria-label": "Moves played" });
  for (let index = 0; index < sans.length; index += 2) {
    const current = index + 2 >= sans.length;
    list.append(
      el(
        "li",
        {},
        el("span", { class: "num" }, `${index / 2 + 1}.`),
        el("span", { class: `ply${current && index === sans.length - 1 ? " current" : ""}` }, sans[index]),
        el("span", { class: `ply${current && index + 1 === sans.length - 1 ? " current" : ""}` }, sans[index + 1] ?? ""),
      ),
    );
  }
  return list;
}

export async function renderPlay(view, route) {
  setTitle("Play");
  const opponents = await api.opponents();
  const state = { game: null, humanColor: "white", opponents };
  const boardHost = el("div", { id: "board" });
  const topStrip = el("div", { class: "player-strip" });
  const bottomStrip = el("div", { class: "player-strip" });
  const status = el("p", { class: "status-line", id: "status" }, "Choose an opponent and start a game.");
  const movesHost = el("div", {}, moveList([]));
  const flip = el("button", { type: "button" }, "Flip board");
  const resign = el("button", { type: "button", disabled: true }, "Resign");
  const select = opponentSelect(opponents, route.query.get("opponent"));
  const colors = ["white", "black", "random"].map((color) =>
    el(
      "label",
      {},
      el("input", { type: "radio", name: "color", value: color, checked: color === "white" }),
      color[0].toUpperCase() + color.slice(1),
    ),
  );
  const form = el(
    "form",
    { id: "new-game" },
    el("div", { class: "field" }, el("label", { for: "opponent" }, "Opponent"), select),
    el("p", { class: "muted", id: "opponent-help" }, "Bots and Stockfish levels are ordered by strength. Institute models appear once published."),
    el("fieldset", { class: "field" }, el("legend", {}, "Your color"), el("div", { class: "radio-row" }, colors)),
    el("div", { class: "button-row field" }, el("button", { type: "submit", class: "button-primary" }, "Start game")),
  );

  const board = new Board(boardHost, {
    label: "Chessboard. Use arrow keys to move between squares and Enter to pick up or drop a piece.",
    onMove: (uci) => play(uci),
  });

  function strips() {
    const game = state.game;
    const bottomColor = board.orientation;
    const topColor = bottomColor === "white" ? "black" : "white";
    for (const [strip, color] of [[topStrip, topColor], [bottomStrip, bottomColor]]) {
      strip.replaceChildren();
      if (!game) continue;
      const player = color === "white" ? game.game.white : game.game.black;
      const toMove = game.game.result === "*" && game.turn === color;
      const side = color === "white" ? "White" : "Black";
      strip.append(
        el(
          "span",
          { class: toMove ? "to-move" : "" },
          `${side}: ${playerLabel(player, opponents)}`,
          toMove ? el("span", { class: "visually-hidden" }, " (to move)") : null,
        ),
      );
    }
  }

  function describe(game) {
    const record = game.game;
    const opponent = record.white.kind === "human" ? record.black : record.white;
    const name = playerLabel(opponent, opponents);
    if (record.result !== "*") {
      const outcome = resultFor(record, state.humanColor);
      const why = terminationText(record.termination);
      const headline = { win: "You win", loss: `${name} wins`, draw: "Draw" }[outcome];
      return `${headline} by ${why}.`;
    }
    const yours = game.turn === state.humanColor;
    const check = game.check_square ? "Check! " : "";
    return yours ? `${check}Your move.` : `${check}${name} to move.`;
  }

  function show(game) {
    state.game = game;
    const ongoing = game.game.result === "*";
    board.setPosition({
      fen: game.fen,
      legalMoves: ongoing && game.turn === state.humanColor ? game.legal_moves : [],
      mover: ongoing ? state.humanColor : null,
      lastMove: game.last_move,
      checkSquare: game.check_square,
    });
    status.textContent = describe(game);
    status.classList.toggle("over", !ongoing);
    movesHost.replaceChildren(moveList(game.san_moves));
    const list = movesHost.querySelector(".moves");
    list.scrollTop = list.scrollHeight;
    resign.disabled = !ongoing;
    strips();
  }

  async function play(uci) {
    const before = state.game.san_moves.length;
    board.setBusy(true);
    status.replaceChildren(el("span", { class: "thinking" }, "Thinking…"));
    try {
      clearError();
      const game = await api.move(state.game.game.game_id, uci);
      board.setBusy(false);
      show(game);
      const played = game.san_moves.slice(before);
      const record = game.game;
      const opponent = playerLabel(record.white.kind === "human" ? record.black : record.white, opponents);
      const parts = [`You played ${played[0]}.`];
      if (played[1]) parts.push(`${opponent} played ${played[1]}.`);
      parts.push(describe(game));
      announce(parts.join(" "));
    } catch (error) {
      board.setBusy(false);
      showError(error);
      show(await api.game(state.game.game.game_id));
    }
  }

  async function load(gameId) {
    const game = await api.game(gameId);
    state.humanColor = game.game.white.kind === "human" ? "white" : "black";
    board.setOrientation(state.humanColor);
    show(game);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();
    const color = form.querySelector('input[name="color"]:checked').value;
    try {
      const game = await api.startGame(parsePlayerKey(select.value), color);
      history.replaceState(null, "", `#/play/${game.game.game_id}`);
      state.humanColor = game.game.white.kind === "human" ? "white" : "black";
      board.setOrientation(state.humanColor);
      show(game);
      announce(`New game against ${playerLabel(parsePlayerKey(select.value), opponents)}. ${describe(game)}`);
      board.focusOn(state.humanColor === "white" ? "e2" : "e7");
    } catch (error) {
      showError(error);
    }
  });
  flip.addEventListener("click", () => {
    board.flip();
    strips();
  });
  resign.addEventListener("click", async () => {
    if (!state.game) return;
    try {
      show(await api.resign(state.game.game.game_id));
      announce(describe(state.game));
    } catch (error) {
      showError(error);
    }
  });

  view.replaceChildren(
    el("h1", {}, "Play"),
    el(
      "div",
      { class: "play-layout" },
      el("section", { class: "board-column", "aria-label": "Board" }, topStrip, boardHost, bottomStrip),
      el(
        "div",
        { class: "stack" },
        el("section", { class: "card", "aria-labelledby": "new-game-title" }, el("h2", { id: "new-game-title" }, "New game"), form),
        el(
          "section",
          { class: "card stack", "aria-labelledby": "game-title" },
          el("h2", { id: "game-title" }, "Game"),
          status,
          movesHost,
          el("div", { class: "button-row" }, flip, resign),
        ),
      ),
    ),
  );
  board.setPosition({ fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" });
  if (route.id) await load(route.id);
}
