// Small DOM helpers shared by every view.

export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else if (name.startsWith("on") && typeof value === "function") node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function announce(message) {
  const region = document.getElementById("announcer");
  region.textContent = "";
  window.setTimeout(() => {
    region.textContent = message;
  }, 50);
}

export function showError(error) {
  const box = document.getElementById("app-error");
  box.textContent = error instanceof Error ? error.message : String(error);
  box.hidden = false;
}

export function clearError() {
  const box = document.getElementById("app-error");
  box.hidden = true;
  box.textContent = "";
}

export function formatDate(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

const TERMINATIONS = {
  checkmate: "checkmate",
  stalemate: "stalemate",
  insufficient_material: "insufficient material",
  seventyfive_moves: "the 75-move rule",
  fivefold_repetition: "fivefold repetition",
  fifty_moves: "the 50-move rule",
  threefold_repetition: "threefold repetition",
  max_plies: "the move limit",
  resignation: "resignation",
  engine_failure: "engine failure",
};

export function terminationText(termination) {
  return TERMINATIONS[termination] ?? termination ?? "";
}

export function resultFor(game, color) {
  if (game.result === "1/2-1/2") return "draw";
  if (game.result === "*") return "ongoing";
  const winner = game.result === "1-0" ? "white" : "black";
  return winner === color ? "win" : "loss";
}

export function playerLabel(player, opponents) {
  if (player.kind === "human") return "You";
  const match = opponents?.find(
    (item) => item.player.kind === player.kind && item.player.player_id === player.player_id,
  );
  return match ? match.label : player.player_id;
}

export function setTitle(section) {
  document.title = section ? `${section} · Chess Arena` : "Chess Arena";
}
