// Accessible chessboard: 64 native buttons with roving focus, click-to-move,
// drag-and-drop, a promotion dialog, and screen-reader labels for every square.
// The server supplies legal moves, so the board never needs chess rules itself.

const TEXT = "\uFE0E"; // request text (not emoji) presentation for chess glyphs
const GLYPHS = { k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F" };
const NAMES = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
const FILES = "abcdefgh";
const DRAG_THRESHOLD = 5;

export function parseFen(fen) {
  const placement = fen.split(" ")[0];
  const pieces = new Map();
  placement.split("/").forEach((row, rowIndex) => {
    const rank = 8 - rowIndex;
    let file = 0;
    for (const char of row) {
      if (/\d/.test(char)) {
        file += Number(char);
        continue;
      }
      const color = char === char.toUpperCase() ? "white" : "black";
      pieces.set(`${FILES[file]}${rank}`, { color, type: char.toLowerCase() });
      file += 1;
    }
  });
  return pieces;
}

export function pieceName(piece) {
  return `${piece.color} ${NAMES[piece.type]}`;
}

export function glyph(piece) {
  return GLYPHS[piece.type] + TEXT;
}

function squaresFor(orientation) {
  const ranks = orientation === "white" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = orientation === "white" ? [...FILES] : [...FILES].reverse();
  const squares = [];
  for (const rank of ranks) for (const file of files) squares.push(`${file}${rank}`);
  return squares;
}

function isDark(square) {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]);
  return (file + rank) % 2 === 0;
}

export function choosePromotion() {
  const dialog = document.getElementById("promotion");
  return new Promise((resolve) => {
    const done = () => {
      dialog.removeEventListener("close", done);
      const value = dialog.returnValue;
      resolve(["q", "r", "b", "n"].includes(value) ? value : null);
    };
    dialog.returnValue = "";
    dialog.addEventListener("close", done);
    dialog.showModal();
    dialog.querySelector("button")?.focus();
  });
}

export class Board {
  constructor(element, { label = "Chessboard", onMove = null, interactive = true } = {}) {
    this.element = element;
    this.onMove = onMove;
    this.interactive = interactive;
    this.orientation = "white";
    this.pieces = new Map();
    this.legal = [];
    this.mover = null;
    this.lastMove = null;
    this.checkSquare = null;
    this.selected = null;
    this.busy = false;
    this.focusSquare = "e2";
    this.drag = null;
    this.suppressClick = false;
    element.classList.add("board");
    element.setAttribute("role", "group");
    element.setAttribute("aria-label", label);
    element.addEventListener("keydown", (event) => this.onKey(event));
    element.addEventListener("click", (event) => this.onClick(event));
    if (interactive) {
      element.addEventListener("pointerdown", (event) => this.onPointerDown(event));
      element.addEventListener("pointermove", (event) => this.onPointerMove(event));
      element.addEventListener("pointerup", (event) => this.onPointerUp(event));
      element.addEventListener("pointercancel", () => this.cancelDrag());
    }
    this.build();
  }

  build() {
    this.element.replaceChildren();
    this.buttons = new Map();
    const order = squaresFor(this.orientation);
    order.forEach((square, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `sq ${isDark(square) ? "dark" : "light"}`;
      button.dataset.square = square;
      button.tabIndex = square === this.focusSquare ? 0 : -1;
      const column = index % 8;
      const row = Math.floor(index / 8);
      if (column === 0) button.append(coordinate("rank", square[1]));
      if (row === 7) button.append(coordinate("file", square[0]));
      this.buttons.set(square, button);
      this.element.append(button);
    });
    this.render();
  }

  setOrientation(orientation) {
    if (orientation === this.orientation) return;
    this.orientation = orientation;
    this.build();
  }

  flip() {
    this.setOrientation(this.orientation === "white" ? "black" : "white");
  }

  setPosition({ fen, legalMoves = [], mover = null, lastMove = null, checkSquare = null }) {
    this.pieces = parseFen(fen);
    this.legal = legalMoves;
    this.mover = mover;
    this.lastMove = lastMove;
    this.checkSquare = checkSquare;
    if (this.selected && !this.targetsFrom(this.selected).length) this.selected = null;
    this.render();
  }

  setBusy(busy) {
    this.busy = busy;
    this.element.setAttribute("aria-busy", busy ? "true" : "false");
    if (busy) this.selected = null;
    this.render();
  }

  targetsFrom(square) {
    return this.legal.filter((uci) => uci.startsWith(square)).map((uci) => uci.slice(2, 4));
  }

  canMoveFrom(square) {
    const piece = this.pieces.get(square);
    return Boolean(
      this.interactive && !this.busy && piece && piece.color === this.mover && this.targetsFrom(square).length,
    );
  }

  render() {
    const targets = new Set(this.selected ? this.targetsFrom(this.selected) : []);
    const last = this.lastMove ? [this.lastMove.slice(0, 2), this.lastMove.slice(2, 4)] : [];
    for (const [square, button] of this.buttons) {
      const piece = this.pieces.get(square);
      button.querySelector(".piece")?.remove();
      if (piece) {
        const span = document.createElement("span");
        span.className = `piece ${piece.color}`;
        span.setAttribute("aria-hidden", "true");
        span.textContent = glyph(piece);
        button.prepend(span);
      }
      const isTarget = targets.has(square);
      button.classList.toggle("target", isTarget);
      button.classList.toggle("capture", isTarget && Boolean(piece));
      button.classList.toggle("selected", square === this.selected);
      button.classList.toggle("last", last.includes(square));
      button.classList.toggle("check", square === this.checkSquare);
      if (this.interactive) button.setAttribute("aria-pressed", square === this.selected ? "true" : "false");
      const parts = [square, piece ? pieceName(piece) : "empty"];
      if (isTarget) parts.push(piece ? "capture available" : "legal move");
      if (last.includes(square)) parts.push("last move");
      if (square === this.checkSquare) parts.push("in check");
      button.setAttribute("aria-label", parts.join(", "));
      button.disabled = false;
    }
  }

  focusOn(square) {
    const button = this.buttons.get(square);
    if (!button) return;
    for (const other of this.buttons.values()) other.tabIndex = -1;
    button.tabIndex = 0;
    this.focusSquare = square;
    button.focus();
  }

  onKey(event) {
    const button = event.target.closest(".sq");
    if (!button) return;
    if (event.key === "Escape" && this.selected) {
      this.selected = null;
      this.render();
      return;
    }
    const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const delta = moves[event.key];
    if (!delta) return;
    event.preventDefault();
    const order = squaresFor(this.orientation);
    const index = order.indexOf(button.dataset.square);
    const row = Math.min(7, Math.max(0, Math.floor(index / 8) + delta[0]));
    const column = Math.min(7, Math.max(0, (index % 8) + delta[1]));
    this.focusOn(order[row * 8 + column]);
  }

  onClick(event) {
    const button = event.target.closest(".sq");
    if (!button || !this.interactive) return;
    if (this.suppressClick) {
      this.suppressClick = false;
      return;
    }
    this.focusSquare = button.dataset.square;
    this.activate(button.dataset.square);
  }

  activate(square) {
    if (this.busy) return;
    if (this.selected && this.targetsFrom(this.selected).includes(square)) {
      this.tryMove(this.selected, square);
      return;
    }
    this.selected = this.canMoveFrom(square) && this.selected !== square ? square : null;
    this.render();
  }

  async tryMove(from, to) {
    const candidates = this.legal.filter((uci) => uci.slice(0, 4) === from + to);
    if (!candidates.length) return;
    let uci = candidates[0];
    if (candidates.length > 1 || uci.length === 5) {
      const piece = await choosePromotion();
      if (!piece) {
        this.focusOn(to);
        return;
      }
      uci = from + to + piece;
    }
    this.selected = null;
    this.render();
    if (this.onMove) this.onMove(uci);
  }

  onPointerDown(event) {
    const button = event.target.closest(".sq");
    if (!button || event.button !== 0 || !this.canMoveFrom(button.dataset.square)) return;
    this.drag = { from: button.dataset.square, x: event.clientX, y: event.clientY, ghost: null };
  }

  onPointerMove(event) {
    const drag = this.drag;
    if (!drag) return;
    if (!drag.ghost) {
      if (Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < DRAG_THRESHOLD) return;
      // Capture only once a real drag starts, so plain clicks still target their square.
      this.element.setPointerCapture(event.pointerId);
      const piece = this.pieces.get(drag.from);
      const ghost = document.createElement("span");
      ghost.className = `drag-ghost piece ${piece.color}`;
      ghost.textContent = glyph(piece);
      ghost.style.fontSize = `${this.buttons.get(drag.from).clientWidth * 0.82}px`;
      document.body.append(ghost);
      drag.ghost = ghost;
      this.selected = drag.from;
      this.render();
      this.buttons.get(drag.from).classList.add("dragging");
    }
    drag.ghost.style.left = `${event.clientX}px`;
    drag.ghost.style.top = `${event.clientY}px`;
  }

  onPointerUp(event) {
    const drag = this.drag;
    if (!drag) return;
    this.drag = null;
    if (!drag.ghost) return;
    drag.ghost.remove();
    this.buttons.get(drag.from)?.classList.remove("dragging");
    this.suppressClick = true;
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest(".sq");
    const to = target?.dataset.square;
    if (to && this.targetsFrom(drag.from).includes(to)) {
      this.tryMove(drag.from, to);
    } else {
      this.render();
    }
  }

  cancelDrag() {
    if (this.drag?.ghost) this.drag.ghost.remove();
    this.drag = null;
    this.render();
  }
}

function coordinate(kind, text) {
  const span = document.createElement("span");
  span.className = `coord ${kind}`;
  span.setAttribute("aria-hidden", "true");
  span.textContent = text;
  return span;
}
