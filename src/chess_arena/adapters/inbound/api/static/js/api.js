// Thin fetch wrappers for the Chess Arena JSON API.

async function request(method, path, body) {
  const options = { method, headers: { Accept: "application/json" } };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = payload && payload.detail;
    const message = typeof detail === "string" ? detail : JSON.stringify(detail ?? response.statusText);
    throw new Error(message);
  }
  return payload;
}

export const api = {
  health: () => request("GET", "/api/health"),
  opponents: () => request("GET", "/api/opponents"),
  ladders: () => request("GET", "/api/ladders"),
  startGame: (opponent, humanColor) =>
    request("POST", "/api/games", { opponent, human_color: humanColor }),
  game: (id) => request("GET", `/api/games/${encodeURIComponent(id)}`),
  move: (id, uci) => request("POST", `/api/games/${encodeURIComponent(id)}/moves`, { uci }),
  resign: (id) => request("POST", `/api/games/${encodeURIComponent(id)}/resign`),
  replay: (id) => request("GET", `/api/games/${encodeURIComponent(id)}/replay`),
  games: (kind, limit = 100) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (kind) params.set("kind", kind);
    return request("GET", `/api/games?${params}`);
  },
  models: () => request("GET", "/api/models"),
  model: (id) => request("GET", `/api/models/${encodeURIComponent(id)}`),
  ladderRuns: () => request("GET", "/api/ladder-runs"),
};

export function playerKey(player) {
  return `${player.kind}:${player.player_id}`;
}

export function parsePlayerKey(key) {
  const index = key.indexOf(":");
  return { kind: key.slice(0, index), player_id: key.slice(index + 1) };
}
