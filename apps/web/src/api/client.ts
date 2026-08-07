// Typed API-client boundary.
// These types mirror services/api/app.py response models. A change on either
// side must be coordinated and covered by contract tests.

export interface HealthResponse {
  status: "ok";
  version: string;
}

export interface ReadyResponse {
  ready: boolean;
  model_loaded: boolean;
  detail: string;
}

export type ApiStatus =
  | { kind: "ready"; version: string }
  | { kind: "alive_not_ready"; detail: string }
  | { kind: "unreachable"; detail: string };

const BASE = "/api";

async function getJson<T>(path: string): Promise<{ status: number; body: T }> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  const body = (await resp.json()) as T;
  return { status: resp.status, body };
}

/** Probe the local API and classify its state honestly. */
export async function fetchApiStatus(): Promise<ApiStatus> {
  let health: HealthResponse;
  try {
    const resp = await getJson<HealthResponse>("/health");
    if (resp.status !== 200) {
      return { kind: "unreachable", detail: `Health check returned ${resp.status}.` };
    }
    health = resp.body;
  } catch {
    return {
      kind: "unreachable",
      detail: "The local API is not responding. Start it, then try again.",
    };
  }

  try {
    const resp = await getJson<ReadyResponse>("/ready");
    if (resp.status === 200 && resp.body.ready) {
      return { kind: "ready", version: health.version };
    }
    return { kind: "alive_not_ready", detail: resp.body.detail };
  } catch {
    return { kind: "alive_not_ready", detail: "Readiness could not be determined." };
  }
}
