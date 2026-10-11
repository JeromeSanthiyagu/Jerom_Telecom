import type {
  AnalyzeResponse, ApiError, ClassifyResponse, Health, NetworkRecord, Reference, RiskResponse, SearchResponse,
} from "./types";

/** An error the backend reported (structured body) or a network failure. */
export class RequestError extends Error {
  constructor(public readonly error: ApiError, public readonly status: number) {
    super(error.message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new RequestError({ code: "connection_error", message: "Cannot reach the API server." }, 0);
  }
  const isJson = res.headers.get("content-type")?.startsWith("application/json");
  const data = isJson ? await res.json() : null;
  if (!res.ok) {
    const err: ApiError = data?.error ?? { code: `http_${res.status}`, message: `Request failed (HTTP ${res.status}).` };
    throw new RequestError(err, res.status);
  }
  return data as T;
}

export type Filters = Record<string, string | boolean | string[]>;
export type Metrics = Record<string, number | string | boolean>;

export const api = {
  health: () => request<Health>("GET", "/health"),
  reference: () => request<Reference>("GET", "/api/v1/reference"),
  analyze: (b: { query: string; filters?: Filters | null; top_k: number; use_llm: boolean; metrics?: Metrics | null },
    signal?: AbortSignal) => request<AnalyzeResponse>("POST", "/api/v1/analyze", b, signal),
  search: (b: { query: string; filters?: Filters | null; top_k: number }, signal?: AbortSignal) =>
    request<SearchResponse>("POST", "/api/v1/search", b, signal),
  record: (id: string) => request<NetworkRecord & { note: string }>("GET", `/api/v1/records/${encodeURIComponent(id)}`),
  classify: (b: { record_id?: string; metrics?: Metrics }) => request<ClassifyResponse>("POST", "/api/v1/classify", b),
  risk: (b: { record_id?: string; metrics?: Metrics }) => request<RiskResponse>("POST", "/api/v1/risk", b),
  feedback: (b: Record<string, unknown>) => request<Record<string, unknown>>("POST", "/api/v1/feedback", b),
};
