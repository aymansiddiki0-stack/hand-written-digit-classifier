import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

function mockFetchSequence(responses: Array<{ status: number; body: unknown } | Error>) {
  const fn = vi.fn();
  for (const r of responses) {
    if (r instanceof Error) fn.mockRejectedValueOnce(r);
    else
      fn.mockResolvedValueOnce({
        status: r.status,
        json: async () => r.body,
      } as Response);
  }
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("App shell status panel", () => {
  it("reports alive-but-not-ready when /ready returns 503", async () => {
    mockFetchSequence([
      { status: 200, body: { status: "ok", version: "0.1.0" } },
      { status: 503, body: { ready: false, model_loaded: false, detail: "No model artifact is loaded; predictions are unavailable." } },
    ]);
    render(<App />);
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(/running but not ready/i)
    );
    expect(screen.getByRole("status")).toHaveTextContent(/No model artifact/i);
  });

  it("reports ready when both endpoints succeed", async () => {
    mockFetchSequence([
      { status: 200, body: { status: "ok", version: "0.1.0" } },
      { status: 200, body: { ready: true, model_loaded: true, detail: "Service can serve predictions." } },
    ]);
    render(<App />);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/API ready/i));
    expect(screen.getByRole("status")).toHaveTextContent("0.1.0");
  });

  it("reports unreachable when fetch rejects, without a fake success", async () => {
    mockFetchSequence([new Error("connection refused")]);
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/not responding/i));
    expect(screen.queryByText(/API ready/i)).not.toBeInTheDocument();
  });

  it("keeps the retry control available after failure", async () => {
    mockFetchSequence([new Error("connection refused")]);
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /check again/i })).toBeEnabled();
  });
});
