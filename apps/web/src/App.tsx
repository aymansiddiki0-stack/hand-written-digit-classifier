import { useCallback, useEffect, useState } from "react";
import { fetchApiStatus, type ApiStatus } from "./api/client";

type ShellState = { phase: "checking" } | { phase: "resolved"; status: ApiStatus };

export default function App() {
  const [state, setState] = useState<ShellState>({ phase: "checking" });

  const check = useCallback(async () => {
    setState({ phase: "checking" });
    const status = await fetchApiStatus();
    setState({ phase: "resolved", status });
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  return (
    <main className="shell">
      <header>
        <h1>Digit Classifier</h1>
        <p className="tagline">
          The app reports whether the local API is reachable and ready before prediction work
          begins.
        </p>
      </header>

      <section aria-live="polite" aria-label="System status" className="status">
        {state.phase === "checking" && <p role="status">Checking the local API…</p>}
        {state.phase === "resolved" && <StatusPanel status={state.status} />}
        <button type="button" onClick={() => void check()} disabled={state.phase === "checking"}>
          Check again
        </button>
      </section>
    </main>
  );
}

function StatusPanel({ status }: { status: ApiStatus }) {
  switch (status.kind) {
    case "ready":
      return (
        <p role="status" data-state="ready">
          API ready (version {status.version}). A model is loaded.
        </p>
      );
    case "alive_not_ready":
      return (
        <p role="status" data-state="not-ready">
          API is running but not ready: {status.detail}
        </p>
      );
    case "unreachable":
      return (
        <p role="alert" data-state="unreachable">
          {status.detail}
        </p>
      );
  }
}
