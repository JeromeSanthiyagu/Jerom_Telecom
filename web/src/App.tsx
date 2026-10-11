import { useEffect } from "react";
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { AppDataProvider, useAppData } from "./lib/store";
import { AnalyzePage } from "./pages/AnalyzePage";
import { AssessPage } from "./pages/AssessPage";
import { RecordPage } from "./pages/RecordPage";
import { SearchPage } from "./pages/SearchPage";
import { EmptyState, Notice } from "./components/ui";

const NAV = [
  { to: "/", label: "Analyze a case", title: "Analyze a case", end: true },
  { to: "/search", label: "Search records", title: "Search records" },
  { to: "/record", label: "Record details", title: "Record details" },
  { to: "/assess", label: "Assess measurements", title: "Assess measurements" },
];

function BackendStatus() {
  const { health, healthError } = useAppData();
  if (healthError) return <span className="status-line"><span className="dot dot--crit" />API unreachable</span>;
  if (!health) return <span className="status-line muted">Checking API…</span>;
  const llm = health.checks.llm;
  const ok = health.status === "ok";
  return (
    <span className="status-line" title={ok ? "All indexes and models loaded" : "Some components failed to load; see /health"}>
      <span className={`dot ${ok ? "dot--ok" : "dot--warn"}`} />
      {ok ? "API connected" : "API degraded"}
      {health.checks.incidents !== undefined && <span className="muted">· {health.checks.incidents.toLocaleString("en-US")} records</span>}
      {llm && <span className="muted">· LLM {llm.provider === "offline" ? "off" : llm.model}</span>}
    </span>
  );
}

function Header() {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <Link to="/" className="brand" aria-label="NetOps Assistant, home">
          <svg className="brand__mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <path d="M12 10v11M8.5 21h7M9 13.5l3-3.5 3 3.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M7.05 5.05a7 7 0 0 0 0 9.9M16.95 5.05a7 7 0 0 1 0 9.9M9.17 7.17a4 4 0 0 0 0 5.66M14.83 7.17a4 4 0 0 1 0 5.66" strokeLinecap="round" />
          </svg>
          NetOps Assistant
        </Link>
        <nav className="nav" aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}>{n.label}</NavLink>
          ))}
        </nav>
        <div className="header-status"><BackendStatus /></div>
      </div>
    </header>
  );
}

function TitleSync() {
  const { pathname } = useLocation();
  useEffect(() => {
    const page = NAV.find((n) => (n.end ? pathname === n.to : pathname.startsWith(n.to)));
    document.title = page ? `${page.title} · NetOps Assistant` : "NetOps Assistant";
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

function Shell() {
  const { referenceError, retry } = useAppData();
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <Header />
      <TitleSync />
      <main id="main" tabIndex={-1}>
        {referenceError && (
          <div style={{ marginBottom: "var(--space-5)" }}>
            <Notice tone="crit" role="alert" title="Could not load reference data from the API">
              Field labels and thresholds come from the backend. Check that the API server is running.{" "}
              <button type="button" className="btn btn--sm" onClick={retry}>Try again</button>
            </Notice>
          </div>
        )}
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/record" element={<RecordPage />} />
          <Route path="/record/:id" element={<RecordPage />} />
          <Route path="/assess" element={<AssessPage />} />
          <Route path="*" element={
            <EmptyState title="Page not found">
              This address does not match a page. Use the navigation above to continue.
            </EmptyState>
          } />
        </Routes>
      </main>
    </>
  );
}

export function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "")}>
      <AppDataProvider>
        <Shell />
      </AppDataProvider>
    </BrowserRouter>
  );
}
