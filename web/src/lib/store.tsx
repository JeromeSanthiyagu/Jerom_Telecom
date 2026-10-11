import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "./api";
import type { ConditionDef, Field, Health, Reference } from "./types";

// ---------------------------------------------------------------------------- session-persistent state
// Form inputs and the latest results survive page switches (in memory) and reloads (sessionStorage).
const memory = new Map<string, unknown>();
const listeners = new Map<string, Set<() => void>>();

function read<T>(key: string, initial: T): T {
  if (memory.has(key)) return memory.get(key) as T;
  try {
    const raw = sessionStorage.getItem(`netops:${key}`);
    if (raw !== null) {
      const value = JSON.parse(raw) as T;
      memory.set(key, value);
      return value;
    }
  } catch {
    /* storage unavailable: fall back to memory only */
  }
  return initial;
}

export function writeSession<T>(key: string, value: T): void {
  memory.set(key, value);
  try {
    sessionStorage.setItem(`netops:${key}`, JSON.stringify(value));
  } catch {
    /* quota or privacy mode: memory copy still works */
  }
  listeners.get(key)?.forEach((fn) => fn());
}

export function useSessionState<T>(key: string, initial: T): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => read(key, initial));
  useEffect(() => {
    const fn = () => setValue(read(key, initial));
    if (!listeners.has(key)) listeners.set(key, new Set());
    listeners.get(key)!.add(fn);
    return () => void listeners.get(key)!.delete(fn);
  }, [key]);
  const set = useCallback(
    (v: T | ((prev: T) => T)) => {
      const next = typeof v === "function" ? (v as (p: T) => T)(read(key, initial)) : v;
      writeSession(key, next);
    },
    [key],
  );
  return [value, set];
}

// ---------------------------------------------------------------------------- reference data + health
interface AppData {
  reference: Reference | null;
  referenceError: string | null;
  health: Health | null;
  healthError: boolean;
  field: (name: string) => Field | undefined;
  condition: (name: string) => ConditionDef | undefined;
  conditionLabel: (name: string) => string;
  conditionsFor: (metric: string) => ConditionDef[];
  stateName: (code: string) => string;
  retry: () => void;
}

const AppDataContext = createContext<AppData | null>(null);

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [reference, setReference] = useState<Reference | null>(null);
  const [referenceError, setReferenceError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let live = true;
    api.reference()
      .then((r) => live && (setReference(r), setReferenceError(null)))
      .catch((e: Error) => live && setReferenceError(e.message));
    const poll = () =>
      api.health()
        .then((h) => live && (setHealth(h), setHealthError(false)))
        .catch(() => live && setHealthError(true));
    poll();
    const timer = window.setInterval(poll, 30_000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [attempt]);

  const value = useMemo<AppData>(() => {
    const fields = new Map(reference?.fields.map((f) => [f.name, f]));
    const conds = new Map(reference?.conditions.map((c) => [c.name, c]));
    return {
      reference, referenceError, health, healthError,
      field: (n) => fields.get(n),
      condition: (n) => conds.get(n),
      conditionLabel: (n) => conds.get(n)?.label ?? n.replace(/_/g, " "),
      conditionsFor: (m) => reference?.conditions.filter((c) => c.metric === m) ?? [],
      stateName: (code) => (reference?.states[code] ? `${reference.states[code]} (${code})` : code),
      retry: () => setAttempt((a) => a + 1),
    };
  }, [reference, referenceError, health, healthError]);

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}

export function useAppData(): AppData {
  const ctx = useContext(AppDataContext);
  if (!ctx) throw new Error("useAppData must be used inside AppDataProvider");
  return ctx;
}
