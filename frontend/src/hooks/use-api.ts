"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";

type PollOptions<T> = {
  pollIntervalMs?: number;
  pollWhile?: (data: T) => boolean;
};

export function useApi<T>(path: string | null, { pollIntervalMs = 0, pollWhile }: PollOptions<T> = {}) {
  const [result, setResult] = useState<{ path: string; version: number; data?: T; error?: string }>();
  const [version, setVersion] = useState(0);
  const refresh = useCallback(() => setVersion((value) => value + 1), []);

  useEffect(() => {
    if (path === null) return;
    const requestPath = path;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function load() {
      let again = true;
      try {
        const data = await api<T>(requestPath, controller.signal);
        if (controller.signal.aborted) return;
        setResult({ path: requestPath, version, data });
        again = pollWhile ? pollWhile(data) : false;
      } catch (error) {
        if (controller.signal.aborted) return;
        setResult((previous) => ({
          path: requestPath, version,
          data: previous?.path === requestPath ? previous.data : undefined,
          error: error instanceof Error ? error.message : "Request failed.",
        }));
      }
      if (!controller.signal.aborted && pollIntervalMs > 0 && again) {
        timer = setTimeout(load, pollIntervalMs);
      }
    }

    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [path, version, pollIntervalMs, pollWhile]);

  const current = result?.path === path && result.version === version ? result : undefined;
  return { data: current?.data, error: current?.error, loading: path !== null && !current, refresh };
}
