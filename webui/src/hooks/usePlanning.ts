import { useCallback, useEffect, useRef, useState } from "react";
import { PlanningRpcError } from "@/lib/nanobot-client";
import type {
  ConnectionStatus, PlanningDetail, PlanningHandoff, PlanningIndex, PlanningPreview,
  PlanningRequest, PlanningStatus, WorkspaceScopePayload,
} from "@/lib/types";
import { useClient } from "@/providers/ClientProvider";

class StalePlanningRequestError extends Error {}

export function usePlanning(scope: WorkspaceScopePayload | null, planId: string | null) {
  const { client } = useClient();
  const [connection, setConnection] = useState<ConnectionStatus>(client.status);
  const [service, setService] = useState<PlanningStatus | null>(null);
  const [plans, setPlans] = useState<PlanningIndex[]>([]);
  const [detail, setDetail] = useState<PlanningDetail | null>(null);
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<"full" | "selected">("full");
  const [selected, setSelected] = useState<string[]>([]);
  const [preview, setPreview] = useState<PlanningPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const busyRef = useRef<string | null>(null);
  const detailRef = useRef(detail);
  const draftRef = useRef(draft);
  detailRef.current = detail;
  draftRef.current = draft;
  const epoch = useRef(0);
  const mounted = useRef(true);
  const generation = useRef<{ mission: string; id: string } | null>(null);
  const handoffId = useRef<{ binding: string; id: string } | null>(null);
  const dirty = Boolean(detail?.document && draft !== detail.document.markdown);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; epoch.current += 1; };
  }, []);
  useEffect(() => client.onStatus(setConnection), [client]);
  const request = useCallback(async <T,>(data: Omit<PlanningRequest, "type" | "request_id" | "workspace_scope">): Promise<T> => {
    const current = epoch.current;
    try {
      const result = await client.planning<T>({ ...data, ...(scope ? { workspace_scope: {
        project_path: scope.project_path, access_mode: scope.access_mode,
      } } : {}) });
      if (!mounted.current || current !== epoch.current) throw new StalePlanningRequestError();
      return result;
    } catch (err) {
      if (!mounted.current || current !== epoch.current) throw new StalePlanningRequestError();
      throw err;
    }
  }, [client, scope?.project_path, scope?.access_mode]);

  const report = useCallback((err: unknown) => {
    if (!mounted.current || err instanceof StalePlanningRequestError) return;
    setError(err instanceof PlanningRpcError ? `${err.code}：${err.message}` : err instanceof Error ? err.message : "规划操作失败");
  }, []);
  const apply = useCallback((next: PlanningDetail, force = false) => {
    const previous = detailRef.current;
    const changed = next.document?.revision !== previous?.document?.revision || next.plan.plan_id !== previous?.plan.plan_id;
    const wasDirty = previous?.document && draftRef.current !== previous.document.markdown;
    detailRef.current = next;
    setDetail(next);
    if (force || (changed && !wasDirty)) {
      draftRef.current = next.document?.markdown ?? "";
      setDraft(draftRef.current);
      setSelected([]);
      setPreview(null);
    }
    if (next.service_error) setError(`${next.service_error.code}：${next.service_error.message}`);
  }, []);
  const refreshList = useCallback(async () => {
    const current = epoch.current;
    const result = await request<{ plans: PlanningIndex[] }>({ action: "list" });
    if (mounted.current && current === epoch.current) setPlans(result.plans);
  }, [request]);

  useEffect(() => {
    epoch.current += 1;
    setDetail(null); detailRef.current = null; draftRef.current = "";
    setDraft(""); setPlans([]); setSelected([]); setPreview(null); setError(null); setService(null);
    generation.current = null; handoffId.current = null;
    return () => { epoch.current += 1; };
    // 连接变化由下方恢复逻辑处理，以保留断线时的编辑草稿。
  }, [request, planId]);

  useEffect(() => {
    if (connection !== "open") return;
    const current = epoch.current;
    void request<PlanningStatus>({ action: "status" }).then((data) => {
      if (current === epoch.current) setService(data);
    }).catch((err) => { if (current === epoch.current) report(err); });
    void refreshList().catch(report);
    if (planId) void request<PlanningDetail>({ action: "get", plan_id: planId,
      ...(detailRef.current?.document ? { revision: detailRef.current.document.revision } : {}),
    }).then((data) => {
      if (current === epoch.current) apply(data);
    }).catch((err) => { if (current === epoch.current) report(err); });
  }, [connection, request, planId, apply, refreshList, report]);

  useEffect(() => {
    if (connection !== "open" || !planId || !detail || !["queued", "running"].includes(detail.plan.status)) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const current = epoch.current;
    const poll = async () => {
      try {
        const next = await request<PlanningDetail>({ action: "get", plan_id: planId });
        if (stopped || current !== epoch.current) return;
        apply(next);
        if (["queued", "running"].includes(next.plan.status)) timer = setTimeout(poll, (service?.poll_interval_s ?? 2) * 1000);
        else void refreshList().catch(report);
      } catch (err) {
        if (!stopped && current === epoch.current) report(err);
      }
    };
    timer = setTimeout(poll, (service?.poll_interval_s ?? 2) * 1000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [connection, planId, detail?.plan.status, request, apply, service?.poll_interval_s, refreshList, report]);

  useEffect(() => { setPreview(null); }, [draft, mode, selected]);

  const run = async <T,>(label: string, operation: () => Promise<T>): Promise<T | null> => {
    if (busyRef.current) return null;
    busyRef.current = label;
    const current = epoch.current;
    setBusy(label); setError(null);
    try {
      const result = await operation();
      return mounted.current && current === epoch.current ? result : null;
    } catch (err) { if (current === epoch.current) report(err); return null; }
    finally { busyRef.current = null; if (mounted.current) setBusy(null); }
  };
  const generate = (mission: string) => run("生成", async () => {
    if (generation.current?.mission !== mission) generation.current = { mission, id: crypto.randomUUID() };
    const next = await request<PlanningDetail>({ action: "create", mission, generation_request_id: generation.current.id });
    await refreshList().catch(report);
    return next.plan.plan_id;
  });
  const loadRevision = (revision: number) => run("加载版本", async () => {
    const next = await request<PlanningDetail>({ action: "get", plan_id: planId!, revision });
    apply(next, true);
  });
  const save = () => run("保存", async () => {
    if (!detail?.document) return null;
    const next = await request<PlanningDetail>({ action: "save", plan_id: detail.plan.plan_id,
      base_revision: detail.document.revision, markdown: draft });
    apply(next, true);
    await refreshList().catch(report);
    return next;
  });
  const reload = () => run("查询", async () => {
    setService(await request<PlanningStatus>({ action: "status" }));
    await refreshList();
    if (planId) apply(await request<PlanningDetail>({ action: "get", plan_id: planId,
      ...(detailRef.current?.document ? { revision: detailRef.current.document.revision } : {}),
    }));
  });
  const cancel = () => run("取消", async () => {
    const next = await request<PlanningDetail>({ action: "cancel", plan_id: planId! });
    apply(next); await refreshList();
  });
  const transfer = (document = detail?.document, previewOnly = false) => run(previewOnly ? "预览交接" : "交接", async () => {
    if (!document || (draftRef.current !== document.markdown && !previewOnly)) throw new Error("请先保存修改");
    const binding = { plan_id: document.plan_id, revision: document.revision, content_hash: document.content_hash,
      selected_task_ids: mode === "full" ? null : selected };
    if (previewOnly) {
      const data = await request<PlanningPreview>({ action: "handoff", ...binding, preview: true });
      setPreview(data); return null;
    }
    const key = JSON.stringify(binding);
    if (handoffId.current?.binding !== key) handoffId.current = { binding: key, id: crypto.randomUUID() };
    return request<PlanningHandoff>({ action: "handoff", ...binding, handoff_id: handoffId.current.id });
  });
  return { connection, service, plans, detail, draft, setDraft, mode, setMode, selected, setSelected,
    preview, dirty, busy, error, generate, loadRevision, save, cancel, transfer, reload, refreshList, report,
    hasPending: () => Boolean(busyRef.current),
    isDirty: () => Boolean(detailRef.current?.document && draftRef.current !== detailRef.current.document.markdown),
  };
}
