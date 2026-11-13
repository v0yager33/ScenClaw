import { useCallback, useEffect, useMemo, useState } from "react";
import { ClipboardList, PanelLeft, Send, Save } from "lucide-react";
import { MarkdownText } from "@/components/MarkdownText";
import { WorkspaceProjectPicker } from "@/components/thread/WorkspaceControls";
import { Button } from "@/components/ui/button";
import { usePlanning } from "@/hooks/usePlanning";
import type { PlanningHandoff, WorkspaceScopePayload, WorkspacesPayload } from "@/lib/types";

const STATUS: Record<string, string> = {
  queued: "排队中", running: "生成中", completed: "已完成", failed: "失败",
  cancelled: "已取消", timed_out: "已超时", interrupted: "已中断，未自动重跑",
};

export function PlanningView({ planId, scope, workspaces, onSelectPlan, onHandoff, onScopeChange,
  onToggleSidebar, onLeaveGuardChange }: {
  planId: string | null; scope: WorkspaceScopePayload | null; workspaces: WorkspacesPayload | null;
  onSelectPlan: (id: string | null) => void; onHandoff: (chatId: string, scope: WorkspaceScopePayload | null) => Promise<void>;
  onScopeChange: (scope: WorkspaceScopePayload) => void; onToggleSidebar: () => void;
  onLeaveGuardChange: (guard: (() => boolean) | null) => void;
}) {
  const p = usePlanning(scope, planId);
  const [mission, setMission] = useState("");
  const [tab, setTab] = useState<"preview" | "edit" | "select">("preview");
  const [interrupted, setInterrupted] = useState<PlanningHandoff | null>(null);
  const document = p.detail?.document;
  const canLeave = useCallback(() => !p.hasPending() && (!p.isDirty() || window.confirm("有未保存的修改，是否放弃草稿并离开？")), [p.hasPending, p.isDirty]);
  useEffect(() => {
    onLeaveGuardChange(canLeave);
    return () => onLeaveGuardChange(null);
  }, [canLeave, onLeaveGuardChange]);
  useEffect(() => {
    if (!p.dirty) return;
    const beforeUnload = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [p.dirty]);
  useEffect(() => { setInterrupted(null); }, [planId]);
  const groups = useMemo(() => {
    const result = new Map<string, { name: string; tasks: NonNullable<typeof document>["tasks"] }>();
    for (const task of document?.tasks ?? []) {
      if (!result.has(task.phase_id)) result.set(task.phase_id, { name: task.phase_name, tasks: [] });
      result.get(task.phase_id)!.tasks.push(task);
    }
    return Array.from(result.entries());
  }, [document]);
  const unavailable = p.connection !== "open" || !p.service?.available;
  const count = p.mode === "full" ? document?.tasks.length ?? 0 : p.selected.length;
  const blocked = Boolean(p.busy) || unavailable || !document || count === 0;
  const openResult = async (result: PlanningHandoff | null) => {
    if (!result) return;
    if (result.status !== "dispatched") { setInterrupted(result); return; }
    try { await onHandoff(result.chat_id, scope); }
    catch (err) { p.report(err); setInterrupted(result); }
  };
  const saveContinue = async () => {
    const saved = await p.save();
    if (!saved?.document) return;
    if (p.mode === "selected") { setTab("select"); return; }
    await openResult(await p.transfer(saved.document));
  };

  return (
    <section className="flex h-full min-w-0 flex-col" aria-label="任务规划">
      <header className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-3">
        <Button variant="ghost" size="icon" className="lg:hidden" aria-label="打开侧栏" onClick={onToggleSidebar}><PanelLeft className="h-4 w-4" /></Button>
        <ClipboardList className="h-5 w-5 text-primary" /><h1 className="text-lg font-semibold">任务规划</h1>
        <span className="ml-auto text-xs text-muted-foreground">MissionOrch → 想定会话</span>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto p-4 md:p-6">
        <div className="mx-auto grid max-w-7xl gap-5 lg:grid-cols-[240px_minmax(0,1fr)]">
          <aside className="space-y-3 rounded-xl border border-border p-3">
            <div className="flex items-center justify-between"><h2 className="font-medium">当前工作区</h2>
              <Button size="sm" variant="ghost" disabled={p.connection !== "open" || !!p.busy} onClick={() => void p.reload()}>刷新</Button></div>
            <p className="break-all text-xs text-muted-foreground">{scope?.project_path ?? "默认工作区"}</p>
            <WorkspaceProjectPicker isHero scope={scope} defaultScope={workspaces?.default_scope ?? null} controls={workspaces?.controls ?? null}
              disabled={!!p.busy} onChange={onScopeChange} />
            <Button className="w-full" variant="outline" disabled={!!p.busy} onClick={() => onSelectPlan(null)}>新建规划</Button>
            <h2 className="pt-2 text-sm font-medium">已保存规划</h2>
            {p.plans.length === 0 && <p className="text-xs text-muted-foreground">暂无规划</p>}
            <div className="max-h-80 space-y-2 overflow-y-auto lg:max-h-none">
              {p.plans.map((plan) => <button key={plan.plan_id} disabled={!!p.busy} onClick={() => onSelectPlan(plan.plan_id)}
                aria-current={planId === plan.plan_id ? "page" : undefined}
                className={`w-full rounded-lg border p-2 text-left text-sm ${planId === plan.plan_id ? "border-primary bg-accent" : "border-border hover:bg-accent"}`}>
                <span className="line-clamp-2 break-words">{plan.mission}</span><span className="text-xs text-muted-foreground">{STATUS[plan.status]} · v{plan.revision}</span>
              </button>)}
            </div>
          </aside>
          <div className="min-w-0 space-y-4">
            {p.connection !== "open" && <p role="alert" className="rounded-lg border p-3 text-sm">连接已断开；重连后将查询原作业，不自动重发生成。草稿仍保留。</p>}
            {p.service?.error && <p role="alert" className="rounded-lg border border-destructive/40 p-3 text-sm text-destructive">{p.service.error.code}：{p.service.error.message}。已保存的规划仍可查看。</p>}
            {p.error && <p role="alert" className="text-sm text-destructive">{p.error}</p>}
            {!planId && <form className="space-y-3 rounded-xl border border-border p-4" onSubmit={async (event) => {
              event.preventDefault(); const id = await p.generate(mission); if (id) onSelectPlan(id);
            }}>
              <label htmlFor="planning-mission" className="font-medium">用自然语言描述任务</label>
              <textarea id="planning-mission" className="min-h-36 w-full rounded-md border border-input bg-background p-3 text-sm" value={mission}
                disabled={!!p.busy} maxLength={65536} onChange={(event) => setMission(event.target.value)} placeholder="描述目标、参与实体、地理范围、阶段及约束…" />
              <Button disabled={unavailable || !!p.busy || !mission.trim()} type="submit">{p.busy ?? "生成任务规划"}</Button>
              <p className="text-xs text-muted-foreground">沿用 MissionOrch 现有模型与编排配置。生成作业在服务端继续运行。</p>
            </form>}
            {p.detail && <div className="space-y-3 rounded-xl border border-border p-4">
              <div className="flex flex-wrap items-center gap-3"><h2 className="min-w-0 flex-1 break-words font-medium">{p.detail.plan.mission}</h2>
                <span className="text-sm">{STATUS[p.detail.plan.status]}</span>
                {["queued", "running"].includes(p.detail.plan.status) && <Button size="sm" variant="outline" disabled={!!p.busy || unavailable} onClick={() => void p.cancel()}>取消生成</Button>}
              </div>
              {p.detail.plan.error && <p role="alert" className="text-sm text-destructive">{p.detail.plan.error.message}</p>}
              {!!p.detail.plan.logs?.length && <details open={!document}><summary className="cursor-pointer text-sm">阶段日志</summary>
                <ul className="max-h-40 overflow-y-auto text-xs text-muted-foreground">{p.detail.plan.logs.map((log, i) => <li key={i}>{log.ts} · {log.stage} · {log.type}</li>)}</ul></details>}
              {document && <>
                <div className="flex flex-wrap items-center gap-2">
                  <label className="text-sm">版本 <select aria-label="规划版本" className="rounded border bg-background px-2 py-1" value={document.revision} disabled={!!p.busy}
                    onChange={(event) => { if (canLeave()) void p.loadRevision(Number(event.target.value)); }}>
                    {Array.from({ length: p.detail.plan.revision }, (_, i) => <option key={i + 1} value={i + 1}>v{i + 1}</option>)}
                  </select></label>
                  {p.dirty && <span className="text-xs text-amber-600">未保存修改</span>}
                  <Button size="sm" variant="outline" disabled={!!p.busy || unavailable || !p.dirty} onClick={() => void p.save()}><Save className="mr-1 h-3 w-3" />保存新版本</Button>
                  <span className="ml-auto text-xs text-muted-foreground">{document.generation ? `来源生成评分：${document.generation.final_score ?? "—"}` : "人工编辑版；来源评分不作为本版评分"}</span>
                </div>
                <div className="flex gap-2 border-b pb-2" role="tablist" aria-label="规划内容">
                  {([["preview", "预览"], ["edit", "编辑 Markdown"], ["select", "任务选择"]] as const).map(([key, label]) =>
                    <Button size="sm" key={key} role="tab" aria-selected={tab === key} variant={tab === key ? "default" : "ghost"} onClick={() => setTab(key)}>{label}</Button>)}
                </div>
                {tab === "preview" && <div className="overflow-x-auto rounded-lg border p-4"><MarkdownText>{p.draft}</MarkdownText></div>}
                {tab === "edit" && <textarea aria-label="完整规划 Markdown" spellCheck={false} disabled={!!p.busy}
                  className="min-h-[28rem] w-full rounded-lg border bg-background p-3 font-mono text-sm" value={p.draft} onChange={(event) => p.setDraft(event.target.value)} />}
                {tab === "select" && <div className="space-y-3">
                  <fieldset className="flex flex-wrap gap-4"><legend className="sr-only">执行范围</legend>
                    <label><input type="radio" name="planning-mode" checked={p.mode === "full"} disabled={!!p.busy} onChange={() => p.setMode("full")} /> 整份规划</label>
                    <label><input type="radio" name="planning-mode" checked={p.mode === "selected"} disabled={!!p.busy} onChange={() => p.setMode("selected")} /> 仅所选任务</label>
                  </fieldset>
                  <p className="text-xs text-muted-foreground">已选 {count} / {document.tasks.length} 条行动。阶段勾选联动全部行动；依赖不会自动扩展。</p>
                  {p.dirty && <p className="text-sm text-amber-600">请先保存并重新选择任务，避免索引错位。</p>}
                  {groups.map(([phaseId, group]) => <fieldset key={phaseId} className="space-y-2 rounded-lg border p-3" disabled={p.mode === "full" || p.dirty || !!p.busy}>
                    <label className="flex gap-2 font-medium"><input type="checkbox" aria-label={`选择阶段 ${group.name}`} checked={group.tasks.every((task) => p.selected.includes(task.id))}
                      onChange={(event) => p.setSelected((current) => event.target.checked ? Array.from(new Set([...current, ...group.tasks.map((t) => t.id)])) : current.filter((id) => !group.tasks.some((t) => t.id === id)))} />{group.name}</label>
                    {group.tasks.map((task) => <label key={task.id} className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={p.selected.includes(task.id)}
                      onChange={(event) => p.setSelected((current) => event.target.checked ? [...current, task.id] : current.filter((id) => id !== task.id))} /><span>{task.unit_name}：{task.text}</span></label>)}
                  </fieldset>)}
                </div>}
                <details><summary className="cursor-pointer text-sm">结构检查报告（不等于仿真通过）</summary><pre className="overflow-x-auto whitespace-pre-wrap text-xs">{JSON.stringify(document.validation, null, 2)}</pre></details>
                <div className="flex flex-wrap gap-2">
                  <Button variant="outline" disabled={blocked || p.dirty} onClick={() => void p.transfer(document, true)}>交接预览</Button>
                  <Button disabled={blocked || p.dirty} onClick={async () => openResult(await p.transfer())}><Send className="mr-2 h-4 w-4" />生成想定并进入聊天</Button>
                  {p.dirty && <Button variant="outline" disabled={!!p.busy || unavailable} onClick={() => void saveContinue()}>保存并继续</Button>}
                </div>
                {p.preview && <details open><summary className="text-sm">选中快照预览</summary><p className="my-2 text-xs text-muted-foreground">{p.preview.selection_note}</p>
                  <div className="overflow-x-auto rounded border p-3"><MarkdownText>{p.preview.snapshot_markdown}</MarkdownText></div></details>}
              </>}
            </div>}
            {interrupted && <div role="alert" className="rounded-lg border p-3 text-sm">交接状态未确认或聊天跳转失败，未自动重放。请打开已有会话确认是否已经生成。
              <Button className="ml-2" size="sm" onClick={() => void onHandoff(interrupted.chat_id, scope).catch(p.report)}>打开已有会话确认</Button></div>}
          </div>
        </div>
      </div>
    </section>
  );
}
