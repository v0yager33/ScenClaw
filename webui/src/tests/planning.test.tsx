import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PlanningView } from "@/components/planning/PlanningView";
import { usePlanning } from "@/hooks/usePlanning";
import { useSessions } from "@/hooks/useSessions";
import { listSessions } from "@/lib/api";
import { PlanningRpcError } from "@/lib/nanobot-client";
import type {
  ConnectionStatus, PlanningDetail, PlanningHandoff, PlanningPreview, PlanningRequest,
  WorkspaceScopePayload,
} from "@/lib/types";

type Request = Omit<PlanningRequest, "type" | "request_id">;
const rpc = vi.fn<(request: Request) => Promise<unknown>>();
const statusHandlers = new Set<(status: ConnectionStatus) => void>();
const scope: WorkspaceScopePayload = { project_path: "/tmp/planning-workspace", access_mode: "restricted" };
const planId = "00000000-0000-4000-8000-000000000001";
const markdown = "# 完整规划\n\n## 实体清单\n侦察单元\n\n## 地理参考\n区域 G1\n\n## 附加约束\n保留未知章节";
const client = {
  status: "open" as ConnectionStatus,
  planning: <T,>(request: Request) => rpc(request) as Promise<T>,
  onStatus: (handler: (status: ConnectionStatus) => void) => {
    statusHandlers.add(handler);
    return () => { statusHandlers.delete(handler); };
  },
  onSessionUpdate: () => () => {},
  newChat: vi.fn(),
};

vi.mock("@/providers/ClientProvider", () => ({ useClient: () => ({ client, token: "offline" }) }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/api")>(),
  listSessions: vi.fn(),
}));
vi.mock("@/components/MarkdownText", () => ({ MarkdownText: ({ children }: { children: string }) => <pre>{children}</pre> }));
vi.mock("@/components/thread/WorkspaceControls", () => ({ WorkspaceProjectPicker: () => null }));

function detail(revision = 1, text = markdown): PlanningDetail {
  return {
    plan: { plan_id: planId, mission: "离线规划任务", revision, status: "completed", created_at: "offline" },
    document: {
      plan_id: planId, revision, markdown: text, content_hash: `hash-${revision}`,
      coa: {}, validation: { structural_only: true },
      tasks: [
        { id: "P1:U1:0", phase_id: "P1", phase_name: "侦察阶段", unit_id: "U1", unit_name: "侦察单元", action_index: 0, text: "侦察区域 G1" },
        { id: "P1:U1:1", phase_id: "P1", phase_name: "侦察阶段", unit_id: "U1", unit_name: "侦察单元", action_index: 1, text: "报告目标" },
        { id: "P2:U1:0", phase_id: "P2", phase_name: "撤离阶段", unit_id: "U1", unit_name: "侦察单元", action_index: 0, text: "安全撤离" },
      ],
      ...(revision === 1 ? { generation: { final_score: 0.8 } } : { source_revision: 1 }),
    },
  };
}

function handoff(revision = 1): PlanningHandoff {
  return {
    plan_id: planId, revision, chat_id: "scenario-chat", handoff_id: "offline-handoff",
    status: "dispatched", coa_path: "scenarios/offline-handoff/coa.md", output_dir: "scenarios/offline-handoff",
  };
}

function respond(request: Request): Promise<unknown> {
  if (request.action === "status") return Promise.resolve({ enabled: true, available: true, poll_interval_s: 2 });
  if (request.action === "list") return Promise.resolve({ plans: [detail().plan] });
  if (request.action === "get" || request.action === "create") return Promise.resolve(detail(request.revision));
  if (request.action === "save") return Promise.resolve(detail((request.base_revision ?? 1) + 1, request.markdown));
  if (request.action === "handoff" && request.preview) {
    const preview: PlanningPreview = {
      snapshot_markdown: "固定选中快照：实体清单、地理参考、所选行动", snapshot_hash: "snapshot-hash",
      selected_task_ids: request.selected_task_ids ?? detail().document!.tasks.map((task) => task.id),
      selection_note: "未选择的依赖请在原聊天澄清",
    };
    return Promise.resolve(preview);
  }
  if (request.action === "handoff") return Promise.resolve(handoff(request.revision));
  return Promise.resolve({ ...detail(), plan: { ...detail().plan, status: "cancelled" } });
}

function emitStatus(status: ConnectionStatus) {
  act(() => {
    client.status = status;
    for (const handler of statusHandlers) handler(status);
  });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function renderPage(id: string | null = planId) {
  const props = {
    planId: id, scope, workspaces: null, onSelectPlan: vi.fn(), onHandoff: vi.fn().mockResolvedValue(undefined),
    onScopeChange: vi.fn(), onToggleSidebar: vi.fn(), onLeaveGuardChange: vi.fn(),
  };
  return { ...render(<PlanningView {...props} />), props };
}

beforeEach(() => {
  client.status = "open";
  statusHandlers.clear();
  rpc.mockReset().mockImplementation(respond);
  client.newChat.mockReset();
  vi.mocked(listSessions).mockReset().mockResolvedValue([]);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("规划页面与固定版本选择", () => {
  it("自然语言生成后只导航到服务返回的规划", async () => {
    const { props } = renderPage(null);
    const submit = await screen.findByRole("button", { name: "生成任务规划" });
    fireEvent.change(screen.getByLabelText("用自然语言描述任务"), { target: { value: "侦察 G1 后撤离" } });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.click(submit);
    await waitFor(() => expect(props.onSelectPlan).toHaveBeenCalledWith(planId));
    expect(rpc).toHaveBeenCalledWith(expect.objectContaining({ action: "create", mission: "侦察 G1 后撤离", generation_request_id: expect.any(String), workspace_scope: scope }));
  });

  it("阶段联动与逐行动取消产生绑定版本的预览和交接", async () => {
    const { props } = renderPage();
    await screen.findByText(markdown);
    fireEvent.click(screen.getByRole("tab", { name: "任务选择" }));
    fireEvent.click(screen.getByRole("radio", { name: "仅所选任务" }));
    expect(screen.getByRole("button", { name: "生成想定并进入聊天" })).toBeDisabled();
    fireEvent.click(screen.getByLabelText("选择阶段 侦察阶段"));
    expect(screen.getByText(/已选 2 \/ 3/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "侦察单元：报告目标" }));
    expect(screen.getByText(/已选 1 \/ 3/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "交接预览" }));
    await screen.findByText(/固定选中快照/);
    expect(rpc).toHaveBeenCalledWith(expect.objectContaining({ action: "handoff", preview: true, revision: 1, content_hash: "hash-1", selected_task_ids: ["P1:U1:0"] }));
    await waitFor(() => expect(screen.getByRole("button", { name: "生成想定并进入聊天" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "生成想定并进入聊天" }));
    await waitFor(() => expect(props.onHandoff).toHaveBeenCalledWith("scenario-chat", scope));
    expect(rpc).toHaveBeenCalledWith(expect.objectContaining({ action: "handoff", revision: 1, selected_task_ids: ["P1:U1:0"], handoff_id: expect.any(String) }));
  });

  it("保存并继续交接整份新版本，保留完整 Markdown 并移除旧评分", async () => {
    const { props } = renderPage();
    await screen.findByText(markdown);
    fireEvent.click(screen.getByRole("tab", { name: "编辑 Markdown" }));
    const edited = `${markdown}\n人工补充实体与地理约束`;
    fireEvent.change(screen.getByRole("textbox", { name: "完整规划 Markdown" }), { target: { value: edited } });
    expect(screen.getByRole("button", { name: "生成想定并进入聊天" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "交接预览" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "保存并继续" }));
    await waitFor(() => expect(props.onHandoff).toHaveBeenCalledWith("scenario-chat", scope));
    expect(rpc).toHaveBeenCalledWith(expect.objectContaining({ action: "save", base_revision: 1, markdown: edited }));
    expect(rpc).toHaveBeenCalledWith(expect.objectContaining({ action: "handoff", revision: 2, content_hash: "hash-2", selected_task_ids: null }));
    expect(screen.getByRole("combobox", { name: "规划版本" })).toHaveValue("2");
    expect(screen.getByText(/人工编辑版；来源评分不作为本版评分/)).toBeInTheDocument();
  });

  it("部分模式保存后清空旧行动索引，必须重新选择", async () => {
    renderPage();
    await screen.findByText(markdown);
    fireEvent.click(screen.getByRole("tab", { name: "任务选择" }));
    fireEvent.click(screen.getByRole("radio", { name: "仅所选任务" }));
    fireEvent.click(screen.getByLabelText("选择阶段 侦察阶段"));
    fireEvent.click(screen.getByRole("tab", { name: "编辑 Markdown" }));
    fireEvent.change(screen.getByRole("textbox", { name: "完整规划 Markdown" }), { target: { value: `${markdown}\n修改行动` } });
    fireEvent.click(screen.getByRole("button", { name: "保存并继续" }));
    await screen.findByText(/已选 0 \/ 3/);
    expect(screen.getByRole("button", { name: "生成想定并进入聊天" })).toBeDisabled();
    expect(rpc.mock.calls.some(([request]) => request.action === "handoff")).toBe(false);
  });

  it("冲突保留草稿；取消离开与版本切换都不丢失编辑", async () => {
    rpc.mockImplementation((request) => request.action === "save"
      ? Promise.reject(new PlanningRpcError("version_conflict", "已有更新版本")) : respond(request));
    const { props } = renderPage();
    await screen.findByText(markdown);
    fireEvent.click(screen.getByRole("tab", { name: "编辑 Markdown" }));
    const input = screen.getByRole("textbox", { name: "完整规划 Markdown" });
    fireEvent.change(input, { target: { value: `${markdown}\n保留用户草稿` } });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const guard = props.onLeaveGuardChange.mock.calls.at(-1)?.[0] as () => boolean;
    expect(guard()).toBe(false);
    fireEvent.change(screen.getByRole("combobox", { name: "规划版本" }), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "保存新版本" }));
    await screen.findByText(/version_conflict/);
    expect(input).toHaveValue(`${markdown}\n保留用户草稿`);
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(rpc.mock.calls.filter(([request]) => request.action === "get")).toHaveLength(1);
  });

  it("中断来源可打开已有会话，跳转失败不自动重放", async () => {
    rpc.mockImplementation((request) => request.action === "handoff" ? Promise.resolve({ ...handoff(), status: "interrupted" }) : respond(request));
    const { props } = renderPage();
    await screen.findByText(markdown);
    fireEvent.click(screen.getByRole("button", { name: "生成想定并进入聊天" }));
    await screen.findByRole("button", { name: "打开已有会话确认" });
    expect(props.onHandoff).not.toHaveBeenCalled();
    props.onHandoff.mockRejectedValueOnce(new Error("聊天暂不可达"));
    fireEvent.click(screen.getByRole("button", { name: "打开已有会话确认" }));
    await screen.findByText("聊天暂不可达");
    expect(rpc.mock.calls.filter(([request]) => request.action === "handoff")).toHaveLength(1);
  });

  it("服务不可用仍能查看已保存规划，交接显示禁用", async () => {
    rpc.mockImplementation((request) => request.action === "status" ? Promise.resolve({ enabled: true, available: false, poll_interval_s: 2, error: { code: "authentication_failed", message: "令牌校验失败" } }) : respond(request));
    renderPage();
    await screen.findByText(markdown);
    expect(await screen.findByText(/authentication_failed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "生成想定并进入聊天" })).toBeDisabled();
  });
});

describe("规划 hook 生命周期", () => {
  it("断线保留草稿，重连只查询原规划及版本", async () => {
    const { result } = renderHook(() => usePlanning(scope, planId));
    await waitFor(() => expect(result.current.detail?.document?.revision).toBe(1));
    act(() => result.current.setDraft(`${markdown}\n离线编辑`));
    emitStatus("closed");
    expect(result.current.dirty).toBe(true);
    const before = rpc.mock.calls.length;
    emitStatus("open");
    await waitFor(() => expect(rpc.mock.calls.slice(before).some(([request]) => request.action === "get")).toBe(true));
    expect(result.current.draft).toBe(`${markdown}\n离线编辑`);
    expect(rpc).toHaveBeenLastCalledWith(expect.objectContaining({ action: "get", plan_id: planId, revision: 1 }));
    expect(rpc.mock.calls.some(([request]) => request.action === "create")).toBe(false);
  });

  it("显式生成重试和重复交接复用各自幂等 ID，不自动重发", async () => {
    const { result } = renderHook(() => usePlanning(scope, planId));
    await waitFor(() => expect(result.current.detail?.document?.revision).toBe(1));
    rpc.mockImplementationOnce(() => Promise.reject(new PlanningRpcError("timeout", "超时")));
    await act(async () => { await result.current.generate("相同任务"); });
    await act(async () => { await result.current.generate("相同任务"); });
    const creates = rpc.mock.calls.map(([request]) => request).filter((request) => request.action === "create");
    expect(creates).toHaveLength(2);
    expect(creates[0].generation_request_id).toBe(creates[1].generation_request_id);
    await act(async () => { await result.current.transfer(); });
    await act(async () => { await result.current.transfer(); });
    const transfers = rpc.mock.calls.map(([request]) => request).filter((request) => request.action === "handoff");
    expect(transfers).toHaveLength(2);
    expect(transfers[0].handoff_id).toBe(transfers[1].handoff_id);
  });

  it("版本切换清空选项与预览", async () => {
    const { result } = renderHook(() => usePlanning(scope, planId));
    await waitFor(() => expect(result.current.detail?.document?.revision).toBe(1));
    act(() => { result.current.setMode("selected"); result.current.setSelected(["P1:U1:0"]); });
    await act(async () => { await result.current.transfer(undefined, true); });
    expect(result.current.preview).not.toBeNull();
    await act(async () => { await result.current.loadRevision(2); });
    expect(result.current.selected).toEqual([]);
    expect(result.current.preview).toBeNull();
  });

  it("作业进行时轮询，断线和卸载停止，完成后不再轮询", async () => {
    vi.useFakeTimers();
    let running = true;
    rpc.mockImplementation((request) => request.action === "get" ? Promise.resolve(running
      ? { ...detail(), plan: { ...detail().plan, revision: 0, status: "running" }, document: null }
      : detail()) : respond(request));
    const { result, unmount } = renderHook(() => usePlanning(scope, planId));
    await act(async () => {});
    expect(result.current.detail?.plan.status).toBe("running");
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    const gets = () => rpc.mock.calls.filter(([request]) => request.action === "get").length;
    expect(gets()).toBe(2);
    emitStatus("closed");
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(gets()).toBe(2);
    emitStatus("open");
    await act(async () => {});
    running = false;
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(result.current.detail?.plan.status).toBe("completed");
    const completedGets = gets();
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(gets()).toBe(completedGets);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(gets()).toBe(completedGets);
  });

  it("旧工作区的迟到结果或错误都不污染新工作区", async () => {
    const lateGet = deferred<PlanningDetail>();
    const lateList = deferred<{ plans: [] }>();
    rpc.mockImplementation((request) => request.workspace_scope?.project_path === scope.project_path
      ? request.action === "get" ? lateGet.promise : request.action === "list" ? lateList.promise : respond(request)
      : respond(request));
    const { result, rerender } = renderHook(({ workspace }) => usePlanning(workspace, planId), { initialProps: { workspace: scope } });
    rerender({ workspace: { ...scope, project_path: "/tmp/other-workspace" } });
    await waitFor(() => expect(result.current.detail?.document?.revision).toBe(1));
    await act(async () => { lateGet.resolve(detail(9, "过期内容")); lateList.reject(new Error("旧工作区错误")); });
    expect(result.current.detail?.document?.markdown).toBe(markdown);
    expect(result.current.error).toBeNull();
  });
});

describe("原会话列表的交接恢复", () => {
  it("刷新失败与迟到空列表不移除已确认会话，服务返回后使用权威行", async () => {
    const { result } = renderHook(() => useSessions());
    await waitFor(() => expect(result.current.loading).toBe(false));
    act(() => result.current.rememberChat("scenario-chat", scope));
    vi.mocked(listSessions).mockRejectedValueOnce(new Error("列表暂不可达"));
    await act(async () => { await result.current.refresh(); });
    expect(result.current.error).toBe("列表暂不可达");
    expect(result.current.sessions[0]).toMatchObject({ chatId: "scenario-chat", workspaceScope: scope });
    await act(async () => { await result.current.refresh(); });
    expect(result.current.sessions).toHaveLength(1);
    const row = { ...result.current.sessions[0], title: "已生成想定", preview: "首轮错误与反馈" };
    vi.mocked(listSessions).mockResolvedValueOnce([row]);
    await act(async () => { await result.current.refresh(); });
    expect(result.current.sessions).toEqual([row]);
    expect(client.newChat).not.toHaveBeenCalled();
  });
});
