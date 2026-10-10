#!/usr/bin/env python3
"""自然语言 → AFSIM 想定 的端到端 smoke 测试。

用法:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/smoke_nl_to_afsim.py                 # 跑内置样例
    python scripts/smoke_nl_to_afsim.py --request-file plan.md
    python scripts/smoke_nl_to_afsim.py --keep          # 保留产物便于人工检查

检查三件事:
  1. 自然语言能否收敛成合法 ScenarioIR (含 IR 层修复轮次);
  2. 编译出的 AFSIM 文本能否同时通过静态 lint 与四层验证器;
  3. 同一份 IR 重复编译是否字节一致 (确定性)。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_REQUEST = """在东海方向组织一次防空压制行动。

蓝方兵力：2 架 J-16D 电子战机在 8000 米高度沿由西向东航线突入，机上装反辐射导弹；
1 架 KJ-500 预警机在后方 400 公里外 9000 米高度盘旋，负责用雷达搜索并跟踪敌方目标。

红方兵力：岛礁上部署 1 部远程警戒雷达阵地和 1 套 S-400 火控雷达阵地，都是固定不动的地面目标。

行动要求：预警机先探测到红方两个雷达阵地并建立稳定航迹；确认目标后，
由 J-16D 在任务开始约 20 分钟时发射反辐射导弹打击 S-400 火控雷达。
整个任务时长 1 小时。
"""


def _fmt_trace(trace: list[dict]) -> str:
    lines = []
    for step in trace:
        flag = "OK " if step["ok"] else "FAIL"
        lines.append(f"    round {step['round']} [{step['stage']:<7}] {flag}")
        for err in step.get("errors", [])[:6]:
            lines.append(f"        错误: {err}")
        for adv in step.get("advisories", [])[:6]:
            lines.append(f"        合理性: {adv}")
        extra = len(step.get("errors", [])) - 6
        if extra > 0:
            lines.append(f"        ... 另有 {extra} 条错误")
    return "\n".join(lines) or "    (无)"


async def run(request: str, output_path: Path, max_repair_rounds: int) -> int:
    from nanobot.afsim import AfsimCompiler, ScenarioIR
    from nanobot.agent.tools.nl_scenario_planner import NLToAfsimTool

    print("=" * 68)
    print("自然语言 → AFSIM 想定 端到端 smoke")
    print("=" * 68)
    print(f"模型      : {os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')}")
    print(f"输入长度  : {len(request)} 字符")
    print(f"最大修复轮: {max_repair_rounds}")
    print()

    started = time.time()
    raw = await NLToAfsimTool().execute(
        request=request,
        output_path=str(output_path),
        center_lat=27.5,
        center_lon=122.5,
        duration_s=3600,
        scenario_id="smoke_sead_eastsea",
        max_repair_rounds=max_repair_rounds,
        include_code=True,
    )
    elapsed = time.time() - started

    if not raw.lstrip().startswith("{"):
        print(f"[FAIL] 工具返回非 JSON: {raw[:500]}")
        return 1

    data = json.loads(raw)
    print(f"耗时      : {elapsed:.1f}s")
    print(f"收敛       : {data['success']}  (共 {data['rounds']} 轮)")
    print("IR 层修复轨迹:")
    print(_fmt_trace(data.get("trace", [])))
    print()

    if not data["success"]:
        print("[FAIL] 未能收敛成合法 ScenarioIR")
        print(data.get("message", ""))
        return 1

    ir = data["scenario_ir"]
    mission = ir.get("mission", {})
    platforms = ir.get("platforms", [])
    sides = {p.get("side") for p in platforms}
    print("── 抽取结果 ──")
    print(f"  scenario_id : {data['scenario_id']}")
    print(f"  任务         : {mission.get('name', '')}")
    print(f"  时长         : {mission.get('duration_s')} s")
    print(f"  平台         : {len(platforms)} 个, 阵营 {sorted(s for s in sides if s)}")
    print(f"  传感器类型   : {[s.get('sensor_type_id') for s in ir.get('sensors', [])]}")
    print(f"  断言         : {len(ir.get('assertions', []))} 条")
    print(f"  交战         : {len(ir.get('engagements', []))} 条")
    for assumption in mission.get("assumptions", [])[:5]:
        print(f"    假设: {assumption}")
    print()

    validator = data["validator"]
    print("── 想定质量 ──")
    print(f"  AFSIM 行数   : {data['lines_of_code']}")
    print(f"  验证器评分   : {validator['score']}/100 "
          f"(error={validator['errors']} warning={validator['warnings']})")
    for issue in validator["issues"][:8]:
        print(f"    [{issue['severity']}] {issue['rule']} L{issue['line']}: {issue['description'][:70]}")
    print(f"  静态 lint    : {len(data['lint_diagnostics'])} 条诊断")
    advisories = data.get("plausibility_advisories", [])
    print(f"  合理性遗留   : {len(advisories)} 条")
    for advisory in advisories:
        print(f"    - {advisory}")
    print(f"  输出文件     : {data.get('output_path', '(未写盘)')}")
    print()

    # 确定性检查: 同一份 IR 编译两次必须字节一致
    scenario = ScenarioIR.model_validate(ir)
    first = AfsimCompiler().compile(scenario)
    second = AfsimCompiler().compile(scenario)
    deterministic = first == second == data["afsim_code"]
    print(f"── 确定性     : {'一致' if deterministic else '不一致'}")
    print()

    print("── 生成的 AFSIM 想定 (前 40 行) ──")
    for line in data["afsim_code"].splitlines()[:40]:
        print("  " + line)
    print()

    ok = (
        data["success"]
        and validator["errors"] == 0
        and deterministic
        and not [d for d in data["lint_diagnostics"] if d.startswith("error")]
    )
    print("=" * 68)
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 68)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", help="自然语言规划文件, 默认用内置样例")
    parser.add_argument("--output", help="想定输出路径, 默认写到临时目录")
    parser.add_argument("--max-repair-rounds", type=int, default=3)
    parser.add_argument("--keep", action="store_true", help="保留输出文件")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[SKIP] 未设置 DEEPSEEK_API_KEY 环境变量")
        return 2

    request = (
        Path(args.request_file).read_text(encoding="utf-8")
        if args.request_file
        else SAMPLE_REQUEST
    )
    output = Path(args.output) if args.output else (
        Path(tempfile.mkdtemp(prefix="scenclaw_smoke_")) / "generated_scenario.txt"
    )

    code = asyncio.run(run(request, output, args.max_repair_rounds))
    if not args.keep and not args.output and output.exists():
        print(f"(产物保留在 {output}, 加 --keep 或 --output 可自定义)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
