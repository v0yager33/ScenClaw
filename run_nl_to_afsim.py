#!/usr/bin/env python3
"""最小可用入口: 一句自然语言 → AFSIM 想定项目 (多文件目录)。

用法:
    export DEEPSEEK_API_KEY=sk-...
    python run_nl_to_afsim.py "两架歼-20在东海6000米高度巡逻，地面雷达站负责搜索预警" -o out

产出:
    out/                        # 项目根目录
    ├── main.txt                # 入口: 传给 mission.exe 的就是这个
    ├── _common.txt             # 聚合 include
    ├── sensors/<type>.txt      # 每种传感器一个文件
    ├── weapons/<type>.txt      # 每种武器效应一个文件
    ├── platforms/<type>.txt    # 每种平台类型一个文件
    ├── event_output.txt        # 事件输出配置
    ├── scenario_single.txt     # 单文件等价版 (方便 diff)
    └── output/                 # 运行时产物目录

只依赖 httpx / pydantic / jinja2 / pyyaml, 不需要 nanobot 完整安装。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="自然语言作战规划")
    parser.add_argument("-o", "--output", default="generated_scenario",
                        help="项目根目录路径 (如已有 .txt 后缀会自动去掉)")
    parser.add_argument("--center-lat", type=float, default=27.5)
    parser.add_argument("--center-lon", type=float, default=122.5)
    parser.add_argument("--duration", type=float, default=3600.0, help="任务时长(秒)")
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--rounds", type=int, default=4, help="最大 IR 修复轮数")
    parser.add_argument("--ir-out", help="同时导出 ScenarioIR JSON")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="单次 LLM 调用的输出预算 (含推理模型思维链); 平台多时调大")
    parser.add_argument("--timeout", type=int, default=120, help="单次 LLM 调用超时(秒)")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        return 2

    from nanobot.agent.tools.nl_scenario_planner import NLToAfsimTool

    kwargs = {
        "request": args.request,
        "output_path": args.output,
        "center_lat": args.center_lat,
        "center_lon": args.center_lon,
        "duration_s": args.duration,
        "max_repair_rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout,
        "include_code": False,
    }
    if args.scenario_id:
        kwargs["scenario_id"] = args.scenario_id

    raw = await NLToAfsimTool().execute(**kwargs)
    if not raw.lstrip().startswith("{"):
        print(raw, file=sys.stderr)
        return 1

    data = json.loads(raw)

    if data.get("out_of_scope"):
        print("超出适用范围:", data["reason"])
        return 1

    if not data["success"]:
        print("未能收敛成合法想定:", data.get("message", ""))
        for step in data.get("trace", []):
            for err in step.get("errors", [])[:5]:
                print(f"  round {step['round']} 错误: {err}")
        return 1

    validator = data["validator"]
    project = data.get("project") or {}
    if project:
        print(f"项目根目录 : {project['project_root']}")
        print(f"入口文件   : {project['entry']}")
        print(f"文件数     : {len(project['files'])} (多文件项目)")
        for f in project["files"]:
            print(f"  - {f}")
    else:
        print(f"想定文件   : {data.get('output_path', '(未指定)')}")
    print(f"规模       : {data['lines_of_code']} 行 (main.txt), {len(data['scenario_ir']['platforms'])} 个平台")
    print(f"修复轮次   : {data['rounds']}")
    print(f"验证器评分 : {validator['score']}/100 (error={validator['errors']} warning={validator['warnings']})")

    for label, key in (
        ("表达不了的需求", "capability_gaps"),
        ("未经真机验证的能力", "unverified_capabilities"),
        ("自动归一化", "normalizations"),
        ("遗留合理性问题", "plausibility_advisories"),
    ):
        items = data.get(key) or []
        if not items:
            continue
        print(f"\n{label} ({len(items)} 条):")
        for item in items:
            text = item["topic"] + " — " + item["why"] if isinstance(item, dict) else str(item)
            print(f"  - {text}")

    if args.ir_out:
        Path(args.ir_out).write_text(
            json.dumps(data["scenario_ir"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nScenarioIR : {args.ir_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
