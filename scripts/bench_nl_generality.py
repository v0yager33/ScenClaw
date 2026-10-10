#!/usr/bin/env python3
"""自然语言 → AFSIM 想定 的通用性基准。

目的不是刷通过率, 而是**暴露失败面**: 拿一批覆盖不同域 (空/海/陆/天)、不同任务
类型 (拦截/巡逻/打击/侦察/护航/电子战/通信)、不同信息完整度 (充分/残缺/超纲) 的
自然语言需求灌进去, 看哪些能成、哪些退化、哪些直接失败。

除了"是否收敛", 还检测**静默退化** —— 需求里明确要海上机动, 结果编译器只有
WSF_AIR_MOVER, 平台被做成静止的; 这种情况语法全绿、评分 100, 但需求根本没被满足。
这类退化比直接报错更危险, 所以单独标记出来。

用法:
    export DEEPSEEK_API_KEY=sk-...
    python scripts/bench_nl_generality.py                 # 跑全部
    python scripts/bench_nl_generality.py --case sea_patrol
    python scripts/bench_nl_generality.py --workers 4 --out bench_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchCase:
    """一条基准用例。

    ``expect_mobile`` / ``expect_sensor`` / ``expect_weapon`` 描述需求语义上要求什么,
    用来检测静默退化 (需求要机动却生成静止平台等)。
    ``should_fail`` 标记本就该被拒绝的需求 (信息不足或超出能力范围)。
    """
    case_id: str
    domain: str
    request: str
    expect_mobile: bool = True
    expect_sensor: bool = True
    expect_weapon: bool = False
    should_fail: bool = False
    notes: str = ""


CASES: list[BenchCase] = [
    # ── 空域: 现有能力应当覆盖 ──
    BenchCase(
        case_id="air_intercept",
        domain="air",
        request=(
            "台海方向防空拦截。红方2架轰炸机从东侧300公里外9000米高度向西突防。"
            "蓝方1个地面雷达站负责搜索预警，2架歼-11从西侧起飞升空到8000米向东拦截，"
            "携带空空导弹，在任务开始15分钟时对第一架轰炸机开火。任务时长40分钟。"
        ),
        expect_weapon=True,
        notes="基线用例, 纯空域 + 地面静止雷达",
    ),
    BenchCase(
        case_id="air_cap_patrol",
        domain="air",
        request=(
            "组织一次战斗空中巡逻。2架歼-20在东海某空域6000米高度沿南北方向来回巡逻，"
            "巡逻边长约80公里，机载雷达全程开机搜索。任务时长2小时，不需要开火。"
        ),
        expect_weapon=False,
        notes="多航点往返航路, 检验 route 生成",
    ),
    # ── 海域: 需要 WSF_SURFACE_MOVER ──
    BenchCase(
        case_id="sea_patrol",
        domain="sea",
        request=(
            "一艘052D驱逐舰在南海以18节航速由西向东巡逻航行约120公里，"
            "全程用舰载相控阵雷达搜索空中目标。红方1架反舰导弹载机从东侧接近。"
            "任务时长3小时。"
        ),
        notes="舰船机动: 需要海面 mover",
    ),
    BenchCase(
        case_id="sea_anti_ship",
        domain="sea",
        request=(
            "反舰打击。蓝方4架轰-6K从大陆起飞，8000米高度向东南方向飞行600公里，"
            "在距目标200公里处发射反舰导弹，打击红方1艘正在以20节向北航行的巡洋舰。"
            "红方舰艇有舰载雷达。任务时长2小时。"
        ),
        expect_weapon=True,
        notes="空对海 + 双方都机动",
    ),
    # ── 陆域: 需要 WSF_GROUND_MOVER ──
    BenchCase(
        case_id="land_advance",
        domain="land",
        request=(
            "地面推进。蓝方1个装甲连(4辆主战坦克)沿公路以每小时40公里向北推进30公里，"
            "红方2套地面防空系统在北侧15公里处固定部署，配备搜索雷达。"
            "坦克在接近到8公里时对第一套防空系统开火。任务时长1小时。"
        ),
        expect_weapon=True,
        notes="地面机动: 需要陆地 mover",
    ),
    # ── 天域: 需要空间 mover ──
    BenchCase(
        case_id="space_isr",
        domain="space",
        request=(
            "卫星侦察。1颗光学侦察卫星在500公里高度轨道上过境，"
            "对地面3个固定目标进行成像侦察。任务时长90分钟。"
        ),
        expect_weapon=False,
        notes="空间域: 高度 500km 超出 IR 的 altitude 上限 100km",
    ),
    # ── 真实雷达参数: 需要 WSF_RADAR_SENSOR ──
    BenchCase(
        case_id="radar_detection",
        domain="air",
        request=(
            "雷达探测试验。1部地面预警雷达工作在S波段，峰值功率100千瓦，"
            "天线增益35dB，对RCS为5平方米的目标探测距离约250公里。"
            "1架战斗机从300公里外以10000米高度向雷达飞来。任务时长30分钟。"
        ),
        expect_mobile=False,
        expect_weapon=False,
        notes="给了雷达方程参数, 几何传感器无法表达功率/增益/RCS",
    ),
    # ── 导弹飞行: 需要 WSF_EXPLICIT_WEAPON ──
    BenchCase(
        case_id="missile_flyout",
        domain="air",
        request=(
            "导弹拦截。红方1枚弹道导弹从东侧发射，蓝方1套反导系统发射拦截弹迎击，"
            "要求能看到拦截弹的飞行轨迹和最终命中判定。任务时长20分钟。"
        ),
        expect_weapon=True,
        notes="要求导弹飞行轨迹, 隐式武器只有发射事件没有弹道",
    ),
    # ── 电子战: 完全未覆盖 ──
    BenchCase(
        case_id="electronic_warfare",
        domain="air",
        request=(
            "电子战压制。2架歼-16D对红方警戒雷达实施频率隔离式阻塞干扰，"
            "干扰频段避开我方数据链的960-1215MHz，压制敌雷达探测能力，"
            "同时用ESM侦收定位辐射源。任务时长1小时。"
        ),
        expect_weapon=False,
        notes="干扰/ESM 完全不在能力白名单内",
    ),
    # ── 通信数据链: 完全未覆盖 ──
    BenchCase(
        case_id="datalink",
        domain="air",
        request=(
            "数据链协同。1架预警机通过Link-16把探测到的航迹分发给3架战斗机，"
            "战斗机依据共享航迹进行拦截引导。任务时长1小时。"
        ),
        expect_weapon=False,
        notes="comm/track 分发未覆盖",
    ),
    # ── 信息残缺: 应当靠假设补全, 而不是崩掉 ──
    BenchCase(
        case_id="minimal_info",
        domain="air",
        request="两架飞机在空中对抗。",
        expect_weapon=False,
        notes="极简需求, 考察假设补全能力",
    ),
    BenchCase(
        case_id="vague_objective",
        domain="air",
        request="做一个能看到雷达探测过程的简单想定就行。",
        expect_mobile=False,
        expect_weapon=False,
        notes="目标模糊, 无兵力构成",
    ),
    # ── 超纲/应拒绝 ──
    BenchCase(
        case_id="out_of_scope_civil",
        domain="other",
        request="帮我做一个城市早高峰地铁客流仿真想定，统计各站点进出站人数。",
        should_fail=True,
        notes="与 AFSIM 作战仿真域无关, 应当拒绝而不是硬编",
    ),
]


# ── 退化检测 ────────────────────────────────────────────────────────


def inspect_degradation(
    case: BenchCase, ir: dict, gap_topics: set[str]
) -> list[str]:
    """对照需求语义检查生成结果是否发生**静默**退化。

    已被 capability_gaps 显式声明的缺口不算退化 —— 工具如实说了"这条做不到",
    使用者不会被误导。只有"没声明却悄悄降级"才是问题。

    注意 mover 与域的匹配不在这里查: ScenarioIR 的 schema 已经强制
    domain↔mover 一致, 违反根本进不了输出。而且真实想定往往跨域 (轰炸机打军舰),
    按用例级单一域去要求"所有平台都用水面 mover"是错的。这里改为检查
    **需求强调的那个域是否真的出现在平台清单里**。
    """
    flags: list[str] = []
    platforms = ir.get("platforms", [])
    if not platforms:
        return ["生成结果没有任何平台"]

    declared_domains = {
        intent.get("platform_id"): intent.get("domain")
        for intent in ir.get("mission", {}).get("platforms", [])
    }
    space_gap = "空间轨道运动" in gap_topics

    # 需求强调的域必须真的建出来 (海上任务不能一条船都没有)
    if case.domain in ("sea", "land", "space"):
        if case.domain not in set(declared_domains.values()):
            flags.append(
                f"需求主体是 {case.domain} 域, 但平台清单里没有任何 {case.domain} 域平台: "
                f"实际为 {sorted(set(v for v in declared_domains.values() if v))}"
            )

    mobile = [p for p in platforms if p.get("mover_type")]
    if case.expect_mobile and not mobile and not space_gap:
        flags.append("需求要求平台机动, 但没有任何平台带 mover (退化成全静止)")

    if case.expect_sensor and not any(p.get("sensors") for p in platforms):
        flags.append("需求涉及探测, 但没有任何平台装传感器")

    if case.expect_weapon and not any(p.get("weapons") for p in platforms):
        flags.append("需求涉及打击, 但没有任何平台挂载武器")

    for platform in platforms:
        altitude = platform.get("position", {}).get("altitude_m", 0)
        if altitude >= 100_000 and not space_gap:
            flags.append(
                f"平台 {platform.get('platform_id')} 高度被压到 {altitude}m 上限"
            )
            break

    return flags


@dataclass
class CaseResult:
    case_id: str
    domain: str
    success: bool = False
    rounds: int = 0
    score: int = 0
    errors: int = 0
    advisories: list[str] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    hard_errors: list[str] = field(default_factory=list)
    normalizations: list[str] = field(default_factory=list)
    out_of_scope: str = ""
    elapsed_s: float = 0.0
    platforms: int = 0
    verdict: str = ""


def run_case(case: BenchCase, max_rounds: int) -> CaseResult:
    from nanobot.agent.tools.nl_scenario_planner import ScenarioIRPlanner
    from nanobot.agent.tools.react_engine import DeepSeekClient
    from nanobot.agent.tools.scenario_validator_tool import ScenarioValidator

    result = CaseResult(case_id=case.case_id, domain=case.domain)
    started = time.time()
    try:
        planner = ScenarioIRPlanner(
            llm=DeepSeekClient(), max_repair_rounds=max_rounds
        )
        plan = planner.plan(case.request, {})
        result.rounds = plan.rounds
        result.success = plan.success
        result.advisories = plan.advisories
        result.out_of_scope = plan.out_of_scope
        result.unverified = plan.unverified_capabilities
        result.normalizations = plan.normalizations
        gap_topics = {gap.topic for gap in plan.capability_gaps}
        result.gaps = sorted(gap_topics)
        if plan.success and plan.scenario_ir:
            result.platforms = len(plan.scenario_ir.get("platforms", []))
            result.degradations = inspect_degradation(case, plan.scenario_ir, gap_topics)
            validation = ScenarioValidator(plan.afsim_code).validate_all()
            result.score = validation.score
            result.errors = validation.errors
        else:
            failed = [a for a in plan.attempts if a.errors]
            result.hard_errors = failed[-1].errors[:6] if failed else []
    except Exception as exc:
        result.hard_errors = [f"{type(exc).__name__}: {exc}"]
    result.elapsed_s = time.time() - started

    if case.should_fail:
        if result.out_of_scope:
            result.verdict = "PASS(正确拒绝)"
        elif not result.success:
            result.verdict = "PASS(未生成)"
        else:
            result.verdict = "FAIL(应拒绝却生成了)"
    elif result.out_of_scope:
        result.verdict = "FAIL(误判超纲)"
    elif not result.success:
        result.verdict = "FAIL(未收敛)"
    elif result.degradations:
        result.verdict = "DEGRADED(静默降级)"
    elif result.gaps:
        result.verdict = "PASS(部分建模+缺口已声明)"
    else:
        result.verdict = "PASS"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="只跑指定 case_id, 可重复")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--out", help="把完整结果写成 JSON")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[SKIP] 未设置 DEEPSEEK_API_KEY")
        return 2

    cases = CASES
    if args.case:
        wanted = set(args.case)
        cases = [c for c in CASES if c.case_id in wanted]
        if not cases:
            print(f"没有匹配的 case: {wanted}")
            return 2

    print("=" * 92)
    print(f"自然语言 → AFSIM 想定 通用性基准  ({len(cases)} 个用例, {args.workers} 并发)")
    print("=" * 92)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda c: run_case(c, args.max_rounds), cases))

    print(f"\n{'case_id':<22}{'域':<7}{'结论':<34}{'轮':<4}{'分':<5}{'平台':<5}{'缺口':<5}{'耗时':>6}")
    print("-" * 96)
    for r in results:
        print(f"{r.case_id:<22}{r.domain:<7}{r.verdict:<34}{r.rounds:<4}"
              f"{r.score:<5}{r.platforms:<5}{len(r.gaps):<5}{r.elapsed_s:>5.0f}s")

    print("\n" + "=" * 96)
    print("明细")
    print("=" * 96)
    for r in results:
        if r.verdict == "PASS" and not r.unverified and not r.normalizations:
            continue
        print(f"\n[{r.case_id}] {r.verdict}")
        if r.out_of_scope:
            print(f"    拒绝理由 : {r.out_of_scope}")
        for item in r.hard_errors:
            print(f"    硬错误   : {item}")
        for item in r.degradations:
            print(f"    静默降级 : {item}")
        for item in r.gaps:
            print(f"    已声明缺口: {item}")
        for item in r.advisories:
            print(f"    合理性   : {item}")
        for item in r.normalizations:
            print(f"    归一化   : {item}")
        for item in r.unverified:
            print(f"    未验证能力: {item}")

    passed = sum(1 for r in results if r.verdict.startswith("PASS"))
    degraded = sum(1 for r in results if r.verdict.startswith("DEGRADED"))
    failed = len(results) - passed - degraded
    print("\n" + "=" * 96)
    print(f"汇总: PASS {passed}  DEGRADED {degraded}  FAIL {failed}   / 共 {len(results)}")
    print("=" * 96)

    if args.out:
        Path(args.out).write_text(
            json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"完整结果已写入 {args.out}")

    return 0 if failed == 0 and degraded == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
