#!/usr/bin/env python3
"""校验 AFSIM 想定文件，并可对整份语料做误报回归。

skill 自带这个脚本是为了两件事:

1. **单文件校验** —— 拿到任意 .txt 想定就能查语法/语义/一致性/战术四层问题;
2. **语料基准** —— 把检查器放到官方 demo 语料上跑, 量化误报。检查器的价值取决于
   它在"已知能跑的代码"上有多干净, 这个数字必须能随时复现, 否则每次改规则都是
   凭感觉。

用法:
    # 查单个文件
    python validate_afsim.py scenario.txt

    # 查一批
    python validate_afsim.py a.txt b.txt --quiet

    # 官方 demo 语料回归 (4749 文件, 约 45 秒)
    python validate_afsim.py --corpus /path/to/afsim_verify/examples/demos

    # 只看某条规则的触发形态, 用来定位误报根因
    python validate_afsim.py --corpus DIR --explain SYN_002
"""

from __future__ import annotations

import argparse
import collections
import glob
import re
import sys
import time
from pathlib import Path

# skill 在 nanobot/skills/afsim-scenario/scripts/ 下, 上溯四级到项目根
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from nanobot.agent.tools.scenario_validator_tool import ScenarioValidator  # noqa: E402

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_file(path: Path, quiet: bool = False) -> tuple[int, int]:
    """校验单个文件, 返回 (error 数, warning 数)。"""
    result = ScenarioValidator(_read(path)).validate_all()
    lines = _read(path).splitlines()

    header = (
        f"{path}  评分 {result.score}/100  "
        f"error={result.errors} warning={result.warnings} info={result.infos}"
    )
    if quiet and result.errors == 0:
        return result.errors, result.warnings

    print(header)
    for issue in sorted(result.issues, key=lambda i: (SEVERITY_ORDER.get(i.severity, 9), i.line_number)):
        if quiet and issue.severity != "error":
            continue
        line_no = issue.line_number or 0
        source = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
        print(f"  [{issue.severity:<7}] {issue.rule_id} L{line_no}: {issue.description}")
        if source:
            print(f"            源: {source[:100]}")
        if issue.suggestion:
            print(f"            建议: {issue.suggestion[:100]}")
    print()
    return result.errors, result.warnings


def run_corpus(corpus: Path, explain: str | None) -> int:
    """在整份语料上跑检查器, 报告误报分布。"""
    files = sorted(Path(p) for p in glob.glob(str(corpus / "**" / "*.txt"), recursive=True))
    if not files:
        print(f"[错误] {corpus} 下没有 .txt 文件")
        return 2

    started = time.time()
    by_rule: collections.Counter[str] = collections.Counter()
    per_file: collections.Counter[str] = collections.Counter()
    shapes: collections.Counter[str] = collections.Counter()
    clean = 0

    for path in files:
        text = _read(path)
        result = ScenarioValidator(text).validate_all()
        if result.errors == 0:
            clean += 1
        else:
            per_file[str(path.relative_to(corpus))] = result.errors
        for issue in result.issues:
            if issue.severity != "error":
                continue
            by_rule[issue.rule_id] += 1
            if explain and issue.rule_id == explain:
                line_no = issue.line_number or 0
                lines = text.splitlines()
                source = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
                shapes[re.sub(r"\d+(?:\.\d+)?", "N", source)[:70]] += 1

    total = len(files)
    print(f"语料: {corpus}")
    print(f"扫描 {total} 个文件, 耗时 {time.time() - started:.0f}s")
    print(f"0-error 文件: {clean}/{total}  ({100.0 * clean / total:.1f}%)")
    print(f"error 总数: {sum(by_rule.values())}")
    if by_rule:
        print("规则分布:")
        for rule, count in by_rule.most_common():
            print(f"   {rule:<10} {count}")
    if per_file:
        print("\nerror 最多的文件:")
        for name, count in per_file.most_common(10):
            print(f"   {count:5d}  {name}")
    if explain:
        print(f"\n{explain} 的触发行形态:")
        for shape, count in shapes.most_common(15):
            print(f"   {count:5d}  {shape}")

    # 检查器在已知能跑的代码上应当接近零误报; 留 1% 余量给病态数据文件
    return 0 if clean >= total * 0.99 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="要校验的 AFSIM .txt 文件")
    parser.add_argument("--corpus", help="对整个目录做误报回归")
    parser.add_argument("--explain", help="只展开指定规则的触发形态, 如 SYN_002")
    parser.add_argument("--quiet", action="store_true", help="只输出 error")
    args = parser.parse_args()

    if args.corpus:
        return run_corpus(Path(args.corpus).expanduser(), args.explain)

    if not args.files:
        parser.print_help()
        return 2

    total_errors = 0
    for name in args.files:
        path = Path(name).expanduser()
        if not path.is_file():
            print(f"[跳过] 文件不存在: {path}")
            continue
        errors, _ = check_file(path, quiet=args.quiet)
        total_errors += errors

    print(f"合计 error: {total_errors}")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
