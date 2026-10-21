"""Export the Python examples in the mathematical modeling notes.

Run with Python 3.11. Add --check to execute every exported example without
opening plot windows. The Markdown files remain the source of truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "src" / "content" / "notes"
DESTINATION = ROOT / "public" / "downloads" / "math-modeling"
SECTION_NAMES = [
    "学习目标与先修知识", "问题背景与使用时机", "核心概念与符号",
    "手算一个最小例子", "建模步骤与完整案例", "Python 实现与结果解释",
    "方法选择与常见误区", "练习", "参考答案", "复习卡片与下一步", "参考资料",
]


def collect_lessons():
    lessons = []
    for path in NOTES.glob("mathematical-modeling-*.md"):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        order_match = re.search(r"^order:\s*(\d+)$", text, re.MULTILINE)
        if not title_match or not order_match:
            raise ValueError(f"Missing title or reading order: {path.name}")
        title = json.loads(title_match.group(1))
        code = re.findall(r"^```python\s*\n(.*?)^```[ \t]*$", text, re.MULTILINE | re.DOTALL)
        if not code:
            raise ValueError(f"No runnable Python example: {path.name}")
        # Remove fenced blocks before checking the actual Markdown headings.
        prose = re.sub(r"^```[^\n]*\n.*?^```[ \t]*$", "", text, flags=re.MULTILINE | re.DOTALL)
        headings = re.findall(r"^## (.+)$", prose, re.MULTILINE)
        if headings != SECTION_NAMES:
            raise ValueError(f"Inconsistent learning structure: {path.name}: {headings}")
        lessons.append((int(order_match.group(1)), path, title, code))
    lessons.sort(key=lambda item: item[0])
    if [item[0] for item in lessons] != list(range(1, len(lessons) + 1)):
        raise ValueError("Reading order must be consecutive and unique")
    return lessons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run every exported example")
    args = parser.parse_args()
    lessons = collect_lessons()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    outputs = []
    for order, source, title, blocks in lessons:
        destination = DESTINATION / f"{source.stem}.py"
        header = (
            "# -*- coding: utf-8 -*-\n"
            f"# {title}\n"
            f"# Source: /notes/{source.stem}/\n"
            "# Generated from the Markdown examples; edit the note and export again.\n\n"
        )
        program = header + "\n\n".join(block.rstrip() for block in blocks) + "\n"
        compile(program, str(destination), "exec")
        destination.write_text(program, encoding="utf-8")
        outputs.append(destination)

    requirements = DESTINATION / "requirements.txt"
    requirements.write_text(
        "# Reproduction environment: Python 3.11.7\n"
        "numpy==1.26.4\npandas==2.1.4\nscipy==1.11.4\nmatplotlib==3.8.0\n"
        "scikit-learn==1.2.2\nstatsmodels==0.14.0\nsympy==1.12\n",
        encoding="utf-8",
    )
    outputs.append(requirements)
    readme = DESTINATION / "README.txt"
    readme.write_text(
        "数学建模零基础学习示例\n\n"
        "对应笔记：https://neurpulse.github.io/notes/#category-2\n"
        "学习地图：https://neurpulse.github.io/notes/mathematical-modeling-roadmap/\n\n"
        "使用 Python 3.11，解压后进入当前文件夹。建议创建独立虚拟环境，\n"
        "安装依赖：python -m pip install -r requirements.txt\n"
        "先运行：python mathematical-modeling-roadmap.py\n"
        "其他示例：python 对应的英文文件名.py\n\n"
        "图形示例会打开图窗。案例使用明确生成的教学数据或内置数据，\n"
        "无需额外 CSV、联网下载数据或安装 PyTorch。合成结果不代表真实赛题复现。\n"
        "固定随机种子有助于复现，浮点值和优化结果仍可能随环境略有差异。\n\n"
        "requirements.txt 记录实际验证环境，不表示这些版本是最新版本。\n"
        "每个脚本对应正文完整案例，阅读公式、假设与结果解释后再运行。\n\n"
        "顺序与文件\n"
        + "\n".join(f"{order:02d} {source.stem}.py  {title}" for order, source, title, _ in lessons)
        + "\n\n参考学习资料：Datawhale《数学建模导论》\n"
        "https://datawhalechina.github.io/intro-mathmodel/#/\n",
        encoding="utf-8",
    )
    outputs.append(readme)
    archive = DESTINATION / "math-modeling-examples.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in outputs:
            bundle.write(path, arcname=path.name)
    print(f"Exported {len(lessons)} scripts and archive: {archive}", flush=True)

    if args.check:
        launch = (
            "import runpy,sys; import matplotlib; matplotlib.use('Agg'); "
            "import matplotlib.pyplot as plt; "
            "plt.show=lambda *a,**k:plt.close('all'); "
            "runpy.run_path(sys.argv[1],run_name='__main__')"
        )
        environment = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        failed = []
        for path in outputs:
            if path.suffix != ".py":
                continue
            result = subprocess.run(
                [sys.executable, "-c", launch, str(path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, env=environment,
            )
            print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {path.name}", flush=True)
            if result.returncode:
                failed.append(path.name)
                print(result.stderr[-4000:], flush=True)
        if failed:
            raise SystemExit(f"Example execution failed: {failed}")


if __name__ == "__main__":
    main()
