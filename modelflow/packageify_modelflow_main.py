#!/usr/bin/env python3
"""
Create a package-style Modelflow layout from a source folder.

This version:
- uses the explicit file list from the CMD file
- copies files into modelflow_clean/modelflow/
- rewrites intra-project imports inside the package modules
- creates top-level compatibility wrappers
- uses a main() wrapper pattern instead of runpy
- creates docs/, tests/, and scripts/ folders
- creates docs/index.md

Usage
-----
python packageify_modelflow_main.py SOURCE_FOLDER
python packageify_modelflow_main.py SOURCE_FOLDER --out OUTPUT_FOLDER
python packageify_modelflow_main.py SOURCE_FOLDER --force
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


FILE_LIST = [
    "modelBLfunk.py",
    "model_Excel.py",
    "model_cvx.py",
    "model_financial_stability.py",
    "model_latex_class.py",
    "modelclass.py",
    "modeldashsidebar.py",
    "modeldekom.py",
    "modeldisplay.py",
    "modelhelp.py",
    "modelinvert.py",
    "modeljupyter.py",
    "modelmanipulation.py",
    "modelmf.py",
    "modelnet.py",
    "modelnewton.py",
    "modelnormalize.py",
    "modelpattern.py",
    "modelreport.py",
    "modeluserfunk.py",
    "modelvis.py",
    "modelwidget.py",
    "modelwidget_input.py",
]

MODULES = {Path(name).stem for name in FILE_LIST}


def split_import_items(spec):
    return [item.strip() for item in spec.split(",") if item.strip()]


def rewrite_import_line(line):
    indent = line[: len(line) - len(line.lstrip())]
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        return None

    m_from = re.match(r"^from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\s+(.+)$", stripped)
    if m_from:
        mod = m_from.group(1)
        rest = m_from.group(2)
        if mod in MODULES:
            return [f"{indent}from .{mod} import {rest}"]
        return None

    m_import = re.match(r"^import\s+(.+)$", stripped)
    if m_import:
        specs = split_import_items(m_import.group(1))
        local_specs = []
        external_specs = []

        for spec in specs:
            parts = spec.split()
            base = parts[0]

            if base in MODULES:
                local_specs.append(spec)
            else:
                external_specs.append(spec)

        if not local_specs:
            return None

        out = []
        if external_specs:
            out.append(f"{indent}import {', '.join(external_specs)}")

        out.append(f"{indent}from . import {', '.join(local_specs)}")
        return out

    return None


def rewrite_source(text):
    out_lines = []

    for line in text.splitlines():
        rewritten = rewrite_import_line(line)
        if rewritten is None:
            out_lines.append(line)
        else:
            out_lines.extend(rewritten)

    if text.endswith("\n"):
        return "\n".join(out_lines) + "\n"
    return "\n".join(out_lines)


def build_wrapper(module_name, package_name):
    return (
        '"""Auto-generated compatibility wrapper."""\n\n'
        f"from {package_name}.{module_name} import *\n\n"
        'if __name__ == "__main__":\n'
        "    try:\n"
        f"        from {package_name}.{module_name} import main as _main\n"
        "    except ImportError:\n"
        "        pass\n"
        "    except Exception as e:\n"
        '        raise SystemExit(f"Error importing main(): {e}")\n'
        "    else:\n"
        "        _main()\n"
    )


def build_init():
    return (
        '"""Modelflow package."""\n\n'
        "try:\n"
        "    from .modelclass import model\n"
        "except Exception:\n"
        "    pass\n"
    )


def build_readme(package_name):
    return f"""# {package_name}

Generated package layout with:
- top-level compatibility wrappers
- package copies of the original modules
- rewritten intra-project imports
- wrapper execution through main() when available
- docs/, tests/, and scripts/ folders

Examples:

from modelclass import model
from {package_name}.modelclass import model
from {package_name} import model

Run as script:

python modelclass.py
python -m {package_name}.modelclass
"""


def build_pyproject(package_name):
    return f"""[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
version = "0.1.0"
description = "Packaged Modelflow codebase with compatibility wrappers"
readme = "README.md"
requires-python = ">=3.9"

[tool.setuptools]
include-package-data = true

[tool.setuptools.packages.find]
where = ["."]
include = ["{package_name}*"]
"""


def build_docs_index(package_name):
    return f"""# {package_name} documentation

This folder was generated automatically.

Suggested next steps:
- Add installation instructions
- Add quick start examples
- Document the main `model` class
- Document the main modules
"""


def build_test_smoke(package_name):
    return f"""def test_import_package():
    import {package_name}


def test_import_modelclass_wrapper():
    import modelclass
"""


def copy_and_rewrite(source, package_dir):
    copied = []

    for filename in FILE_LIST:
        src = source / filename

        if not src.exists():
            print(f"WARNING missing: {src}")
            continue

        text = src.read_text(encoding="utf-8")
        rewritten = rewrite_source(text)

        dst = package_dir / filename
        dst.write_text(rewritten, encoding="utf-8")

        copied.append(Path(filename).stem)

    return copied


def write_wrappers(out_dir, package_name, module_names):
    for mod in module_names:
        wrapper = build_wrapper(mod, package_name)
        (out_dir / f"{mod}.py").write_text(wrapper, encoding="utf-8")


def create_support_structure(out_dir, package_name):
    docs_dir = out_dir / "docs"
    tests_dir = out_dir / "tests"
    scripts_dir = out_dir / "scripts"

    docs_dir.mkdir(exist_ok=True)
    tests_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)

    (docs_dir / "index.md").write_text(build_docs_index(package_name), encoding="utf-8")
    (tests_dir / "test_smoke.py").write_text(build_test_smoke(package_name), encoding="utf-8")
    (scripts_dir / ".gitkeep").write_text("", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Create package-style Modelflow layout and rewrite imports."
    )

    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing the original model files. Default: folder where this script is located."
    )
    parser.add_argument("--out", type=Path, help="Output folder. Default: SOURCE/modelflow_clean")
    parser.add_argument("--package-name", default="modelflow", help="Package name. Default: modelflow")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output folder")

    args = parser.parse_args()

    source = args.source.resolve()

    if not source.is_dir():
        print(f"ERROR: source is not a folder: {source}")
        sys.exit(1)

    out_dir = args.out.resolve() if args.out else source / "modelflow_clean"
    package_name = args.package_name

    if out_dir.exists():
        if not args.force:
            print(f"ERROR: output folder exists: {out_dir}")
            print("Use --force to overwrite it.")
            sys.exit(1)

        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    package_dir = out_dir / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_and_rewrite(source, package_dir)

    if not copied:
        print("ERROR: no files copied")
        sys.exit(1)

    (package_dir / "__init__.py").write_text(build_init(), encoding="utf-8")
    (out_dir / "README.md").write_text(build_readme(package_name), encoding="utf-8")
    (out_dir / "pyproject.toml").write_text(build_pyproject(package_name), encoding="utf-8")

    write_wrappers(out_dir, package_name, copied)
    create_support_structure(out_dir, package_name)

    print("Created package in:", out_dir)

    print("\nModules copied:")
    for mod in copied:
        print(" ", mod + ".py")

    missing = [f for f in FILE_LIST if not (source / f).exists()]

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(" ", f)

    print("\nCreated support folders:")
    print("  docs/")
    print("  tests/")
    print("  scripts/")
if __name__ == "__main__":
    main()