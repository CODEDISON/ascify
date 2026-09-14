#!/usr/bin/env python3
"""Check the source repository's product identity, files, and documentation links.

Runs with the Python standard library, including from a source archive without
Git. It does not format or modify files, download dependencies, or run devices.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote


REQUIRED = (
    "README.md", "README.zh-CN.md", "CONTRIBUTING.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "LICENSES/Clang.txt", "LICENSES/LLVM.txt",
    "CHANGELOG.md", "CMakeLists.txt", "build.sh", "run.sh", ".gitignore",
    ".gitattributes", ".editorconfig", ".github/workflows/host-contracts.yml",
    ".github/conda-native.yml",
    "docs/README.md", "docs/user-guide.en.md", "docs/user-guide.zh-CN.md",
    "docs/validation-matrix.md", "docs/release-process.md",
)
PRODUCT_DIRECTORIES = ("src", "include", "frontend_compat", "runtime", "acl_cub", "cmake")
PRODUCT_FILES = ("CMakeLists.txt", "build.sh", "run.sh")
IGNORED_DIRECTORIES = {
    ".git", ".work", "build", "ascify_install", "dist", "__pycache__",
    ".pytest_cache", ".cache", ".idea", ".vscode", "CMakeFiles",
}
ARTIFACT_SUFFIXES = (
    ".o", ".obj", ".so", ".dylib", ".dll", ".a", ".lib", ".exe",
    ".pyc", ".pyo", ".log", ".tar", ".tar.gz", ".zip", ".cu.dpp",
)
# Match identifiers/words, not substrings such as 'ownership' or 'getParamDecl'.
LEGACY_TARGET = re.compile(
    r"\b(?:hip\w*|rocm\w*|hcc\w*|cuda2hip|isHip\w*|"
    r"ASCIFY_INCLUDE_IN_(?:HIP|ASCIFY)_SDK)\b", re.IGNORECASE)
UNIMPLEMENTED_FLAGS = re.compile(
    r"(?<![\w-])--?(?:perl|python|md|csv|doc-format|o-ascify-perl-dir|"
    r"o-python-map-dir|hip-kernel-execution-syntax|cuda-kernel-execution-syntax)"
    r"(?=[\s`=|,)]|$)")
MARKDOWN_LINK = re.compile(r"\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")


def source_files(root):
    for directory, subdirs, filenames in os.walk(root):
        candidates = sorted(
            name for name in subdirs
            if name not in IGNORED_DIRECTORIES
            and not name.startswith(("build-", "cmake-build-")))
        subdirs[:] = []
        for name in candidates:
            path = Path(directory) / name
            if path.is_symlink():
                yield path
            else:
                subdirs.append(name)
        for name in sorted(filenames):
            if name in (".git", ".DS_Store", "Thumbs.db") or (
                    name.startswith(".env") and name != ".env.example"):
                continue
            yield Path(directory) / name


def check(root):
    root = root.resolve()
    errors = []
    checked = 0
    links = 0

    def problem(path, message, line=None):
        entry = {"path": path, "message": message}
        if line is not None:
            entry["line"] = line
        errors.append(entry)

    for name in REQUIRED:
        if not (root / name).is_file():
            problem(name, "Required project file is missing")

    for path in source_files(root):
        name = path.relative_to(root).as_posix()
        checked += 1
        if path.is_symlink():
            problem(name, "Source files and directories must be self-contained, not symlinks")
            continue
        if name.endswith(ARTIFACT_SUFFIXES):
            problem(name, "Generated or compiled artifact belongs in an ignored work directory")
            continue
        # Images and other intentional documentation assets are not source text.
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".svg", ".ico"):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            problem(name, "Unexpected binary or unreadable file in the source tree")
            continue
        if "\0" in content:
            problem(name, "Unexpected NUL byte in source text")
            continue
        is_product = name in PRODUCT_FILES or any(
            name.startswith(directory + "/") for directory in PRODUCT_DIRECTORIES)
        if is_product:
            for line, text in enumerate(content.splitlines(), 1):
                match = LEGACY_TARGET.search(text)
                if match:
                    problem(name, "Legacy target/product identifier: " + match.group(), line)

        if path.suffix != ".md":
            continue
        # Historical ADRs and third-party attribution may name upstream tools.
        if name in ("README.md", "README.zh-CN.md", "CONTRIBUTING.md",
                    "docs/conversion-reference.md", "docs/user-guide.en.md",
                    "docs/user-guide.zh-CN.md"):
            for line, text in enumerate(content.splitlines(), 1):
                match = UNIMPLEMENTED_FLAGS.search(text)
                if match:
                    problem(name, "User documentation advertises a removed option: " + match.group(), line)
        # Check real inline links outside code examples; do not access the network.
        in_fence = False
        for line, text in enumerate(content.splitlines(), 1):
            if text.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for match in MARKDOWN_LINK.finditer(text):
                target = match.group(1).strip("<>")
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not target:
                    continue
                if target.startswith("/"):
                    problem(name, "Use repository-relative documentation links", line)
                    continue
                destination = (path.parent / target).resolve()
                try:
                    destination.relative_to(root)
                except ValueError:
                    problem(name, "Documentation link escapes the source repository: " + target, line)
                    continue
                if not destination.exists():
                    problem(name, "Broken local documentation link: " + target, line)
                links += 1
    return {"schema": "ascify.repository-check.v1", "passed": not errors,
            "files_checked": checked, "local_links_checked": links, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be a source directory")
    result = check(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
