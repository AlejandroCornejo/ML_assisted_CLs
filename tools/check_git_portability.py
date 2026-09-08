"""Check indexed paths for Windows compatibility and blobs over 100 MiB.

Run before committing: python3 tools/check_git_portability.py
Read-only; checks tracked/indexed content, not untracked local datasets.
"""
import re
import subprocess
import sys


RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)


def invalid_windows_path(path):
    return any(
        any(c in '<>:"\\|?*' or ord(c) < 32 for c in part)
        or part.endswith((" ", "."))
        or RESERVED.match(part)
        for part in path.split("/")
    )


def main():
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"])
    issues, objects, names = [], {}, {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = meta.decode().split()
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if stage != "0":
            issues.append(f"Unmerged index entry: {path!r}")
        if invalid_windows_path(path):
            issues.append(f"Windows-invalid path: {path!r}")
        parts = path.split("/")
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            previous = names.setdefault(prefix.casefold(), prefix)
            if previous != prefix:
                issues.append(f"Case collision: {previous!r} / {prefix!r}")
        if mode != "160000":  # Submodule objects need not exist locally.
            objects.setdefault(oid, []).append(path)
    if objects:
        info = subprocess.check_output(
            ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
            input="\n".join(objects) + "\n", text=True,
        )
        for line in info.splitlines():
            oid, size = line.split()
            if not size.isdigit() or int(size) > 100 * 1024**2:
                issues.append(f"Missing object or blob over 100 MiB ({size}): {objects[oid]!r}")
    for issue in sorted(set(issues)):
        print(issue)
    if issues:
        return 1
    print(f"PASS: {len(objects)} unique indexed objects; no Windows-invalid paths, "
          "case collisions, or blobs over 100 MiB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
