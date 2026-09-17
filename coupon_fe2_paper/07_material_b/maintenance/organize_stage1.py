"""One bounded, recoverable reorganization; leave operational inputs in place."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ARCHIVED = ("mesh_check_v1", "pilot_v1", "refinement_retry_800", "nonlinear_reference_v1",
            "physical_response_v1", "expanded_plan_reference_v1", "expanded_plan_check_v1")
DIAGNOSTICS = ("compression_diagnostic_v1.json", "reference_verification_v2.json")
RESULTS = ("preflight_decision_v1.json", "expanded_decision_v1.json", "reference_linearity_v1.json",
           "constant_tangent_v1.json", "saved_field_audit_v2.json", "reference_verification_v2_complete.json",
           "physical_response_v2", "nonlinear_response_v1", "expanded_field_audit_v1")
REPORTS = ("PILOT_REPORT.md", "PREFLIGHT_REPORT.md", "NONLINEAR_EXPLORATION.md", "EXPANDED_BOX_REPORT.md")
TESTS = ("test_geometry.py", "test_continuation.py", "test_preflight_summary.py", "test_box_summary.py",
         "test_field_audit.py", "test_reference_linearity.py")
MOVES = ([(p, "archive/"+p) for p in ARCHIVED]
         +[(p, "archive/diagnostics/"+p) for p in DIAGNOSTICS]
         +[(p, "results/"+p) for p in RESULTS]
         +[(p, "reports/"+p) for p in REPORTS]
         +[(p, "tests/"+p) for p in TESTS])
MANIFEST = BASE / "archive/relocation_manifest.json"
POST_RELOCATION_EDITS = {
    "tests/test_geometry.py":
        "Resolve pilot_spec.json from the parent directory after moving the test into tests/.",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files(path):
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "apply", "finalize", "verify"))
    args = parser.parse_args()
    if args.mode in ("finalize", "verify"):
        manifest = json.loads(MANIFEST.read_text())
        for item in manifest["files"]:
            if item["new"] in POST_RELOCATION_EDITS:
                current = digest(BASE / item["new"])
                if args.mode == "finalize":
                    recorded = item.get("post_relocation_sha256")
                    if recorded is not None and recorded != current:
                        raise ValueError("Post-relocation source changed: "+item["new"])
                    item["post_relocation_sha256"] = current
                    item["post_relocation_reason"] = POST_RELOCATION_EDITS[item["new"]]
                elif current != item.get("post_relocation_sha256"):
                    raise ValueError("Post-relocation source changed: "+item["new"])
                continue
            preserved = BASE / item.get("original_navigation_copy", item["new"])
            if digest(preserved) != item["sha256"]:
                raise ValueError("Preserved payload changed: "+str(preserved))
        for name, expected in manifest["unmoved_source_sha256"].items():
            if digest(BASE / name) != expected:
                raise ValueError("Unmoved operational source changed: "+name)
        navigation = {"reports/"+p:digest(BASE / "reports" / p) for p in REPORTS}
        if args.mode == "verify" and manifest.get("navigation_sha256", navigation) != navigation:
            raise ValueError("Navigation changed since layout verification")
        manifest.update(status="verified", navigation_sha256=navigation)
        MANIFEST.write_text(json.dumps(manifest, indent=2)+"\n")
        action = "Finalized" if args.mode == "finalize" else "Verified"
        print(action, len(manifest["files"]), "preserved files and", len(manifest["unmoved_source_sha256"]), "unmoved sources")
        return
    if MANIFEST.exists():
        raise FileExistsError("Existing relocation manifest; do not run cleanup twice")
    records = []
    for old, new in MOVES:
        source, target = BASE / old, BASE / new
        if not source.exists() or target.exists():
            raise ValueError("Missing source or occupied destination: "+old)
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Refuse to move symbolic links")
        for p in files(source):
            relative = p.relative_to(BASE)
            destination = Path(new) / p.relative_to(source) if source.is_dir() else Path(new)
            item = dict(old=str(relative), new=str(destination), bytes=p.stat().st_size, sha256=digest(p))
            if old in REPORTS:
                item["original_navigation_copy"] = "archive/navigation_before/"+old
            records.append(item)
    unmoved = {str(p.relative_to(BASE)):digest(p) for p in BASE.glob("*.py") if p.name not in TESTS}
    unmoved.update({str(p.relative_to(BASE)):digest(p) for p in BASE.glob("*spec.json")})
    manifest = dict(status="planned", moves=[dict(old=a, new=b) for a, b in MOVES], files=records,
        unmoved_source_sha256=unmoved,
        scope="Only designated query outputs, navigation documents, tests and inactive history are relocated. "
              "JSON/NPZ/MDPA/source payloads retain exact bytes. Frozen historical paths are not rewritten. "
              "Navigation originals are copied for reversal; functional run directories remain in place.")
    print(json.dumps(dict(moves=manifest["moves"], preserved_files=len(records)), indent=2))
    if args.mode == "plan":
        return
    MANIFEST.parent.mkdir(parents=True, exist_ok=False)
    originals = BASE / "archive/navigation_before"
    originals.mkdir()
    for name in REPORTS:
        shutil.copy2(BASE / name, originals / name)
    MANIFEST.write_text(json.dumps(manifest, indent=2)+"\n")
    for old, new in MOVES:
        target = BASE / new
        target.parent.mkdir(parents=True, exist_ok=True)
        (BASE / old).rename(target)
    manifest["status"] = "relocated_navigation_pending"
    MANIFEST.write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
