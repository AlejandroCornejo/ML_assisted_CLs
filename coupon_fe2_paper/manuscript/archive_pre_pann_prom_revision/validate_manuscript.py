"""Read-only source/artifact checks; write only the validation report."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    paths = []
    def expand(path):
        paths.append(path)
        text = path.read_text()
        return re.sub(r"\\input\{([^}]+)\}",
                      lambda m: expand(HERE / (m.group(1) if m.group(1).endswith(".tex")
                                               else m.group(1)+".tex")), text)
    src = expand(HERE / "manuscript.tex")
    keys = re.findall(r"\\bibitem\{([^}]+)\}", src)
    cites = [k.strip() for g in re.findall(r"\\cite(?:[tp])?\{([^}]+)\}", src)
             for k in g.split(",")]
    labels = re.findall(r"\\label\{([^}]+)\}", src)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", src)
    duplicates = lambda seq: [key for key, count in Counter(seq).items() if count > 1]
    assert not duplicates(keys), duplicates(keys)
    assert not duplicates(labels), duplicates(labels)
    assert set(cites) == set(keys), (set(cites)-set(keys), set(keys)-set(cites))
    assert not set(refs)-set(labels), set(refs)-set(labels)
    assert "aresdeparga2026nonlinear" in keys
    main_text = src.split(r"\appendix", 1)[0]
    section_matches = list(re.finditer(r"\\section\{([^}]+)\}", main_text))
    section_titles = [match.group(1) for match in section_matches]
    assert section_titles[0] == "Introduction"
    introduction = main_text[section_matches[0].start():section_matches[1].start()]
    assert r"\label{tab:literature}" in introduction, "Literature table must be inside Introduction"
    assert r"\label{sec:discussion}" in introduction, "Positioning must be inside Introduction"
    assert "Position relative to prior work and limitations" not in section_titles
    assert section_titles[-1] == "Conclusions"
    assert keys == list(dict.fromkeys(cites)), "Bibliography is not in first-citation order"
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
    assert all((HERE/"figures"/f).is_file() for f in figures)
    renames = json.loads((HERE/"audit/new_sources/rename_manifest.json").read_text())
    for entry in renames:
        target = Path(entry["target"])
        assert target.stem == entry["key"]
        assert entry["key"] in keys
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]
    methods = json.loads((HERE/"method_figures_manifest.json").read_text())
    assert hashlib.sha256((ROOT/methods["source"]).read_bytes()).hexdigest() == methods["sha256"]
    assert methods["primary_dimension"] == 3
    assert methods["secondary_dimension"] == 36
    assert methods["coordinate_transform_identity_error"] < 1e-10
    assert methods["orthogonality_V_Vbar"] < 1e-10
    log = (HERE/"manuscript.log").read_text(errors="replace")
    bad = [line for line in log.splitlines()
           if ("undefined" in line.lower() or "Overfull" in line
               or "multiply defined" in line.lower() or line.startswith("!"))]
    assert not bad, bad
    info = subprocess.check_output(["pdfinfo", str(HERE/"manuscript.pdf")], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    report = {
        "status": "passed", "scope": "source consistency, PDF log, input hashes; no simulation or network test",
        "pages": pages, "references": len(keys), "cited_keys": len(set(cites)),
        "main_sections": section_titles,
        "literature_and_positioning_in_introduction": True,
        "figures": len(figures), "tables": len(re.findall(r"\\begin\{(?:table|longtable)\}", src)),
        "labels": len(labels), "referenced_labels": len(set(refs)),
        "new_pdf_renames_verified": len(renames),
        "source_files": [str(p.relative_to(HERE)) for p in paths],
        "source_sha256": {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "pdf_sha256": hashlib.sha256((HERE/"manuscript.pdf").read_bytes()).hexdigest(),
        "log_issues": bad,
        "remaining_nonfatal_warnings": [line for line in log.splitlines() if "Underfull" in line],
    }
    (HERE/"validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256","source_files")},indent=2))


if __name__ == "__main__":
    main()
