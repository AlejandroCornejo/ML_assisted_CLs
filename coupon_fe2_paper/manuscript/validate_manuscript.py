"""Read-only source/artifact checks; write only the validation report."""
from collections import Counter
import argparse
from pathlib import Path
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=HERE,
                        help="Directory containing compiled PDFs and logs.")
    build_dir = parser.parse_args().build_dir.resolve()
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
    learned = (HERE/"sections/learned_laws.tex").read_text()
    results = (HERE/"sections/results_and_discussion.tex").read_text()
    # The comparison must follow the feature and guarantee definitions; a
    # particular feature count belongs to the numerical protocol, not here.
    assert learned.index(r"\label{eq:features}") < learned.index(r"\label{tab:tiers}")
    assert results.index(r"\label{app:nonnegative}") < results.index(r"\label{eq:lowerbound}")
    assert learned.index(r"\label{eq:energy_stress_relation}") < learned.index(r"\label{eq:unconstrained_inputs}")
    assert learned.index(r"\label{eq:unconstrained_inputs}") < learned.index(r"\label{eq:unconstrained_energy}")
    assert learned.index(r"\label{eq:unconstrained_energy}") < learned.index(r"\label{eq:features}")
    assert "aresdeparga2026nonlinear" in keys
    main_text = src.split(r"\appendix", 1)[0]
    section_matches = list(re.finditer(r"\\section\{([^}]+)\}", main_text))
    section_titles = [match.group(1) for match in section_matches]
    assert section_titles[0] == "Introduction"
    introduction = main_text[section_matches[0].start():section_matches[1].start()]
    assert r"\label{tab:literature}" not in src, "Superseded literature table must not be included"
    assert r"\label{sec:discussion}" in introduction, "Positioning must be inside Introduction"
    assert r"\begin{tikzpicture}" in introduction, "Figure 1 must use native LaTeX typography"
    assert "Position relative to prior work and limitations" not in section_titles
    assert section_titles[-1] == "Conclusions"
    assert r"\subsection{Physics-augmented neural networks}" in introduction
    assert r"\subsection{Projection-based reduced-order models}" in introduction
    assert "reduced micromechanics" not in main_text.lower()
    assert "Mechanics-informed constitutive learning" not in main_text
    assert "Physics-augmented neural networks and projection-based reduced-order models for anisotropic hyperelasticity" in src
    assert "author list and affiliations to be confirmed" not in src
    assert "This work was conducted while S. Ares de Parga was affiliated with CIMNE." in src
    assert keys == list(dict.fromkeys(cites)), "Bibliography is not in first-citation order"
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
    assert all((HERE/"figures"/f).is_file() for f in figures)
    # The supplement preserves evidence removed from the main reading path.
    supplemental = expand(HERE / "supplementary.tex")
    supplemental_labels = re.findall(r"\\label\{([^}]+)\}", supplemental)
    supplemental_refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", supplemental)
    assert not duplicates(supplemental_labels), duplicates(supplemental_labels)
    assert not set(supplemental_refs)-set(supplemental_labels)
    supplemental_figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", supplemental)
    assert all((HERE/"figures"/f).is_file() for f in supplemental_figures)
    assert "probe_errors.pdf" not in figures
    assert "probe_errors.pdf" in supplemental_figures
    assert r"\label{tab:rankone}" in supplemental
    assert r"\label{tab:rankone}" not in src
    assert r"\input{tables/constitutive_probe_errors.tex}" in (HERE/"supplementary.tex").read_text()
    assert "probe" not in (HERE/"tables/constitutive_errors.tex").read_text()
    assert r"\label{sec:material_b}" in src and r"\pendingresult{" in src
    assert src.index(r"\label{sec:coupon}") < src.index(r"\label{sec:material_a_reduction}")
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
    constitutive_audit_path = HERE/"audit/section4_selected_models_20260910.json"
    constitutive_audit = json.loads(constitutive_audit_path.read_text())
    for relative, expected in constitutive_audit["source_sha256"].items():
        assert hashlib.sha256((ROOT.parent/relative).read_bytes()).hexdigest() == expected, relative
    assert constitutive_audit["unit_tests"]["status"] == "passed"
    assert len(constitutive_audit["results"]) == 2
    for result in constitutive_audit["results"].values():
        assert result["gradcheck"] and result["gradgradcheck"]
        assert result["nonnegative_energy_certificate"]["certified"]
    log = (build_dir/"manuscript.log").read_text(errors="replace")
    bad = [line for line in log.splitlines()
           if ("undefined" in line.lower() or "Overfull" in line
               or "multiply defined" in line.lower() or line.startswith("!"))]
    assert not bad, bad
    supplemental_log = (build_dir/"supplementary.log").read_text(errors="replace")
    supplemental_bad = [line for line in supplemental_log.splitlines()
                        if ("undefined" in line.lower() or "Overfull" in line
                            or "multiply defined" in line.lower() or line.startswith("!"))]
    assert not supplemental_bad, supplemental_bad
    supplemental_info = subprocess.check_output(["pdfinfo", str(build_dir/"supplementary.pdf")], text=True)
    supplemental_pages = int(re.search(r"Pages:\s+(\d+)", supplemental_info).group(1))
    info = subprocess.check_output(["pdfinfo", str(build_dir/"manuscript.pdf")], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    report = {
        "status": "passed", "scope": "source consistency, PDF log, input hashes; no simulation or network test",
        "pages": pages, "references": len(keys), "cited_keys": len(set(cites)),
        "main_sections": section_titles,
        "literature_and_positioning_in_introduction": True,
        "pann_prom_framing_and_frontmatter": True,
        "figures": len(re.findall(r"\\begin\{figure\}", src)),
        "native_latex_figures": len(re.findall(r"\\begin\{tikzpicture\}", src)),
        "tables": len(re.findall(r"\\begin\{(?:table|longtable)\}", src)),
        "labels": len(labels), "referenced_labels": len(set(refs)),
        "new_pdf_renames_verified": len(renames),
        "constitutive_audit_sha256": hashlib.sha256(constitutive_audit_path.read_bytes()).hexdigest(),
        "frozen_constitutive_audits_verified": len(constitutive_audit["results"]),
        "source_files": [str(p.relative_to(HERE)) for p in paths],
        "source_sha256": {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "pdf_sha256": hashlib.sha256((build_dir/"manuscript.pdf").read_bytes()).hexdigest(),
        "supplementary_pages": supplemental_pages,
        "supplementary_pdf_sha256": hashlib.sha256((build_dir/"supplementary.pdf").read_bytes()).hexdigest(),
        "supplementary_log_issues": supplemental_bad,
        "material_b_status": "planned; no results",
        "build_dir": str(build_dir),
        "log_issues": bad,
        "remaining_nonfatal_warnings": [line for line in log.splitlines() if "Underfull" in line],
    }
    (HERE/"validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256","source_files")},indent=2))


if __name__ == "__main__":
    main()
