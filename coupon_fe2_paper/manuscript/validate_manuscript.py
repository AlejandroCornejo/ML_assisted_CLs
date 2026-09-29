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
    # The construction overview (native TikZ) opens Section 3, next to the notation it uses (moved 2026-09-27).
    section3 = main_text[section_matches[2].start():section_matches[3].start()]
    assert r"\label{fig:constitutive_pipeline}" in section3 and r"\begin{tikzpicture}" in section3
    assert r"\label{fig:constitutive_pipeline}" not in introduction
    assert "Position relative to prior work and limitations" not in section_titles
    assert section_titles[-1] == "Conclusions"
    assert r"\subsection{Energy-based constitutive learning and polyconvexity}" in introduction
    assert r"\subsection{Directional features and representational expressiveness}" in introduction
    assert r"\subsection{Projection-based reduced-order models}" in introduction
    assert "reduced micromechanics" not in main_text.lower()
    assert "Mechanics-informed constitutive learning" not in main_text
    title = "Polyconvex neural-network surrogates with learned paired directional features for anisotropic hyperelastic homogenization"
    supplement_src = (HERE / "supplementary.tex").read_text()
    assert "\\title{" + title + "}" in src and title in supplement_src
    assert "Option 1" not in src and "Working draft" not in src + supplement_src
    assert "author list and affiliations to be confirmed" not in src
    assert "This work was conducted while S. Ares de Parga was affiliated with CIMNE." in src
    assert len(keys) == len(set(keys)) == len(set(cites)), "Bibliography keys must remain unique and complete"
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
    assert all((HERE/"figures"/f).is_file() for f in figures)
    # The supplement is an internal reserve for reviewer replies; it is not submitted,
    # so the paper must never cite it.
    assert re.search(r"[Ss]upplementary", src) is None, "The paper must not cite the internal supplement"
    # PROM(s) names the method class and HPROMs the three deployed models; cited titles are exempt.
    body = src.split(r"\begin{thebibliography}")[0]
    assert r"\begin{thebibliography}" in src
    assert re.search(r"reduced microscopic|microscopic surrogate|reduced evaluator|intrusive reduced reference"
                     r"|\breduced models?\b", body, re.I) is None, "Use PROMs or HPROMs for the reduced models"
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
    assert r"\label{sec:material_b}" in src
    assert r"\subsection{Multicavity RVE: learning the paired features}" in results
    assert r"\subsection{Single-cavity RVE: constitutive accuracy and FE$^2$ deployment}" in results
    assert r"\label{fig:feature_count_sensitivity}" in results
    assert r"\label{fig:mc_representative_path}" not in results
    assert r"\label{fig:common_rve_fields}" in results
    assert r"\label{fig:mc_rve_m06_paths}" not in results
    assert r"\label{tab:mc_rve_m06}" in results
    assert r"\label{sec:training_protocol}" in results
    assert (results.index(r"\label{fig:common_rve_fields}")
            < results.index(r"\label{sec:training_protocol}")
            < results.index(r"\subsection{Multicavity RVE: learning the paired features}")
            < results.index(r"\label{tab:mc_rve_m06}")
            < results.index(r"\label{fig:mc_beyond_data}")
            < results.index(r"\subsection{Single-cavity RVE: constitutive accuracy and FE$^2$ deployment}")
            < results.index(r"\label{fig:rve}"))
    assert r"\pendingresult{" not in src
    assert r"\subsubsection{Admissibility beyond the data}" "\n" r"\label{sec:mc_guarantees}" in results
    mc_mechanics = json.loads((ROOT / "07_material_b/results/feature_count_analysis_v1"
                               "/m06_mechanics_audit_v1/audit.json").read_text())
    certified = {row["slug"]: row["nonnegative_energy_certificate"]["certified"]
                 for row in mc_mechanics["rows"] if "nonnegative_energy_certificate" in row}
    assert len(mc_mechanics["rows"]) == 15 and len(certified) == 12
    assert all(value == ("learned" in slug) for slug, value in certified.items())
    supplement = (HERE/"supplementary.tex").read_text()
    for text in (results, supplement):
        assert re.search(r"\bFree\b|Regression|earlier 32-feature|unresolved", text) is None
    assert r"\label{app:capacity}" in results and r"\label{sec:mc_guarantees}" in results
    for name in ("capacity_mc", "capacity_sc", "cycle_work", "rank_one", "fe2_comparison"):
        assert (HERE/"tables"/f"{name}.tex").is_file()
    assert r"\input{tables/cycle_work.tex}" in supplement
    assert "Regression" not in (HERE/"tables/fe2_comparison.tex").read_text()
    assert r"\label{sec:confined}" in results and r"\label{fig:confined}" in results
    assert results.index(r"\label{sec:coupon}") < results.index(r"\label{sec:confined}") < results.index(r"\label{sec:conclusions}")
    confined = ROOT / "06_fe2/results/confined_compression_v1"
    for name in ("ICNN", "ICKAN", "Unconstrained"):
        for d, last in (("x", 9), ("y", 8)):
            record = json.loads((confined / f"{name}_{d}_incremental_500.json").read_text())
            if name == "Unconstrained":
                assert record["status"] == "not_converged" and record["n_steps_completed"] == last
            else:
                assert record["status"] == "converged" and record["n_steps_completed"] == 20
                assert 4 <= min(s["iterations"] for s in record["steps"]) and max(s["iterations"] for s in record["steps"]) <= 9
    # Gauss-point rank-one curvature of the repeated runs (declared post-hoc diagnostic).
    audit = json.loads((confined / "rank_one/rank_one_audit.json").read_text())
    assert audit["status"] == "complete" and all(audit["identical_records"].values())
    for d, last, negative in (("x", 9, 1), ("y", 8, 2)):
        steps = audit["cases"][f"Unconstrained_{d}_incremental_500"]["steps"]
        assert len(steps) == last and steps[-1]["negative_points"] == negative
        assert all(s["negative_points"] == 0 for s in steps[:-1])
        assert all(v < 0 for v in steps[-1]["second_difference_MPa"])
    assert audit["constrained_minimum_MPa"] > 194
    assert "at one of the 384 Gauss points in $x$ and at two in $y$" in results
    assert "above 194~MPa at every Gauss point and increment" in results
    assert "single increment" not in results
    beyond = json.loads((ROOT / "07_material_b/results/feature_count_analysis_v1"
                         "/m06_beyond_data_v1/summary.json").read_text())
    assert beyond["status"] == "complete" and len(beyond["rows"]) == 15
    for row in beyond["rows"]:
        search = row["directed_search"]
        if row["model"] == "Unconstrained":
            assert search["nearest"]["negative_verified"] and search["most_negative"]["negative_verified"]
        else:
            assert search["cloud_negative_states"] == 0 and search["most_negative"]["curvature_Pa"] > 0
    assert src.index(r"\label{sec:material_a_reduction}") < src.index(r"\label{sec:coupon}")
    feature_audit_path = ROOT / "07_material_b/results/feature_count_analysis_v1/validation_audit_v1/validation_summary.json"
    feature_decision_path = ROOT / "07_material_b/results/feature_count_analysis_v1/m06_reporting_decision_v1/decision.json"
    feature_evaluation_path = ROOT / "07_material_b/results/feature_count_analysis_v1/m06_independent_evaluation_v1/summary.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    feature_decision = json.loads(feature_decision_path.read_text())
    feature_evaluation = json.loads(feature_evaluation_path.read_text())
    assert len(feature_audit["rows"]) == 84
    assert feature_decision["status"] == "frozen_before_independent_evaluation"
    assert feature_decision["selected_feature_count"] == 6
    assert feature_decision["test_or_path_labels_used_for_selection"] is False
    assert feature_evaluation["status"] == "complete"
    source_path_root = ROOT / "07_material_b/results/common_path_evidence_v1"
    source_path_report = json.loads((source_path_root / "report.json").read_text())
    source_path_evidence = source_path_root / "evidence.npz"
    assert hashlib.sha256(source_path_evidence.read_bytes()).hexdigest() == source_path_report["evidence_sha256"]
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
    sc_campaign = ROOT/"06_pann/results/sc_m06_learned_v1"
    sc_selection_path = sc_campaign/"training/validation_selection.json"
    sc_audit_path = sc_campaign/"independent_audit.json"
    sc_mechanics_path = sc_campaign/"mechanics_audit.json"
    sc_selection = json.loads(sc_selection_path.read_text())
    sc_audit = json.loads(sc_audit_path.read_text())
    sc_mechanics = json.loads(sc_mechanics_path.read_text())
    assert sc_selection["status"] == "frozen_before_test_probe"
    assert sc_selection["test_probe_accessed"] is False
    assert len(sc_selection["selected"]) == len(sc_audit) == 2
    for selected in sc_selection["selected"]:
        checkpoint = Path(selected["checkpoint"])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected["checkpoint_sha256"]
        result = next(value for key, value in sc_audit.items()
                      if ((Path(key) if Path(key).is_absolute() else ROOT/Path(key)).resolve()
                          == checkpoint.resolve()))
        assert result["gradcheck"] and result["gradgradcheck"]
        assert result["nonnegative_energy_certificate"]["certified"]
        assert result["metrics"]["test"]["count"] == 400
    for name in ("ICNN", "ICKAN"):
        checkpoint_record = sc_mechanics["checkpoints"][name]
        assert hashlib.sha256(Path(checkpoint_record["path"]).read_bytes()).hexdigest() == checkpoint_record["sha256"]
        for cloud in sc_mechanics["rank_one"]["clouds"].values():
            assert cloud["models"][name]["states_with_a_negative_sampled_direction"] == 0
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
        "sc_m06_independent_audit_sha256": hashlib.sha256(sc_audit_path.read_bytes()).hexdigest(),
        "frozen_constitutive_audits_verified": len(sc_audit),
        "source_files": [str(p.relative_to(HERE)) for p in paths],
        "source_sha256": {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "pdf_sha256": hashlib.sha256((build_dir/"manuscript.pdf").read_bytes()).hexdigest(),
        "supplementary_pages": supplemental_pages,
        "supplementary_pdf_sha256": hashlib.sha256((build_dir/"supplementary.pdf").read_bytes()).hexdigest(),
        "supplementary_log_issues": supplemental_bad,
        "mc_rve_feature_count_sweep": {
            "validation_fits": len(feature_audit["rows"]),
            "reporting_feature_count": feature_decision["selected_feature_count"],
            "independent_evaluation_status": feature_evaluation["status"],
        },
        "build_dir": str(build_dir),
        "log_issues": bad,
        "remaining_nonfatal_warnings": [line for line in log.splitlines() if "Underfull" in line],
    }
    (HERE/"validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256","source_files")},indent=2))


if __name__ == "__main__":
    main()
