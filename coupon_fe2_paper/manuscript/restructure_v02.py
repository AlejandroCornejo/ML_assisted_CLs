"""Guarded one-time extraction of v0.1 sections; preserve source and PDF."""
from pathlib import Path
import re
import shutil

root = Path(__file__).resolve().parent
path = root / "manuscript.tex"
text = path.read_text()
if "Evidence-based draft v0.1" not in text:
    raise SystemExit("Refusing migration: source is not v0.1.")
archive = root / "archive_v01"
archive.mkdir(exist_ok=True)
for name in ("manuscript.tex", "manuscript.pdf", "REFERENCE_AUDIT.md", "README.md"):
    src, dst = root / name, archive / name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
intro = text.index(r"\section{Introduction}")
fom = text.index(r"\section{Finite-strain")
learned = text.index(r"\section{Learned constitutive")
reduced = text.index(r"\section{Reduced microscopic")
cell = text.index(r"\section{Numerical example I:")
bib = text.index(r"\begin{thebibliography}")
fom_text = text[fom:learned]
learned_text = text[learned:reduced].replace(
    r"\label{sec:learned}",
    "\\label{sec:learned}\n\\input{sections/mechanical_requirements}")
(root / "sections/homogenization.tex").write_text(fom_text)
(root / "sections/learned_laws.tex").write_text(learned_text)
(root / "sections/results_and_discussion.tex").write_text(text[cell:bib])
existing_bib = text[bib:text.index(r"\end{thebibliography}") + len(r"\end{thebibliography}")]
old = (root.parents[1] / "RVE_NeoHookean_Homogenization/pann/anisotropic/PANN_anisotropic_claude.tex").read_text()
items = {m.group(1): m.group(0).strip() for m in re.finditer(
    r"\\bibitem\{([^}]+)\}.*?(?=\\bibitem|\\end\{thebibliography\})", old, re.S)}
keys = [
    "Miehe1999", "Hernandez2014", "Hernandez2020", "Amsallem2012",
    "Lee2020", "Barnett2022", "An2008", "Farhat2014", "Grimberg2021",
    "AresDeParga2023", "Hernandez2024", "Bravo2024", "Aldakheel2023",
    "Liu2022", "AsadFarhat2026", "GaoNeffRoventaThiel2017", "Ciarlet1988",
    "SchroderNeff2003", "Gasser2006", "Linka2023", "Tac2022",
    "Xu2021", "Abdolazizi2025", "Ghaderi2020", "Chmiel2024",
]
extra = "\n\n".join(items[k] for k in keys)
extra += r"""

\bibitem{aresdeparga2026nonlinear}
S. Ares De Parga, R. Tezaur, C.G. Hern\'andez, C. Farhat,
Nonlinear projection-based model order reduction with machine learning regression for closure error modeling in the latent space,
Computer Methods in Applied Mechanics and Engineering 448 (2026) 118443.
\doi{10.1016/j.cma.2025.118443}

\bibitem{Fritzen2016}
F. Fritzen, M. Hodapp, The finite element square reduced (FE$^{2R}$) method with GPU acceleration: towards three-dimensional two-scale simulations,
International Journal for Numerical Methods in Engineering 107 (2016) 853--881.
\doi{10.1002/nme.5188}

\bibitem{Tac2024}
V. Tac, K. Linka, F. Sahli-Costabal, E. Kuhl, A. Buganza Tepole,
Benchmarking physics-informed frameworks for data-driven hyperelasticity,
Computational Mechanics 73 (2024) 49--65.
\doi{10.1007/s00466-023-02355-2}
Consulted local version: arXiv:2301.10714v1, with the title \emph{Benchmarks for physics-informed data-driven hyperelasticity}.

\bibitem{Kalina2024}
K.A. Kalina, J. Brummund, W. Sun, M. K\"astner,
Neural networks meet anisotropic hyperelasticity: A framework based on generalized structure tensors and isotropic tensor functions,
Computer Methods in Applied Mechanics and Engineering 433 (2025) 117725.
\doi{10.1016/j.cma.2024.117725}
Consulted local version: arXiv:2410.03378v1 (2024).
"""
existing_bib = existing_bib.replace(r"\end{thebibliography}", extra + "\n\\end{thebibliography}")
existing_bib = re.sub(
    r"\\bibitem\{linden2023\}.*?(?=\\bibitem)",
    lambda _: r"""\bibitem{linden2023}
L. Linden, D.K. Klein, K.A. Kalina, J. Brummund, O. Weeger, M. K\"astner,
Neural networks meet hyperelasticity: A guide to enforcing physics,
Journal of the Mechanics and Physics of Solids 179 (2023) 105363.
\doi{10.1016/j.jmps.2023.105363}
Consulted full text: arXiv:2302.02403v2.
""", existing_bib, flags=re.S)
existing_bib = re.sub(
    r"\\bibitem\{Abdolazizi2025\}.*?(?=\\bibitem)",
    lambda _: r"""\bibitem{Abdolazizi2025}
K.P. Abdolazizi, R.C. Aydin, C.J. Cyron, K. Linka,
Constitutive Kolmogorov--Arnold networks (CKANs): Combining accuracy and interpretability in data-driven material modeling,
Journal of the Mechanics and Physics of Solids 203 (2025) 106212.
\doi{10.1016/j.jmps.2025.106212}
""", existing_bib, flags=re.S)
(root / "references.tex").write_text(existing_bib + "\n")
new = text[:intro] + r"""
\input{sections/introduction}
\input{sections/homogenization}
\input{sections/reduced_micromechanics}
\input{sections/learned_laws}
\input{sections/results_and_discussion}
\input{references}
\end{document}
"""
new = new.replace("6 September 2026", "7 September 2026")
new = new.replace("Evidence-based draft v0.1; full bibliography audit in progress",
                  "Expanded working draft v0.2; source and implementation audit accompanying the manuscript")
path.write_text(new)
print("Extracted sections;", len(re.findall(r"\\bibitem", existing_bib)), "bibliography entries.")
