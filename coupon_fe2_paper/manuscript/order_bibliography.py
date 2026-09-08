"""Mechanical bibliography ordering; keep every existing key and entry."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def expand(path):
    src = path.read_text()
    def child(match):
        name = match.group(1)
        if name == "references":
            return ""
        return expand(HERE / (name if name.endswith(".tex") else name + ".tex"))
    return re.sub(r"\\input\{([^}]+)\}", child, src)


def main():
    path = HERE / "references.tex"
    src = path.read_text()
    blocks = re.findall(
        r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
        src, re.S)
    entries = dict(blocks)
    assert len(entries) == len(blocks), "Duplicate bibliography key"
    order = []
    for group in re.findall(r"\\cite(?:[tp])?\{([^}]+)\}", expand(HERE / "manuscript.tex")):
        for key in group.split(","):
            key = key.strip()
            if key not in order:
                order.append(key)
    assert set(order) == set(entries), (set(order)-set(entries), set(entries)-set(order))
    verified_dois = {
        "Miehe1999": "10.1016/S0045-7825(98)00218-7",
        "GaoNeffRoventaThiel2017": "10.1007/s10659-016-9601-6",
        "Gasser2006": "10.1098/rsif.2005.0073",
        "Tac2022": "10.1016/j.cma.2022.115248",
        "Xu2021": "10.1016/j.jcp.2020.110072",
        "Ghaderi2020": "10.3390/polym12112628",
        "Chmiel2024": "10.2514/6.2024-0250",
    }
    result = [r"\begin{thebibliography}{99}"]
    for key in order:
        body = entries[key].strip()
        if key in verified_dois and r"\doi{" not in body:
            body += "\n" + r"\doi{" + verified_dois[key] + "}"
        result.append(r"\bibitem{" + key + "}\n" + body)
    result.append(r"\end{thebibliography}")
    path.write_text("\n\n".join(result) + "\n")
    print(f"Ordered {len(order)} unchanged keys by first citation.")


if __name__ == "__main__":
    main()
