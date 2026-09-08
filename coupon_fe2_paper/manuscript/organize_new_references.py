"""Identify-to-key map, collision-safe renames, and generated text provenance.

Dry-run by default. --rename changes only the nine explicitly listed PDFs;
requires write permission to the supplied literature directory. --extract
generates page-preserving text inside this manuscript's audit directory.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/sares/PapersForCLsPROMsAndPANNs')
MAPPING = {
    's10659-016-9601-6.pdf': 'GaoNeffRoventaThiel2017',
    '1-s2.0-S0021999120308469-main.pdf': 'Xu2021',
    '1-s2.0-S0045782522003838-main.pdf': 'Tac2022',
    '1-s2.0-S0022509625001887-main.pdf': 'Abdolazizi2025',
    'polymers-12-02628.pdf': 'Ghaderi2020',
    'rsif20050073.pdf': 'Gasser2006',
    'chmiel-et-al-2024-assessment-of-projection-based-model-order-reduction-for-a-benchmark-hypersonic-flow-problem.pdf': 'Chmiel2024',
    '1-s2.0-S0045782598002187-main.pdf': 'Miehe1999',
    'balls.pdf': 'Ciarlet1988',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rename', action='store_true')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    destination = HERE / 'audit' / 'new_sources'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    # Preflight every target before performing any rename.
    for old_name, key in MAPPING.items():
        old, new = ROOT / old_name, ROOT / (key + '.pdf')
        if old.exists() and new.exists():
            raise FileExistsError(f'Refusing collision: {old} -> {new}')
        if not old.exists() and not new.exists():
            raise FileNotFoundError(old)
    for old_name, key in MAPPING.items():
        old, new = ROOT / old_name, ROOT / (key + '.pdf')
        actual = old if old.exists() else new
        digest = hashlib.file_digest(actual.open('rb'), 'sha256').hexdigest()
        info = subprocess.check_output(['pdfinfo', str(actual)], text=True)
        if args.rename and actual == old:
            old.rename(new)
            actual = new
            assert hashlib.file_digest(actual.open('rb'), 'sha256').hexdigest() == digest
        record = dict(key=key, original=str(old), target=str(new), actual=str(actual),
                      sha256=digest, pdfinfo=info,
                      reading_status='identified; reading tracked in REFERENCE_AUDIT.md')
        if args.extract:
            target = destination / (key + '.txt')
            subprocess.run(['pdftotext', '-layout', str(actual), str(target)], check=True)
            record['text'] = str(target.relative_to(HERE))
        records.append(record)
        print(f'{old_name} -> {key}.pdf; present as {actual.name}')
    (destination / 'rename_manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
