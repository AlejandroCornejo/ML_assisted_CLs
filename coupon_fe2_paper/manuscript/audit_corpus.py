"""Inventory local primary sources and extract page-preserving text for review.

Extraction is explicitly not recorded as reading or claim verification.
Run from any directory with python3 -B manuscript/audit_corpus.py.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/sares/PapersForCLsPROMsAndPANNs')


def main():
    destination = HERE / 'audit' / 'extracted'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, pdf in enumerate(sorted(SOURCE.glob('*.pdf')), 1):
        identifier = f'{index:02d}_' + re.sub(r'[^a-zA-Z0-9_-]+', '_', pdf.stem)
        target = destination / (identifier + '.txt')
        subprocess.run(['pdftotext', '-layout', str(pdf), str(target)], check=True)
        content = target.read_text()
        pages = content.split('\f')
        if not pages[-1].strip():
            pages.pop()
        info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True, check=True).stdout
        records.append(dict(id=identifier, source=str(pdf), sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                            extracted=str(target.relative_to(HERE)), pages=len(pages), words=len(content.split()),
                            pdfinfo=info, review_status='extracted; verification recorded separately'))
        print(f'{identifier}: {len(pages)} pages, {len(content.split())} words')
    (HERE / 'audit' / 'corpus_inventory.json').write_text(json.dumps(records, indent=2) + '\n')
    print('TOTAL', len(records), 'documents;', sum(r['pages'] for r in records), 'pages;',
          sum(r['words'] for r in records), 'words')


if __name__ == '__main__':
    main()
