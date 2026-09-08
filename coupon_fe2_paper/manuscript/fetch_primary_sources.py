"""Download three missing primary references from public institutional sources."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'audit' / 'additional_sources'
SOURCES = {
    'linden2023': 'https://arxiv.org/pdf/2302.02403',
    'amos2017': 'https://proceedings.mlr.press/v70/amos17b/amos17b.pdf',
    'ball1977': 'https://www.mat.univie.ac.at/~stefanelli/cv/paper12.pdf',
}

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, url in SOURCES.items():
        try:
            path = ROOT / (name + '.pdf')
            if not path.exists():
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    data = response.read()
                if not data.startswith(b'%PDF'):
                    raise ValueError('Response is not a PDF')
                path.write_bytes(data)
            results[name] = dict(url=url, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as error:
            results[name] = dict(url=url, error=str(error))
        print(name, results[name], flush=True)
    (ROOT / 'provenance.json').write_text(json.dumps(results, indent=2) + '\n')
