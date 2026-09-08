"""Generate an OCR reading aid; never alter the supplied scanned PDF."""
import concurrent.futures
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/tmp/coupon-ocr-bQuu3X/runtime')
WORK = Path('/tmp/coupon-ocr-bQuu3X/pages')
SOURCE = Path('/home/sares/PapersForCLsPROMsAndPANNs/Ciarlet1988.pdf')
DEST = HERE / 'audit' / 'new_sources' / 'Ciarlet1988_ocr.txt'


def page(number):
    stem = WORK / f'page_{number:03d}'
    target = stem.with_suffix('.txt')
    if not target.exists():
        subprocess.run(['pdftoppm', '-f', str(number), '-l', str(number),
                        '-scale-to', '2400', '-gray', '-png', '-singlefile',
                        str(SOURCE), str(stem)], check=True, capture_output=True)
        env = dict(os.environ, OMP_THREAD_LIMIT='1',
                   LD_LIBRARY_PATH=str(RUNTIME / 'usr/lib/x86_64-linux-gnu'),
                   TESSDATA_PREFIX=str(RUNTIME / 'usr/share/tesseract-ocr/5/tessdata'))
        subprocess.run([str(RUNTIME / 'usr/bin/tesseract'), str(stem.with_suffix('.png')),
                        str(stem), '-l', 'eng', '--psm', '3'],
                       check=True, env=env, capture_output=True)
    print(f'OCR page {number}/125', flush=True)
    return number, target.read_text()


if __name__ == '__main__':
    WORK.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = dict(pool.map(page, range(1, 126)))
    DEST.write_text('\f'.join(f'PDF PAGE {n}\n{result[n]}' for n in sorted(result)))
    print(DEST, flush=True)
