"""Bundle the current two-page design into self-contained HTML previews.

Run from the repository root: uv run python -m scripts.export_frontend
No uploaded recordings, result payloads or private artifacts are included.
"""
from base64 import b64encode
from html import escape
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'service/static'
DEST = ROOT / 'reports/frontend-export'



def build():
    DEST.mkdir(parents=True, exist_ok=True)
    css = (ROOT / 'styles/tokens.css').read_text() + '\n' + (STATIC / 'workspace.css').read_text()
    for font in (STATIC / 'fonts').glob('*.woff2'):
        css = css.replace('/static/fonts/' + font.name, 'data:font/woff2;base64,' + b64encode(font.read_bytes()).decode())
    common = (STATIC / 'workspace.js').read_text()
    common = common.replace('  const job = currentJob();\n  updateResultLinks(job?.id);', '')
    licenses = '\n\n'.join(p.name + '\n' + p.read_text() for p in sorted((STATIC / 'fonts').glob('*OFL*')))
    head = '\n'.join('<script>' + (STATIC / name).read_text() + '</script>' for name in ['theme.js', 'shell.js'])
    head += '\n<style>' + css + '</style>'
    for source, target in [('index.html', 'fortebank-review.html'), ('evaluation.html', 'evaluation.html')]:
        html = (STATIC / source).read_text().replace('<body data-page=', '<body data-offline="true" data-page=')
        html = re.sub(r'<script[^>]*src="[^"]+"[^>]*></script>\s*', '', html)
        html = re.sub(r'<link[^>]*rel="stylesheet"[^>]*>', '', html)
        html = html.replace('</head>', head + '\n</head>')
        scripts = '<script>' + common + '</script>'
        if source == 'index.html':
            scripts += '<script>' + (STATIC / 'results.js').read_text() + '</script><script>' + (STATIC / 'review.js').read_text() + '</script>'
        else:
            data = (STATIC / 'evaluation.json').read_text().replace('<', '\\u003c')
            scripts += '<script id="export-evaluation-data" type="application/json">' + data + '</script>'
            evaluation = (STATIC / 'evaluation.js').read_text()
            scripts += '<script>' + evaluation + '</script>'
        html = html.replace('href="/review"', 'href="fortebank-review.html"').replace('href="/evaluation"', 'href="evaluation.html"')
        html = html.replace('</body>', '<template id="bundled-font-licenses">' + escape(licenses) + '</template>\n' + scripts + '\n</body>')
        (DEST / target).write_text(html)
    # Keep previously distributed results.html links usable as an alias.
    (DEST / 'results.html').write_text((DEST / 'fortebank-review.html').read_text())
    (DEST / 'README.txt').write_text('''CashLint — DATATHON 26 design preview

Open fortebank-review.html in a browser. Keep the HTML files in one folder.
Review combines file selection, local video preview, findings and evidence.
Evaluation contains the unchanged measured development metrics and recipe control.
results.html is a compatibility alias for Review, not a separate product page.

This is a design export: it does not upload video or run analysis.
No example findings, source recordings or private result payloads are included.
Use the running CashLint service for real processing and evidence inspection.
Fonts, styles, charts, theme and sidebar controls are bundled for offline use.
Font license text is embedded in each file's bundled-font-licenses template.
The 90% accuracy figure is a future target; measured results are labelled separately.
''')
    archive = ROOT / 'reports/fortebank-frontend-html.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for name in ['fortebank-review.html', 'evaluation.html', 'results.html', 'README.txt']:
            bundle.write(DEST / name, name)
    print(f'Exported Review + Evaluation and results compatibility alias to {archive}')


if __name__ == '__main__':
    build()
