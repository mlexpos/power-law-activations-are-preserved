"""Check a locally built copy of the site (tools/build_site_preview.rb) for problems GitHub Pages would
not report: math that kramdown did not convert, leftover Liquid or placeholders, and links or anchors
that point nowhere.

Usage:
    GEM_HOME=$(ls -d ~/.gem/ruby/*) ruby tools/build_site_preview.rb docs /tmp/site
    python3 tools/check_site.py /tmp/site /power-law-activations-are-preserved
"""

import glob
import os
import re
import sys


def main(root, baseurl=""):
    os.chdir(root)
    pages = sorted(glob.glob("**/*.html", recursive=True))
    ids = {p: set(re.findall(r'id="([^"]+)"', open(p).read())) for p in pages}
    problems = 0
    for p in pages:
        s = open(p).read()
        body = re.sub(r"<script.*?</script>|<pre.*?</pre>|<code>.*?</code>", "", s, flags=re.S)
        text = re.sub(r"<[^>]+>", "", body)
        for pat, what in [(r"\$\$", "unconverted $$"), (r"(?<![\\\w])\$(?!\$)[^$\n]{1,40}\$", "single-$ math"),
                          (r"\{:\s*\.", "attribute list shown as text"), (r"\{\{|\{%", "Liquid left over")]:
            for m in re.finditer(pat, text):
                problems += 1
                print(f"{p}: {what}: ...{text[max(0, m.start() - 30):m.end() + 30]!r}")
        for _, url in re.findall(r'(href|src)="([^"]+)"', s):
            if url.startswith(("http", "mailto")):
                continue
            path, _, frag = url.partition("#")
            if baseurl and (path == baseurl or path.startswith(baseurl + "/")):
                path = path[len(baseurl):] or "/"
            if path.startswith("/"):
                tgt = path.lstrip("/") or "index.html"
            else:
                tgt = os.path.normpath(os.path.join(os.path.dirname(p), path)) if path else p
            if not os.path.exists(tgt):
                problems += 1
                print(f"{p}: missing target {url}")
            elif frag and tgt.endswith(".html") and frag not in ids.get(tgt, set()):
                problems += 1
                print(f"{p}: missing anchor {url}")
    print(f"{len(pages)} pages, {problems} problems")
    return problems


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "") else 0)
