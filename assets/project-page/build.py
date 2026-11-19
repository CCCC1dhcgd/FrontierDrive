"""Package the research page at a GitHub Pages site root using the standard library."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit


class PageReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for key in ("src", "href"):
            if attrs.get(key):
                self.references.append(attrs[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    destination = args.output.resolve()
    if destination == repo or destination == source or source.is_relative_to(destination):
        raise ValueError("Choose a separate output directory.")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("The output directory must be empty.")
    destination.mkdir(parents=True, exist_ok=True)

    page = (source / "index.html").read_text(encoding="utf-8")
    page = page.replace("../../assest/", "figures/")
    if args.base_url:
        base = args.base_url.rstrip("/") + "/"
        parsed = urlsplit(base)
        if parsed.scheme != "https" or not parsed.netloc or any(c in base for c in '<>"'):
            raise ValueError("Use an HTTPS base URL.")
        page = page.replace(
            'property="og:image" content="figures/framework.png"',
            'property="og:image" content="' + base + 'figures/framework.png"',
        )
        page = page.replace(
            "</head>",
            '  <link rel="canonical" href="' + base + '">\n'
            '  <meta property="og:url" content="' + base + '">\n</head>',
        )
    (destination / "index.html").write_text(page, encoding="utf-8", newline="\n")
    for name in ("styles.css", "app.js"):
        shutil.copy2(source / name, destination / name)
    for folder, original in (("media", source / "media"), ("figures", repo / "assest")):
        target = destination / folder
        target.mkdir(exist_ok=True)
        for asset in sorted(original.iterdir()):
            if asset.suffix.lower() in {".png", ".webp", ".svg"}:
                shutil.copy2(asset, target / asset.name)
    (destination / ".nojekyll").touch()

    parsed_page = PageReferences()
    parsed_page.feed(page)
    if len(parsed_page.ids) != len(set(parsed_page.ids)):
        raise ValueError("Duplicate HTML IDs.")
    references = 0
    for reference in parsed_page.references:
        value = urlsplit(reference)
        if value.scheme or value.netloc:
            continue
        if value.path:
            target = (destination / unquote(value.path)).resolve()
            if not target.is_relative_to(destination) or not target.is_file():
                raise ValueError("Missing or external asset: " + reference)
        elif value.fragment and value.fragment not in parsed_page.ids:
            raise ValueError("Missing anchor: " + reference)
        references += 1
    print("Packaged FrontierDrive: " + str(references) + " local references verified.")


if __name__ == "__main__":
    main()
