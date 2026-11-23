# FrontierDrive project page

A buildless research website for GitHub Pages.

## Preview

From the FrontierDrive repository root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/assets/project-page/`.

## GitHub Pages

The public project page is **[FrontierDrive](https://cccc1dhcgd.github.io/FrontierDrive/)**.

The [GitHub Actions workflow](../../.github/workflows/project-page.yml) publishes updates from `main` whenever the page, paper figures, or workflow changes. It can also be run manually. In the repository's **Settings > Pages**, select **GitHub Actions** as the source.

`build.py` packages the page at the site root, collects its images and videos (including nested media directories), and checks local asset paths and anchors. The original source remains in `assets/project-page/`, with paper figures in `assest/`.

To prepare a fresh deployment directory locally:

```bash
python assets/project-page/build.py --output build/project-page --base-url https://cccc1dhcgd.github.io/FrontierDrive/
```

The page uses local HTML, CSS, JavaScript, SVG, PNG, and WebP assets, with MP4 and WebM videos supported by the packager. It has no package installation or build step.

## Content and design

- `index.html`: research narrative, author list, links, accessible controls, and citation.
- `styles.css`: responsive layout and the mist-blue / warm-rose palette.
- `app.js`: original animated contour artwork, local pointer response, motion controls, navigation, result and scenario selectors, figure viewer, and citation copying.
- `build.py`: portable publication packaging and local-reference verification.
- `media/`: line-art identity and paper-panel crops. Crop coordinates and sources are recorded in `media/sources.json`.

Set the paper URL on the `data-paper-link` anchor in `index.html`. The script recognizes a supplied URL automatically.

The result selectors use the manuscript's reported scores. Stage panels and real-world sequences are crops of the original paper figures. All original full figures are accessible through the figure viewer.

Motion follows the visitor's reduced-motion setting and can be paused in the header. Background drawing stops while the hero is offscreen or the tab is hidden.
