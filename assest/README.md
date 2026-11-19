# FrontierDrive paper figures

Figures accompanying *Self-learning at the frontier of competence in autonomous driving*.

| Paper figure | README asset | Original figure |
| --- | --- | --- |
| 1. Learning along a moving capability frontier | [framework.png](framework.png) | `framework.pdf` |
| 2. Learning gains near the capability frontier | [capability-frontier.png](capability-frontier.png) | `fig2_capability_frontier.pdf` |
| 3. Cross-planner performance and component study | [planner-performance.png](planner-performance.png) | `fig3_method_performance.pdf` |
| 4. Frontier migration across curriculum stages | [frontier-migration.png](frontier-migration.png) | `fig4_diffusiondrive_spatial_migration.pdf` |
| 5. Trajectory-level driving behavior | [driving-behavior.webp](driving-behavior.webp) | `fig5_driving_behavior.pdf` |
| 6. Real-vehicle deployment and driving sequences | [real-vehicle.webp](real-vehicle.webp) | `Vehicle_Test.pdf` |

The assets are rendered from the original one-page PDF figures at 2,400 pixels wide, preserving the complete panels, labels, colors, and plotted content. Charts and the framework use PNG; the two photograph-rich figures use high-quality WebP for efficient loading.

Source and exported-file SHA-256 checksums, image dimensions, and figure numbers are recorded in [figure_sources.json](figure_sources.json). The original PDF figures accompany `main.tex` in the manuscript source directory.

## Export recipe

Render a source figure with Poppler:

```bash
pdftoppm -f 1 -singlefile -png -scale-to-x 2400 -scale-to-y -1 \
  /path/to/framework.pdf assest/framework
```

PNG files are optimized losslessly. `driving-behavior.webp` and `real-vehicle.webp` are converted from the rendered PNGs with Pillow using `quality=94` and `method=6`.
