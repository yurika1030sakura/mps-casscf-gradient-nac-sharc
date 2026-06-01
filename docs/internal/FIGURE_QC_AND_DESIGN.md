# Figure QC And Design Notes

## Error Scale

Gradient errors in the manuscript table and convergence plots are reported in
mEh/Bohr.  A value of `1e-4` in that column is `1e-7` Eh/Bohr, well below the
0.1 mEh/Bohr reference line used in the figure.

From the current 27-system public benchmark matrix at largest M:

- 27/27 systems have gradient error below 0.1 mEh/Bohr.
- 26/27 systems have gradient error below 0.001 mEh/Bohr.
- 24/27 systems have absolute NAC error below `1e-4` a.u.
- The 6-31G main-text systems have gradient errors from `1.9e-6` to
  `3.8e-4` mEh/Bohr and NAC errors from `3.9e-7` to `1.0e-4` a.u.

For NACs, absolute error is the main plotted quantity.  Relative NAC error is
not meaningful for symmetry-small reference couplings.

## Figure Design

The revised figures use reproducible Matplotlib code rather than generated
bitmap diagrams for numerical data.

- Double-column PDF width is about 7 inches.
- Fonts are set to Arial/Helvetica-compatible sans serif, with embedded
  TrueType fonts in PDFs.
- Line widths and labels follow ACS-style minimum legibility constraints.
- Colors use an Okabe-Ito style colorblind-safe palette.
- Main convergence figure shows six representative systems; the full
  molecule/basis coverage is shown by the high-M heatmap.
