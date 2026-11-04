# Pipeline walkthrough

Next implementation steps:

1. Load Sentinel-1 before/after GRD scenes.
2. Ensure both scenes use the same polarization/orbit geometry where possible.
3. Calibrate/use sigma0 backscatter in dB.
4. Align scenes to a common grid.
5. Create a change raster.
6. Detect candidate inundation.
7. Apply DEM/slope constraints.
8. Remove permanent water.
9. Convert flood raster to polygons.
10. Intersect with OSM buildings, roads and landuse.
11. Calculate affected area/counts.
12. Export GeoTIFF, GeoJSON and CSV.
13. Visualize in a web GIS.
