# The Monongahela Project

Code and documentation for the workflow of the Monongahela Project at Shepherd University.

## Data Preprocessing
Data for this project were collected by the [National Ecological Observatory Network ](https://data.neonscience.org/home). The data products used are [high-resolution orthorectified camera imagery mosaic](https://data.neonscience.org/data-products/DP3.30010.001) and [discrete return LiDAR point cloud](https://data.neonscience.org/data-products/DP1.30003.001) collected in May of 2018 at the [Great Smoky Mountains National Park](https://www.neonscience.org/field-sites/field-sites-map/GRSM).

The area of interest is the tile with lower left corner at 275000 Easting and 3941000 Northing in the UTM Zone 17N.

The LiDAR file was preprocessed in ArcGIS Pro. `LAS Dataset to Raster` tool was used on ground and non-ground points to create the Digital Terrain Model (DTM) and Digital Surface Model (DSM) respectively. The DSM was subtracted from the DTM to create the Canopy Height Model (CHM).

This CHM was then used in R with the [lidR](https://cran.r-project.org/web/packages/lidR/lidR.pdf) package to find tree centers using a dynamic, circular window with a local maxima function. The CHM was first passed through a gaussian filter with sigma = 0.5 and window size 3 to soften elevations. The local maxima function was then applied to the smooth CHM using a window function (`y = 0.025*x + 3`), where x is the value of the raster pixel. The tree centers were saved as a shapefile.

## Building the CNN
In Python, these tree center points as well as the orthomosaic were imported. For each point, a buffer as a function of the "tree height" (as assigned by `lidR`)(`y = 0.025*x + 2`) was created and squared off. Each tree square was extracted from the mosaic raster to create smaller tree "chips". These chips were manually classified as `spruce` or `not_spruce` and loaded into an HDF5 file using [this script](https://github.com/ebcyford/mon/blob/master/tif_to_hdf5.py). Chips were augmented using [this script](https://github.com/ebcyford/mon/blob/master/prep_data.py). The architecture of the CNN was tweaked and [this model](https://github.com/ebcyford/mon/blob/master/train.py) was found to provide the best results.

## Results
### Running the Model on the Same Tile
<p align="center">
  <img src="https://github.com/ebcyford/mon/blob/master/imgs/predictions.png" alt="Result of CNN on orthomosaic"/>
</p>

### Running the Model on a Different Tile
This is the result of running the model on the tile with lower left corner at 275000 Easting and 3942000 Northing in UTM Zone 17N.
<p align="center">
  <img src="https://github.com/ebcyford/mon/blob/master/imgs/predictions2.png" alt="Result of CNN on orthomosaic"/>
</p>

## Acknowledgments
The National Ecological Observatory Network is a program sponsored by the National Science Foundation and operated under cooperative agreement by Battelle Memorial Institute. This material is based in part upon work supported by the National Science Foundation through the NEON Program.

This research project was funded and made possible by the [NASA West Virginia Space Grant Consortium](wvspacegrant.org).
