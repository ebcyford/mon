library(lidR)
library(rgdal)

BASE_PATH <- "D:/capstone/mon"
OUT_PATH <- file.path(BASE_PATH, "data", "neon")

tile_name <- "GRSM_275000_3941000"
chm_file <- file.path(BASE_PATH, "data", "neon", paste0(tile_name, "_CHM.tif"))
chm_raw <- raster(chm_file)

# Gaussian filter to remove noise points
gauss_filter <- focalWeight(chm_raw, c(0.5, 3), "Gauss")
CHM <- focal(chm_raw, w = gauss_filter, na.rm = TRUE, pad = TRUE)

writeRaster(CHM, filename = file.path(OUT_PATH, paste0(tile_name, "_CHM_smooth") ), format = "GTiff", overwrite = TRUE)

# Define window function
win <- function(x) {
  x * 0.1 + 1
}

# Find tree centers
tree_centers <- tree_detection(CHM, algorithm = lmf(ws = win, 
                                                    hmin = 10,
                                                    shape = "circular"))
writeOGR(obj = tree_centers, 
         dsn = OUT_PATH, 
         layer = paste0(tile_name, "_treeCenters"), 
         driver = "ESRI Shapefile")
