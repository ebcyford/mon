library(lidR)
library(rgdal)
library(progress)

CHM_DIR <- "D:/capstone/neon/output/CHM"
OUT_DIR <- "D:/capstone/neon/output/treeCenters"
files <- dir(CHM_DIR, pattern = ".tif$")

progress <- progress_bar$new(format = "  Finding Trees [:bar] :percent eta: :eta",
                             total = length(files))

main <- function(){
  progress$tick(0)
  Sys.sleep(3)
  
  for (file in files){
    progress$tick()
    Sys.sleep(1/length(files))
    
    tile_name <- gsub("_CHM.tif", "", file)
    file_path <- file.path(CHM_DIR, file)
    
    chm_raw <- raster(file_path)
    
    # Gaussian filter to remove noise points
    gauss_filter <- focalWeight(chm_raw, c(0.5, 3), "Gauss")
    chm_smooth <- focal(chm_raw, w = gauss_filter, na.rm = TRUE, pad = TRUE)
    
    writeRaster(chm_smooth, filename = file.path(CHM_DIR, paste0(tile_name, "_CHM_smooth")), format = "GTiff", overwrite = TRUE)
    
    # Define window function
    win <- function(x) {
      x * 0.1 + 1
    }
    
    # Find tree centers
    tree_centers <- tree_detection(chm_smooth, algorithm = lmf(ws = win, 
                                                               hmin = 5,
                                                               shape = "circular"))
    writeOGR(obj = tree_centers, 
             dsn = OUT_DIR, 
             layer = paste0(tile_name, "_treeCenters"), 
             driver = "ESRI Shapefile")
  }
}

main()


