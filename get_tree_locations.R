# This script fetches NEON inventory data using the NEON Science API. Results are filtered to retrieve locations of 
# Red Spruce (Picea rubens) and return relevant columns that can be used to identify currently tagged trees. 
# Code developed by Emily Cyford

library(geoNEON)
library(readr)
library(dplyr)
library(tidyr)
library(neonUtilities)
options(stringsAsFactors = F)

# Query NEON API and load data from the year 2018
raw <- loadByProduct(dpID = 'DP1.10098.001', 
                     site = 'GRSM', 
                     startdate = '2018-01', 
                     enddate = '2018-12')

# The vst_mappingandtagging allows us to precisely locate tagged trees
trees <- raw$vst_mappingandtagging
# Geolocate the trees
trees_geo <- def.calc.geo.os(trees, 'vst_mappingandtagging')

# Get rid of duplicate entries, filter for non null data, and select relevant columns
trees_geo <- distinct(trees_geo) %>%
  filter(!is.na(adjNorthing)) %>%
  select(individualID, 
         taxonID, 
         scientificName,
         date, 
         stemDistance, 
         stemAzimuth, 
         adjEasting, 
         adjNorthing)

# Write the locations of all of the trees for future reference
write.csv(trees_geo,
          file = 'NEON.GRSM.locatedtrees.csv',
          row.names = FALSE)

# Filter for only red spruce trees
spruce <- trees_geo[which(trees_geo$taxonID == 'PIRU'),]

# Write to separate CSV file
write.csv(spruce, 
          'NEON.GRSM.locatedspruce.csv')