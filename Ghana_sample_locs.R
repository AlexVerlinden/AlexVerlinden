# Script for extracting representative locations for soil samples for GhaSIS
# basis is the 1 Million point survey for Africa of 2014 conducted by AfSIS for Africa
# Alex Verlinden June 2015
#+ Required packages
# install.packages(c("downloader","raster","rgdal")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("GH_samples", showWarnings=F)
dat_dir <- "./GH_samples"
# download 1MQ Geosurvey data 
# these are crop present hits from the 1 million points survey 2014
download("https://www.dropbox.com/s/9y1fso9g8qks1pp/1MQ_CRP_pos.csv?dl=0", "./GH_samples/1MQ_CRP_pos.csv", mode="wb")
geosv <- read.table(paste(dat_dir, "/1MQ_CRP_pos.csv", sep=""), header=T, sep=",")

#download grids for Ghana
downlaod("https://www.dropbox.com/s/cf75wpbs3z94kty/GH_soil_grids.zip?dl=0","./GH_samples/GH_soil_grids.zip")
unzip("./GH_samples/GH_soil_grids.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup --------------------------------------------------------------
# Project test data to grid CRS
geosv.proj <- as.data.frame(project(cbind(geosv$Lon, geosv$Lat), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geosv.proj) <- c("x","y")
geosv <- cbind(geosv, geosv.proj)
coordinates(geosv) <- ~x+y
projection(geosv) <- projection(grid)

# Extract gridded variables for Ghana to test data observations 1 Million points
gsexv <- data.frame(coordinates(geosv), extract(grid, geosv))
gsexv <- na.omit(gsexv)
coordinates(gsexv) <- ~x+y
projection(gsexv) <- projection(grid)

#write file
GH_locs_LL <- as.data.frame(spTransform(gsexv, CRS("+proj=longlat +datum=WGS84")))
colnames(GH_locs_LL)[1:2] <- c("Lon", "Lat")

# Write potential trial locations these are 1439 points for Ghana where cropland was observed
write.csv(GH_locs_LL, "./GH_samples/GH_locs_LL.csv")

#download predicted cropland for Ghana
#note that when you repeat the script, this cropland file with be included in the grid!
download("https://www.dropbox.com/s/mzyxy4ktbdboo1a/GH_cropland.zip?dl=0", "./GH_samples/GH_cropland.zip", mode="wb")
unzip("./GH_samples/GH_cropland.zip", exdir=dat_dir, overwrite=T)
crop=raster("GH_cropland.tif")
#create random points for predicted cropland as background
# for testing we take random samples as backgroudn data from cropland mask
set.seed(1234)
n=randomPoints(crop, 3000)
#extract covariates for random points in predicted cropland
nextr=extract(grid,n)
nextr=na.omit(nextr)
#statistical tests
#mahalanobis on traindata

#mahalanobis distance on all data of observed cropland
#take only points locations from observed croplands
gsexv <- data.frame(coordinates(geosv), extract(grid, geosv))
gsexv <- na.omit(gsexv)
crppts=gsexv[,3:9]
# take a subsample for training the Mahalanobis model
samp=sample(nrow(crppts), round(0.75*nrow(crppts)))
traindata=crppts[samp,]
#create a test set from the cropland data
#so what is done here is take the 1439 cropland observations as "presence" and the 300 random points as "background"
#this is a test to investigate the representativeness of the sample against the cropland mask using the 7 variables
testdata=crppts[-samp,]
croppoints=gsexv[,1:2]
#mahalaobis model on training data
mh=mahal(traindata)
#evaluate the training model with testdata and also the background data
e=evaluate(testdata, nextr, mh)
#plot the AUC
plot(e, 'ROC')
#Plot predictions
pm2=predict(grid,mh, progress= '')
tr=threshold(e, 'spec_sens')
pm2[pm2<tr]=0
plot(pm2)
rf=writeRaster(pm2, filename="./GH_samples/Ghana_threshold_2015", format= "GTiff", overwrite=TRUE)

