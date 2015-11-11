# Script for verifying representative locations for soil samples for Nutrient Omission Trials TZ
# basis for cropland mask is the 1 Million point survey for Africa of 2014-2015 conducted by AfSIS for Africa
# Alex Verlinden November 2015
#+ Required packages
# install.packages(c("downloader","raster","rgdal")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("NOT_samples", showWarnings=F)
dat_dir <- "./NOT_samples"
# download NOT trial locations
# these are trials from TAMASA
#download("https://www.dropbox.com/s/h58lay7iid52odp/Final_NOT.csv?dl=0", "./NOT_samples/Final_NOT.csv", mode="wb")
NOT <- read.table(paste(dat_dir, "/Final_NOT.csv", sep=""), header=T, sep=",")

#download grids for STZ  5 MB
download("https://www.dropbox.com/s/4ijtvd4cb037211/STZ_250m.zip?dl=0","./NOT_samples/STZ_250m.zip")
unzip("./STZ_250.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup --------------------------------------------------------------
# Project test data to grid CRS
NOT.proj <- as.data.frame(project(cbind(NOT$X, NOT$Y), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(NOT.proj) <- c("x","y")
coordinates(NOT.proj) <- ~x+y  #convert to Spatial DataFrame
projection(NOT.proj) <- projection(grid)

# Extract gridded variables for TZ to test data observations 
NOTex <- data.frame(coordinates(NOT.proj), extract(grid, NOT.proj))
NOTex <- na.omit(NOTex)
coordinates(NOTex) <- ~x+y
projection(NOTex) <- projection(grid)

#download predicted cropland for TZ
#note that when you repeat the script, this cropland file will be included in the grid!
download("https://www.dropbox.com/s/d24zfs86xxsbc13/STZ_cropland_250m.tif.zip?dl=0", "./NOT_samples/STZ_cropland_250m.tif.zip", mode="wb")
unzip("./NOT_samples/STZ_cropland.tif.zip", exdir=dat_dir, overwrite=T)
maize=raster("./NOT_samples/STZ_cropland_250m.tif")
#create random points for predicted cropland as background
# for testing we take random samples as background data from STZ cropland mask
set.seed(1234)
n=randomPoints(maize, 1000)
#extract covariates for random points in predicted cropland
nextr=extract(grid,n)
nextr=na.omit(nextr)


#statistical tests
#mahalanobis on traindata

#mahalanobis distance on all data of observed cropland
#take only points locations from observed croplands (only x y)
Maizeex <- data.frame(coordinates(NOT.proj), extract(grid, NOT.proj))
Maizeex <- na.omit(Maizeex)
crppts=Maizeex[,3:5] #check covariate numbers
# take a subsample for training the Mahalanobis model
samp=sample(nrow(crppts), round(0.75*nrow(crppts)))
traindata=crppts[samp,]
#create a test set from the cropland data
#so what is done here is take the trial locations as "presence" and the 1000 random points as "background"
#this is a test to investigate the representativeness of the sample against the cropland mask using the 7 variables
testdata=crppts[-samp,]
#mahalanobis model on training data
mh=mahal(traindata)
#evaluate the training model with testdata and also the background data
e=evaluate(testdata, nextr, mh)
#plot the AUC
plot(e, 'ROC')
#Plot predictions
pm1=predict(grid,mh, progress= '')
tr=threshold(e, 'spec_sens')
pm2=pm1>tr
plot(pm2)
points(NOT.proj)
pm3=pm2*maize
dir.create("NOT_results", showWarnings=F)
rf=writeRaster(pm3, filename="./NOT_results/STZ_threshold_2015", format= "GTiff", overwrite=TRUE)



