#Regression analysis for OFRA maize no fertilizer
rm(list=ls())
# install.packages(c("downloader","raster","rgdal","caret","MASS",randomForest","gbm","nnet")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(caret)
require(MASS)
require(randomForest)
require(gbm)
require(nnet)
library(rgdal)
library (maptools)
library(rgeos)
library(dismo)
library(sp)
library(gstat)
require(rpart)

#rm(list=ls()) #to remove what was previously loaded as there are issues when reloading modified shapefiles, 
#the old one remains and is not overwritten
#getwd()
#load tiffs of standardized variables
# Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("Data", showWarnings=F)
dat_dir <- "./Data"

# download Africa Gtifs (zipped ~ 850 Mb unzipped ~9  Gb) and stack in raster
download("https://www.dropbox.com/s/lp6sisnun4pw6nm/AF_grids_std.zip?dl=0", "./Data/AF_grids_std.zip", mode="wb")
unzip("./Data/AF_grids_std.zip", exdir="./Data", overwrite=T)
glist <- list.files(path="./Data", pattern="tif", full.names=T)
#stack grids 29 layers
grids <- stack(glist)

#read legacy data maize n=956 for all trials with Y0 or control yield in T/ha
download("https://www.dropbox.com/s/0mh8jvc34yf47iw/Legacy_Maize_Y0.zip?dl=0", "./Data/Legacy_Maize_Y0.zip", mode="wb")
unzip("./Data/Legacy_Maize_Y0.zip", exdir="./Data", overwrite=TRUE)
maize_Y0=readShapeSpatial(paste(dat_dir,"/Legacy_Maize_Y0.shp", sep=""))#close to 1000 samples
proj4string(maize_Y0)=CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
maize_Y0_laea=spTransform(maize_Y0, CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")) # reproject vector file to LAEA
plot(maize_Y0_laea, axes=TRUE)

#data frame for all maize sites n=956
m2=as.data.frame(maize_Y0_laea)
#delete last field
m2=m2[,1:6]
mxy=m2[,1:2] #coords only
#+ Data setup --------------------------------------------------------------
# Project maize coords to grid CRS
coordinates(mxy) <- ~x+y
projection(mxy) <- projection(grids)

# Extract gridded variables at maize locations
mgrid <- extract(grids, mxy)

# Assemble dataframes

Y0 <- m2$Yo
mdat <- cbind.data.frame(Y0, mgrid)
mdat <- na.omit(mdat)
#dataset with coordinates
mdatc=cbind.data.frame(Y0,mxy, mgrid) 
mdatc=na.omit(mdatc)
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Cropland train/test split
index<- createDataPartition(mdatc$Y0, p = 0.75, list = FALSE, times = 1)
mTrain <- mdat [ index,]
mTest  <- mdat [-index,]
#Train and test sets with coordinates
mTrainc=mdatc[ index,]
mTestc=mdatc[-index,]
mTestlocs= mTestc[,2:3]
#+ Stepwise main effects GLM's <MASS> --------------------------------------
# 5-fold CV
step <- trainControl(method = "cv", number = 5)

# GLM all maize
maize.glm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                 family = gaussian, 
                 method = "glmStepAIC",
                 trControl = step)
maizeglm.pred <- predict(grids, maize.glm) ## spatial predictions
x=exp(maizeglm.pred)
x[x>=8]=8
plot (x)
points(m2, cex=0.2, col="red")
locs.train=mTrainc[,2:3]
points(locs.train, cex=0.2, col ="blue")

# Regression trees for all maize
#reverse partioning

#reverse partitioning regression with rpart on Yc
rt= trainControl(method= "cv", number = 10, repeats= 5)

# Control yield predictions (Yc)
Yc.rt <- train(log(mTrainc$Y0) ~ ., method="rpart", data=mTrain)
ycrt <- predict(grids, Yc.rt)
plot(ycrt)
points(m2, cex =0.2, col="red")
points(locs.train, cex= 0.2, col= "blue")


# Random forests (no tuning default) for all maize
# out-of-bag predictions
oob <- trainControl(method = "oob")
# Control yield predictions (Yc)
Yc.rf <- train(log(mTrainc$Y0) ~ ., method= "rf" ,
                      data=mTrain, trControl = oob)
ycrf <- predict(grids, Yc.rf)
plot (ycrf)
points(m2, cex=0.2, col= "red")
points(locs.train, cex= 0.2, col= "blue")
#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# gbm
Yc.gbm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                 method = "gbm",
                 trControl = gbm)
Ycgbm.pred <- predict(grids,Yc.gbm) ## predict train-set
locs.train=mTrainc[,2:3]
points(m2, cex=0.2, col= "red")
points(locs.train,cex=0.2, col="blue")



#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# neural network of control yields Maize
Y0.nn <- train(log(mTrain$Y0) ~ ., data = mTrain,
               method = "nnet",
               trControl = nn)
Yc.nn <- predict(grids,Y0.nn)
points(m2, cex=0.2, col= "red")
points(locs.train,cex=0.2, col="blue")

#regression ensembles
# Test set ensemble predictions for Yc
#largely based on Markus------------------------------------------
#bring first 5 regression predictions together in a stack
ycpred <- stack(maizeglm.pred, ycrt, ycrf, Ycgbm.pred, Yc.nn)
names(ycpred) <- c("ycglm", "ycrt", "ycrf", "ycgbm", "ycnn")
exyc <- extract(ycpred, mxy) #extract predictions from stack regressions at all points
Ycens <- cbind.data.frame(Y0, exyc)
Ycens <- na.omit(Ycens)
YcensTest <- Ycens[-index,]

# do Ensemble control yield predictions (Yc) on test set

# Regularized ensemble weighting on the test set <test>
# 5 fold CV
ens <- trainControl(method = "cv", number = 5)
Yc.ens <- train(Y0~ycglm+ycrt+ycrf+ycgbm+ycnn,family=gaussian(link="log"), 
                data = YcensTest, 
                method = "ridge",
                trControl = ens)
Yc.pred=predict(ycpred,Yc.ens)
plot(Yc.pred)
points(m2, cex=0.1, col= "red")
points(locs.train,cex=0.1, col="blue")
# test predictions with observations
Ycwgt_extr=extract(Yc.pred, mTestlocs)
plot(mTest$Y0,Ycwgt_extr, xlim=c(0, 7), ylim=c(0,6))
ens=lm(Ycwgt_extr~mTest$Y0+0)
abline(ens, col="red")

#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("AF_results", showWarnings=F)

# Export Gtif's to "./AF_results"
writeRaster(Yc.pred, filename="./AF_results/AF_maizepreds.tif", datatype="Geotiff", overwrite=T)

