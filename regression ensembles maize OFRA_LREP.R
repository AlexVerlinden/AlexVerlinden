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

#read march 2015 legacy data
download("https://www.dropbox.com/s/50jl10nz0lqessu/maize_Y0_legacy_march2015.zip?dl=0", "./Data/maize_Y0_legacy_march2015.zip")
unzip("./Data/maize_Y0_legacy_march2015.zip", exdir="./Data", overwrite=TRUE)
maizeY0= readShapeSpatial(paste(dat_dir,"/legacy_OFRA_LREP_maize.shp", sep=""))
proj4string(maizeY0)=CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
plot(maizeY0, axes=TRUE)


#data frame for all maize sites n=325
m2_2015=as.data.frame(maizeY0)
m2=m2_2015[c("x","y", "Grain_Yiel")]
m2xy=m2[,1:2] #coords only

#+ Data setup --------------------------------------------------------------
#for al samples baseline yield (control yields)

# Project maize coords to grid CRS for n=325 # is projection used by AfSIS
#coordinates(m2xy) <- ~x+y
#projection(m2xy) <- projection(grids)


# Extract gridded variables at maize locations for n =325
#mgrid = extract (grids, m2xy)

# Assemble dataframes for n=325
#Y0=m2$Grain_Yiel
#mdat <- cbind.data.frame(Y0, mgrid)
#mdat <- na.omit(mdat)
#dataset with coordinates
#mdatc=cbind.data.frame(Y0,m2xy, mgrid) 
#mdatc=na.omit(mdatc)

#for LREP data Malawi

mwsite <- read.table(paste(dat_dir, "/Location.csv", sep=""), header=TRUE, sep=",")
mwtrial <- read.table(paste(dat_dir, "/Trial.csv", sep=""), header=TRUE, sep=",")
test= merge(mwsite, mwtrial)
#get duplicate coordinates
dups <- duplicated(test[, c( 'Easting' ,  'Northing')])
#remove duplicate coordinates
maize_LREP=test[!dups,]
coordinates(maize_LREP)= ~Easting+Northing
proj4string(maize_LREP)= CRS('+proj=utm +zone=36 +south +datum=WGS84 +units=m +no_defs')
maize_LREP_laea=spTransform(maize_LREP, CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")) # reproject vector file to LAEA
maize_LREP_laea=as.data.frame(maize_LREP_laea)
maize_LREP= cbind(maize_LREP_laea$x,maize_LREP_laea$y, maize_LREP_laea$Yc/1000)
colnames(maize_LREP)=c("x", "y","Grain_Yiel")

# merge OFRA and LREP

OFRA_LREP=merge(maize_LREP)
# set train/test set randomization seed
seed <- 123456
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# maize control yields train/test split
index<- createDataPartition(mdat$Y0, p = 0.75, list = FALSE, times = 1)
mTrain <- mdat [ index,] # trainingData:'data.frame':  194 obs. of  31 variables
mTest  <- mdat [-index,]
mTrain =na.omit(mTrain)
#Train and test sets with coordinates
mTrainc=mdatc[ index,]
mTrainlocs=mTrainc[,2:3]
mTestc=mdatc[-index,]
mTestlocs= mTestc[,2:3]
#+ Stepwise main effects GLM's <MASS> --------------------------------------
# 5-fold CV  cross validation
step <- trainControl(method = "cv", number = 10)

# GLM maize control yields on 194 training samples
maize.glm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                 family = gaussian, 
                 method = "glmStepAIC",
                 trControl = step)
ycglm.pred <- predict(grids, maize.glm) ## spatial predictions

# Regression trees for training set maize
#reverse partioning

#reverse partitioning regression with rpart on Yc
rt= trainControl(method= "cv", number = 10, repeats= 5)

# Control yield predictions (Yc) using reverese partitioning regression
Yc.rt <- train(log(mTrain$Y0) ~ ., method="rpart", data=mTrain)
ycrt.pred <- predict(grids, Yc.rt)


# Random forests (no tuning default) for training set maize
# out-of-bag predictions
oob <- trainControl(method = "oob")
# Control yield predictions (Yc) using random forests (out of bag)
Yc.rf <- train(log(mTrain$Y0) ~ ., method= "rf" ,
                      data=mTrain, trControl = oob)
ycrf.pred <- predict(grids, Yc.rf)

#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's cross validation
gbm <- trainControl(method = "repeatedcv", number = 10, repeats= 5)

# gbm gradient boosting for training set control yields
Yc.gbm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                 method = "gbm",
                 trControl = gbm)
ycgbm.pred <- predict(grids,Yc.gbm) ## predict train-set



#+ Neural nets <nnet> this will change in future using h2o------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# neural network of control yields Maize
Y0.nn <- train(log(mTrain$Y0) ~ ., data = mTrain,
               method = "nnet",
               trControl = nn)
ycnn.pred <- predict(grids,Y0.nn)


#regression ensembles

#plot ensembles
#largely based on Markus Walsh------------------------------------------
#bring first 5 regression predictions together in a stack
ycpred <- stack(ycglm.pred, ycrt.pred, ycrf.pred, ycgbm.pred, ycnn.pred)
names(ycpred) <- c("ycglm", "ycrt", "ycrf", "ycgbm", "ycnn")
plot(ycpred, axes = F)

# Test set ensemble predictions for Yc

exyc <- extract(ycpred, m2xy) #extract predictions from stack regressions at all points
Ycens <- cbind.data.frame(Y0, exyc) #adds yields to covariates
Ycens <- na.omit(Ycens) # deletes NAs
YcensTest <- Ycens[-index,] # splits dataset to test set using same seed 12345

# do Ensemble control yield predictions (Yc) on test set

# Regularized ensemble weighting on the test set <test>
# 5 fold CV again cross validation to compensate for collinearity
ens <- trainControl(method = "cv", number = 5)
Yc.ens <- train(Y0~ycglm+ycrt+ycrf+ycgbm+ycnn,family=gaussian(link="log"), 
                data = YcensTest, 
                method = "ridge",
                trControl = ens)
Yc.pred=predict(ycpred,Yc.ens)
plot(Yc.pred)
points(m2, cex=0.1, col= "red")
points(mTrainlocs,cex=0.1, col="blue")
# test predictions with observations on the Test set
Ycwgt_extr=extract(Yc.pred, mTestlocs)
plot(mTest$Y0,Ycwgt_extr, xlim=c(0, 7), ylim=c(0,6))
ens=lm(Ycwgt_extr~mTest$Y0+0)
abline(ens, col="red")

#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("AF_results", showWarnings=F)

# Export Gtif's to "./AF_results"
writeRaster(Yc.pred, filename="./AF_results/AF_maizepreds2.tif", datatype="GeoTiff", overwrite=T)

