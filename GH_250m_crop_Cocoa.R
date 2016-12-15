# Script for crop distribution  models GH using ensemble regressions
# basis for cropland mask is 13000 point survey for Ghana conducted by AfSIS for Africa
# grids are from Africasoils.net
# field data are collected by GhaSIS 2016 
#script in development to test crop distribution model with glmnet based on presence/absence from crop scout
# Alex Verlinden April 2016 based on M. Walsh and J.Chen
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret", "ROSE","dismo", "SpatialEco", "doParallel")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require (ROSE) # for sample imbalances
require(spatialEco)
require(doParallel)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("GH_data", showWarnings=F)
dat_dir <- "./GH_data"
# download crop types, livestock etc presence/absence locations in the field
download.file("https://www.dropbox.com/s/2kouegkbrgaug8t/GH_crops_2016%20Sept.zip?dl=0",
              "./GH_data/GH_crops_2016%20Sept.zip", mode ="wb")
unzip("./GH_data/GH_crops_2016%20Sept.zip",exdir="./GH_data")
GH_crops <- read.table(paste(dat_dir, "/GH_crops_2016 Sept.csv", sep=""), header=T, sep=",")
GH_crops <- na.omit(GH_crops)

#download grids for GH crops ~200 MB !!!!! and stack in raster
download.file("https://www.dropbox.com/s/nbpi0l4utm32cgw/GH_250_grids.zip?dl=0", "./GH_data/GH_250_grids.zip", mode="wb")
unzip("./GH_data/GH_250_grids.zip", exdir="./GH_data", overwrite=T)
glist <- list.files(path="./GH_data/GH_250_grids", pattern="tif", full.names=T)

#download cover files for GHana
download.file ("https://www.dropbox.com/s/ejho3jo3jn171w6/GH_250m_cov.zip?dl=0",
               "./GH_data/GH_250m_cov.zip", mode="wb" )
unzip("./GH_data/GH_250m_cov.zip", exdir="./GH_data/GH_250m_cov", overwrite=T)
glist2=list.files(path="./GH_data/GH_250m_cov", pattern="tif", full.names = T)
glist=c(glist,glist2)
grid <- stack(glist)

t=scale(grid, center=TRUE,scale=TRUE) # scale all covariates

#cropmask
download.file("https://www.dropbox.com/s/0pb5jtlsd6hghet/GH_crp1mask.zip?dl=0",
              "./GH_data/GH_crp1mask.zip", mode="wb")
unzip("./GH_data/GH_crp1mask.zip", exdir = "./GH_data",overwrite = T)

#+ Data setup for Crops Ghana--------------------------------------------------------------
# Project crop data to grid CRS
ghcrop.proj <- as.data.frame(project(cbind(GH_crops$X_gps_longitude, GH_crops$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ghcrop.proj) <- c("x","y")
coordinates(ghcrop.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ghcrop.proj) <- projection(grid)

# Extract gridded variables for GH crop data observations 
ghcropex <- data.frame(coordinates(ghcrop.proj), extract(t, ghcrop.proj))
ghcropex=  ghcropex[,3:48]#exclude coordinates

#subset only on cropland
GH_cr_agric=GH_crops[GH_crops$crop_pa=="Y",]
# subset CRS
ghag.proj <- as.data.frame(project(cbind(GH_cr_agric$X_gps_longitude, GH_cr_agric$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ghag.proj) <- c("x","y")
coordinates(ghag.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ghag.proj) <- projection(grid)

#subset extract
GH_cr_ex=extract(t, ghag.proj)


###### Regressions for crops

#____________

# now bind crop species column to the covariates
# this has to change with every new crop
#use names (GH_crops) to check crop name
#crop presence
croppresabs=cbind(GH_crops$crop_pa, ghcropex)
colnames(croppresabs)[1]="crop"
croppresabs$crop=as.factor(croppresabs$crop)
prop.table(table(croppresabs$crop))
#for cocoa
cocopresabs=cbind(GH_crops$Cocoa, ghcropex)
cocopresabs=na.omit(cocopresabs)
colnames(cocopresabs)[1]="coco"
cocopresabs$coco=as.factor(cocopresabs$coco)
summary(cocopresabs)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(cocopresabs$coco))

#for cocoa only on cropland
cocoagric=data.frame(GH_cr_agric$Cocoa,GH_cr_ex)
cocoagric=na.omit(cocoagric)
colnames(cocoagric)[1]= "coco1"
cocoagric$coco1=as.factor(cocoagric$coco1)
prop.table(table(cocoagric$coco1))

cropmask=raster("./GH_data/GH_crp1mask.tif")

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#parallel processing
mc <- makeCluster(detectCores())
registerDoParallel(mc)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split
#cocoa
cocoIndex <- createDataPartition(cocopresabs$coco, p = 2/3, list = FALSE, times = 1)
cocoTrain <- cocopresabs[ cocoIndex,]
cocoTest  <- cocopresabs[-cocoIndex,]
cocoTest= na.omit(cocoTest)

#cocoa on cropland only
coco1Index=createDataPartition(cocoagric$coco1, p = 2/3, list = FALSE, times = 1)
coco1Train =cocoagric[coco1Index,]
coco1Test=cocoagric[-coco1Index,]

#____________
#set up data for caret

#cocoa on cropland
objControl <- trainControl(method='cv', number=10, classProbs = T,
                           returnResamp='none', allowParallel = TRUE,
                           summaryFunction = twoClassSummary)
#glmnet using binomial distribution for coco cropland
coco1.glm=train(coco1 ~ ., data=coco1Train, family= "binomial",method="glmnet",
               metric="ROC", trControl=objControl)
confusionMatrix(coco1.glm)
coco1glm.pred=predict(t,coco1.glm, type= "prob")
plot(varImp(coco1.glm,scale=F))

#coco1 rf
coco1.rf=train(coco1 ~ ., data=coco1Train, family= "binomial",method="rf",
                metric="ROC", ntree=501, trControl=objControl)
confusionMatrix(coco1.rf)
coco1rf.pred=predict(t,coco1.rf, type= "prob")
plot(varImp(coco1.rf,scale=F))

#coco gbm
coco1.gbm=train(coco1 ~ ., data=coco1Train,method="gbm",
               metric="ROC", trControl=objControl)
confusionMatrix(coco1.gbm)
coco1gbm.pred=predict(t,coco1.gbm, type= "prob")
plot(varImp(coco1.gbm,scale=F))

#+ Ensemble predictions <glm> <rf>, <gbm>,  -------------------------------
# Ensemble set-up
pred <- stack(1-coco1glm.pred, 
              1-coco1rf.pred, 1-coco1gbm.pred)
names(pred) <- c("cocoglm",
                 "cocorf","cocogbm")
geospred <- extract(pred, ghag.proj)
# presence/absence of coco (present = Y, absent = N)
cocoens <- cbind.data.frame(GH_cr_agric$Cocoa, geospred)
cocoens <- na.omit(cocoens)
cocoensTest <- cocoens[-cocoIndex,] ## replicate previous test set
names(cocoensTest)[1]= "coco"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10, allowParallel = TRUE )

# presence/absence of coco (present = Y, absent = N)
coco.ens <- train(coco ~. , data = cocoensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)

cocoens.pred <- predict(coco.ens, cocoensTest,  type="prob") ## predict test-set
confusionMatrix(coco.ens) ## print validation summaries
coco.test <- cbind(cocoensTest, cocoens.pred)
cocop <- subset(coco.test, coco=="Y", select=c(Y))
cocoa <- subset(coco.test, coco=="N", select=c(Y))
coco.eval <- evaluate(p=cocop[,1], a=cocoa[,1]) ## calculate ROC's on test set <dismo>
coco.eval
plot(coco.eval, 'ROC') ## plot ROC curve
coco.thld <- threshold(coco.eval, 'spec_sens') ## TPR+TNR threshold for classification
cocoens.pred <- predict(pred, coco.ens, type="prob") ## spatial prediction
cocoens.pred=(1-cocoens.pred)*cropmask
plot((1-cocoens.pred)*cropmask, axes=F, main ="coco probability ensemble in cropland")
cocoensmask <- 1-cocoens.pred >coco.thld  #THLD =0.078
cocoensmask= cocoensmask*cropmask
plot(cocoensmask, axes = F, legend = F, main= "Ensemble distribution prediction of coco")
plot(varImp(coco.ens,scale=F))
dir.create("./GH_results", showWarnings=F)
rf=writeRaster(cocoens.pred, filename="./GH_results/GH_cocoagric_2015_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(cocoensmask, filename="./GH_results/GH_cocoagric_2015_mask.tif", format= "GTiff", overwrite=TRUE)
