# Script for crop distribution  models GH using ensemble regressions
# basis for cropland mask is 13000 point survey for Ghana conducted by AfSIS for Africa
# grids are from Africasoils.net
# field data are collected by GhaSIS 2016 
#script in development to test crop distribution model with glmnet based on presence/absence from crop scout
# Alex Verlinden April 2016 based on M. Walsh and J.Chen
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret","dismo", "doParallel")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
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

cropmask=raster("./GH_data/GH_crp1mask.tif")

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
#GH_cr_agric=GH_crops[GH_crops$crop_pa=="Y",]

#add random points approx to get the % of crop/non cropland of field points vs country cropland
# create random background points

set.seed=1234

cropmask2=cropmask
cropmask1=(1-cropmask2)
cropmask1[cropmask1==0]=NA
x=as.data.frame (randomPoints(cropmask1, 400, ghag.proj, excludep=TRUE))

ghcrop.proj <- as.data.frame(project(cbind(GH_crops$X_gps_longitude, GH_crops$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ghcrop.proj) <- c("x","y")

mzall.proj=rbind(ghcrop.proj, x)
coordinates(mzall.proj)=c("x", "y")
projection(mzall.proj)=projection(grid)

coordinates(x) <- ~x+y  #convert to Spatial DataFrame
projection(x) <- projection(grid)
GH_ran=extract(t,x) #extract random not crop samples
GH_ran=cbind.data.frame(maize="N",GH_ran)


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
#for maize
mzpresabs=cbind(GH_crops$cereal.maize, ghcropex)
mzpresabs=na.omit(mzpresabs)
colnames(mzpresabs)[1]="maize"
mzpresabs$maize=as.factor(mzpresabs$maize)
summary(mzpresabs)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(mzpresabs$maize))

#all samples + random
mzall=rbind(mzpresabs, GH_ran)

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#parallel processing
mc <- makeCluster(detectCores())
registerDoParallel(mc)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split

#maize all
mzallIndex=createDataPartition(mzall$maize, p=2/3, list=FALSE, times=1)
mzallTrain=mzall[mzallIndex,]
mzallTest=mzall[-mzallIndex,]

#____________
#set up data for caret

#maize on all samples
objControl <- trainControl(method='cv', number=10, classProbs = T,
                           returnResamp='none', allowParallel = TRUE,
                           summaryFunction = twoClassSummary)
#glmnet using binomial distribution for maize
mz.glm=train(maize ~ ., data=mzallTrain, family= "binomial",method="glmnet",
               metric="ROC", trControl=objControl)
confusionMatrix(mz.glm)
mzglm.pred=predict(t,mz.glm, type= "prob")
plot(varImp(mz.glm,scale=F))

#mz rf
mz.rf=train(maize ~ ., data=mzallTrain, family= "binomial",method="rf",
                metric="ROC", ntree=501, trControl=objControl)
confusionMatrix(mz.rf)
mzrf.pred=predict(t,mz.rf, type= "prob")
plot(varImp(mz.rf,scale=F))

#mz gbm
mz.gbm=train(maize ~ ., data=mzallTrain,method="gbm",
               metric="ROC", trControl=objControl)
confusionMatrix(mz.gbm)
mzgbm.pred=predict(t,mz.gbm, type= "prob")
plot(varImp(mz.gbm,scale=F))

#+ Ensemble predictions <glm> <rf>, <gbm>,  -------------------------------
# Ensemble set-up
pred <- stack(1-mzglm.pred, 
              1-mzrf.pred, 1-mzgbm.pred)
names(pred) <- c("mzglm",
                 "mzrf","mzgbm")
geospred <- extract(pred, mzall.proj)
# presence/absence of maize (present = Y, absent = N)
mzens <- cbind.data.frame(mzall$maize, geospred)
mzens <- na.omit(mzens)
mzensTest <- mzens[-mzallIndex,] ## replicate previous test set
names(mzensTest)[1]= "maize"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10, allowParallel = TRUE )

# presence/absence of maize (present = Y, absent = N)
mz.ens <- train(maize ~. , data = mzensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)

mzens.pred <- predict(mz.ens, mzensTest,  type="prob") ## predict test-set
confusionMatrix(mz.ens) ## print validation summaries
mz.test <- cbind(mzensTest, mzens.pred)
mzp <- subset(mz.test, maize=="Y", select=c(Y))
mza <- subset(mz.test, maize=="N", select=c(Y))
mz.eval <- evaluate(p=mzp[,1], a=mza[,1]) ## calculate ROC's on test set <dismo>
mz.eval
plot(mz.eval, 'ROC') ## plot ROC curve
mz.thld <- threshold(mz.eval, 'spec_sens') ## TPR+TNR threshold for classification
mzens.pred <- predict(pred, mz.ens, type="prob") ## spatial prediction
mzens.pred=(1-mzens.pred)*cropmask
par(mai=c(0.85, 0.85, 0.3, 0.1))
plot((mzens.pred)*cropmask, axes=F, main ="maize probability ensemble in cropland")
mzensmask <- mzens.pred >mz.thld  #THLD =0.226
mzensmask= mzensmask*cropmask
plot(mzensmask, axes = F, legend = F, main= "Ensemble distribution prediction of maize")
plot(varImp(mz.ens,scale=F))
dir.create("./GH_results", showWarnings=F)
rf=writeRaster(mzens.pred, filename="./GH_results/GH_maize_2015_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(mzensmask, filename="./GH_results/GH_maize_2015_mask.tif", format= "GTiff", overwrite=TRUE)

