# Script for crop distribution  models TZ using glmnet
# basis for cropland mask is the 1 Million point survey for Africa of 2014-2015 conducted by AfSIS for Africa
# grids are from Africasoils.net
# field data are collected by TanSIS in Tanzania 2015 and 2016
#script in development to test crop distribution model with glmnet based on presence/absence from crop scout
# geospatial superlearner imagenet 
#Alex Verlinden December 2016 based on M. Walsh and J.Chen
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret", e1071")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(e1071)
require(doParallel)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("TZ_crops", showWarnings=F)
dat_dir <- "./TZ_crops"
# download crop presence/absence locations
# these are data from 2016 crop scout ODK forms n= 8900
download.file("https://www.dropbox.com/s/jg1zv9tv0pjpj5n/Crop_TZ_DEC_2016.zip?dl=0",
              "./TZ_crops/Crop_TZ_DEC_2016.zip", mode="wb")
unzip("./TZ_crops/Crop_TZ_DEC_2016.zip", exdir=dat_dir, overwrite=T)
# note that 0 and 1 are not ok for Caret for classifications, should be N and Y or similar
ban <- read.csv(paste(dat_dir, "/Crop_TZ_DEC_2016.csv", sep= ""), header=T, sep=",")

#Download all test geosurvey data and select non crop areas
#Download all geosurvey data
download.file("https://www.dropbox.com/s/339k17oic3n3ju6/TZ_geos_012015.csv?dl=0", "./TZ_crops/TZ_geos_012015.csv", mode="wb")
geos <- read.csv(paste(dat_dir, "/TZ_geos_012015.csv", sep=""), header=T, sep=",")
geos <- geos[,1:7]
geos.no= subset.data.frame(geos, geos$CRP=="N") # select non crop areas
geos.no=na.omit(geos.no)

#download grids for TZ  40 MB
download.file("https://www.dropbox.com/s/fwps69p6bl5747t/TZ_grids2.zip?dl=0","./TZ_crops/TZ_grids2.zip",  mode="wb")
unzip("./TZ_crops/TZ_grids2.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup for TZ crops--------------------------------------------------------------
# Project crop data to grid CRS
ban.proj <- as.data.frame(project(cbind(ban$X_gps_longitude, ban$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ban.proj) <- c("x","y")
coordinates(ban.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ban.proj) <- projection(grid)

# Extract gridded variables for TZ data observations cropland only
banex <- data.frame(coordinates(ban.proj), extract(grid, ban.proj))

mc <- makeCluster(detectCores())
registerDoParallel(mc)
banex=  banex[,3:28] #exclude coordinates
# now bind crop species column to the covariates

#for banana in cropland
banpresabs=cbind(ban$Banana, banex)
banpresabs=na.omit(banpresabs)
colnames(banpresabs)[1]="ban"
banpresabs$ban=as.factor(banpresabs$ban)
summary(banpresabs)
prop.table(table(banpresabs$ban))

#download cropmask
dir.create("./TZ_cropmask")
download.file("https://www.dropbox.com/s/nyvzq5a5v6v4io9/TZ_cropmask.zip?dl=0","./TZ_cropmask/TZ_cropmask.zip",  mode="wb")
unzip("./TZ_cropmask/TZ_cropmask.zip", exdir= "./TZ_cropmask", overwrite=T)

crp=raster("./TZ_cropmask/TZ_cropmask.tif")
crp[crp==0]=NA

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split
#banana
banIndex <- createDataPartition(banpresabs$ban, p = 2/3, list = FALSE, times = 1)
banTrain <- banpresabs[ banIndex,]
banTest  <- banpresabs[-banIndex,]
banTest= na.omit(banTest)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(banpresabs$ban)) # shows imbalanced samples
#print structure
print(str(banpresabs))

#____________
#set up data for caret
objControl <- trainControl(method='cv', number=5, classProbs = T,
                           returnResamp='none',summaryFunction = twoClassSummary)
#glmnet using binomial distribution for banana
ban.glm=train(ban ~ ., data=banTrain, family= "binomial",
              method="glmnet",metric="ROC", trControl=objControl)

predictions <- predict(ban.glm, banTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(ban.glm)
#variable importance
plot(varImp(ban.glm,scale=F), main= "Variable Importance GLMnet")


#spatial predictions
banglm.pred <- predict(grid,ban.glm, type="prob") 
banglmnet.pred=1-banglm.pred

#randomforest
ban.rf=train(ban ~ ., data=banTrain, family= "binomial",method="rf",
             metric="ROC", trControl=objControl)
predictions <- predict(ban.rf, banTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(ban.rf)
#spatial predictions
banrf.pred <- predict(grid,ban.rf, type="prob") 
banrf.pred=1-banrf.pred
plot(banrf.pred, main= "Banana prediction, RandomForest")

#gbm
ban.gbm=train(ban ~ ., data=banTrain, 
              #family= "binomial",
              method="gbm",metric="ROC", trControl=objControl)
predictions <- predict(ban.gbm, banTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(ban.gbm)
#spatial predictions
bangbm.pred <- predict(grid,ban.gbm, type="prob") 
bangbm.pred=1-bangbm.pred
plot(bangbm.pred, main= "Banana prediction, GBM")

#neural net
ban.nn=train(ban ~ ., data=banTrain, 
              family= "binomial",
              method="nnet",metric="ROC", trControl=objControl)
predictions <- predict(ban.nn, banTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(ban.nn)
#spatial predictions
bannn.pred <- predict(grid,ban.nn, type="prob") 
bannn.pred=1-bannn.pred
plot(bannn.pred, main= "Banana prediction, NNET")

#+ Ensemble predictions <glm> <rf>, <gbm>, <dnn>  -------------------------------
# Ensemble set-up
pred <- stack(banglmnet.pred, 
              banrf.pred, bangbm.pred, bannn.pred)
names(pred) <- c("banglm",
                 "banrf","bangbm", "bannn")
geospred <- extract(pred, ban.proj)
# presence/absence of banana (present = Y, absent = N)
banens <- cbind.data.frame(ban$Banana, geospred)
banens <- na.omit(banens)
banensTest <- banens[-banIndex,] ## replicate previous test set
names(banensTest)[1]= "Banana"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Banana (present = Y, absent = N)
ban.ens <- train(Banana ~. , data = banensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)

banens.pred <- predict(ban.ens, banensTest,  type="prob") ## predict test-set
confusionMatrix(ban.ens) ## print validation summaries
ban.test <- cbind(banensTest, banens.pred)
banp <- subset(ban.test, Banana=="Y", select=c(Y))
bana <- subset(ban.test, Banana=="N", select=c(Y))
ban.eval <- evaluate(p=banp[,1], a=bana[,1]) ## calculate ROC's on test set <dismo>
ban.eval
plot(ban.eval, 'ROC') ## plot ROC curve
ban.thld <- threshold(ban.eval, 'spec_sens') ## TPR+TNR threshold for classification
banens.pred <- predict(pred, ban.ens, type="prob") ## spatial prediction
plot((1-banens.pred)*crp, axes=F, main ="Banana probability ensemble in cropland")
banensmask <- 1-banens.pred >ban.thld
banensmask2= banensmask*crp
plot(banensmask2, axes = F, legend = F, main= "Ensemble distribution prediction of Banana")
plot(varImp(ban.ens,scale=F))

dir.create("./TZ_results")
rf=writeRaster(1-banens.pred, filename="./TZ_results/TZ_ban_2016_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(banensmask2, filename="./TZ_results/TZ_ban_2016_mask.tif", format= "GTiff", overwrite=TRUE)

#all data
#set up data for caret
objControl <- trainControl(method='cv', number=5, classProbs = T,
                           returnResamp='none',summaryFunction = twoClassSummary)
#glmnet using binomial distribution for banana
ban.glm=train(ban ~ ., data=banpresabs, family= "binomial",
              method="glmnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.glm)
#spatial predictions
banglm.pred <- predict(grid,ban.glm, type="prob") 
banglmnet.pred=1-banglm.pred

#randomforest
ban.rf=train(ban ~ ., data=banpresabs, family= "binomial",method="rf",
             metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.rf)
#spatial predictions
banrf.pred <- predict(grid,ban.rf, type="prob") 
banrf.pred=1-banrf.pred
plot(banrf.pred, main= "Banana prediction, RandomForest")

#gbm
ban.gbm=train(ban ~ ., data=banpresabs, 
              #family= "binomial",
              method="gbm",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.gbm)
#spatial predictions
bangbm.pred <- predict(grid,ban.gbm, type="prob") 
bangbm.pred=1-bangbm.pred
plot(bangbm.pred, main= "Banana prediction, GBM")

#neural net
ban.nn=train(ban ~ ., data=banpresabs, 
             family= "binomial",
             method="nnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.nn)
#spatial predictions
bannn.pred <- predict(grid,ban.nn, type="prob") 
bannn.pred=1-bannn.pred
plot(bannn.pred, main= "Banana prediction, NNET")

#+ Ensemble predictions <glm> <rf>, <gbm>, <dnn>  -------------------------------
# Ensemble set-up
pred <- stack(banglmnet.pred, 
              banrf.pred, bangbm.pred, bannn.pred)
names(pred) <- c("banglm",
                 "banrf","bangbm", "bannn")
geospred <- extract(pred, ban.proj)
# presence/absence of banana (present = Y, absent = N)
banens <- cbind.data.frame(ban$Banana, geospred)
banens <- na.omit(banens)
banensTest <- banens[-banIndex,] ## replicate previous test set
names(banensTest)[1]= "Banana"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Banana (present = Y, absent = N)
ban.ens <- train(Banana ~. , data = banensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
banens.pred <- predict(pred, ban.ens, type="prob") ## spatial prediction
plot((1-banens.pred)*crp, axes=F, main ="Banana probability ensemble in cropland")
banensmask <- 1-banens.pred >ban.thld
banensmask2= banensmask*crp
plot(banensmask2, axes = F, legend = F, main= "Ensemble distribution prediction of Banana")

#final for all data
rf=writeRaster(1-banens.pred, filename="./TZ_results/TZ_ban_2016_ens_all.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(banensmask2, filename="./TZ_results/TZ_ban_2016_mask_all.tif", format= "GTiff", overwrite=TRUE)
