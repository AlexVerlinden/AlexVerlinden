# Script for crop distribution  models NG using glmnet
# basis for cropland mask is the 1 Million point survey for Africa of 2014-2015 conducted by AfSIS for Africa
# grids are from Africasoils.net
# field data are collected by NiSIS- OCP in Nigeria 2016 
#script in development to test crop distribution model with glmnet based on presence/absence from crop scout
# Alex Verlinden April 2016 based on M. Walsh and J.Chen
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret", e1071")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(e1071)
require(spatialEco)
require(GWmodel)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("NG_crops_OCP", showWarnings=F)
dat_dir <- "./NG_crops_OCP"
# download crop presence/absence locations
# these are data from 2016 crop scout ODK forms n= 2379
download.file("https://www.dropbox.com/s/u0wiq8fh87udd58/Crop_scout_2016_all.csv?dl=0", "./NG_crops_OCP/Crop_scout_2016_all.csv", mode="wb")
# note that 0 and 1 are not ok for Caret for classifications, should be N and Y or similar
ngcrop <- read.csv(paste(dat_dir, "/Crop_scout_2016_all.csv", sep= ""), header=T, sep=",")
#for cover classes crop, woodland and settlement n=5000
ngcov=read.csv(paste(dat_dir,"/NG_OCP_COV.csv", sep= ""), header=T, sep=",")
#download grids for NG OCP ~9 MB
download.file("https://www.dropbox.com/s/mhr8b50jkfpvozm/OCP_GRIDS_1k.zip?dl=0","./NG_crops_OCP/OCP_GRIDS_1k.zip",  mode="wb")
unzip("./NG_crops_OCP/OCP_grids_1k.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid,center=TRUE, scale=TRUE)
#+ Data setup for NG cover OCP--------------------------------------------------------------
# Project cover data to grid CRS
ngcov.proj <- as.data.frame(project(cbind(ngcov$Longitude, ngcov$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ngcov.proj) <- c("x","y")
coordinates(ngcov.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ngcov.proj) <- projection(grid)

# Extract gridded variables for NG data observations 
ngcovex <- data.frame(coordinates(ngcov.proj), extract(t, ngcov.proj))
ngcovex=  ngcovex[,3:24]

###### Forest cover
###!!!! note this layer is used to refine cropland predictions
#for woodland presence
wldpresabs=cbind(ngcov$WDL, ngcovex)
wldpresabs=na.omit(wldpresabs)
colnames(wldpresabs)[1]="WLD"
wldpresabs$WLD=as.factor(wldpresabs$WLD)

summary(wldpresabs)
#to test if woodland is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(wldpresabs$WLD))


#+ Split data into train and test sets ------------------------------------
# for woodland
wldIndex <- createDataPartition(wldpresabs$WLD, p = 2/3, list = FALSE, times = 1)
wldTrain <- wldpresabs[ wldIndex,]
wldTest  <- wldpresabs[-wldIndex,]
wldTest= na.omit(wldTest)

#____________
#set up data for caret
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for cropland
wld.glm=train(WLD ~ ., data=wldTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(wld.glm, wldTest[,2:23], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wld.glm)
#variable importance
plot(varImp(wld.glm,scale=F))
wldtest=cbind(wldTest, predictions)
wldp=subset(wldtest, WLD=="Y", select=c(Y) )
wlda=subset(wldtest, WLD=="N", select=c(Y))
wld.eval=evaluate(p=wldp[,1],a=wlda[,1])
wld.eval
plot(wld.eval, 'ROC') ## plot ROC curve
wld.thld <- threshold(wld.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
wldglm.pred <- predict(t,wld.glm, type="prob") 
wldglm.pred=1-wldglm.pred
wldmask=wldglm.pred>wld.thld
plot(wldmask, legend=F, main= "Woodland")

#Random Forest Woodland
wld.rf=train(WLD ~ ., data=wldTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(wld.rf, wldTest[,2:23], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wld.rf)
#variable importance
plot(varImp(wld.rf,scale=F))
wldtest=cbind(wldTest, predictions)
wldp=subset(wldtest, WLD=="Y", select=c(Y) )
wlda=subset(wldtest, WLD=="N", select=c(Y))
wld.eval=evaluate(p=wldp[,1],a=wlda[,1])
wld.eval
plot(wld.eval, 'ROC') ## plot ROC curve
wld.thld <- threshold(wld.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
wldrf.pred <- predict(t,wld.rf, type="prob") 
wldrf.pred=1-wldrf.pred
wldmask=wldrf.pred>wld.thld
plot(wldmask, legend=F, main= "Woodland")

#Gradient Boosting Woodland
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
wld.gbm=train(WLD ~ ., data=wldTrain, method="gbm",metric="Accuracy", 
              trControl=gbm)

predictions <- predict(wld.gbm, wldTest[,2:23], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wld.gbm)
#variable importance
plot(varImp(wld.gbm,scale=F))
wldtest=cbind(wldTest, predictions)
wldp=subset(wldtest, WLD=="Y", select=c(Y) )
wlda=subset(wldtest, WLD=="N", select=c(Y))
wld.eval=evaluate(p=wldp[,1],a=wlda[,1])
wld.eval
plot(wld.eval, 'ROC') ## plot ROC curve
wld.thld <- threshold(wld.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
wldgbm.pred <- predict(t,wld.gbm, type="prob") 
wldgbm.pred=1-wldgbm.pred
wldmask=wldgbm.pred>wld.thld
plot(wldmask, legend=F, main= "Woodland")

#+ Ensemble predictions <glm> <rf>, <gbm>, woodland-------------------------------
# Ensemble set-up
pred <- stack(wldglm.pred, wldrf.pred, wldgbm.pred)
names(pred) <- c("wldglm","wldrf","wldgbm")
geospred <- extract(pred, ngcov.proj)
# presence/absence of woodland (present = Y, absent = N)
wldens <- cbind.data.frame(ngcov$WDL, geospred)
wldens <- na.omit(wldens)
wldensTest <- wldens[-wldIndex,] ## replicate previous test set
names(wldensTest)[1]= "WLD"


# Regularized ensemble weighting on the test set <gbm>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Cropland (CRP, present = Y, absent = N)
wld.ens <- train(WLD ~. , data = wldensTest,
                 #family = "binomial", 
                 method = "gbm",
                 trControl = ens)

wldens.pred <- predict(wld.ens, wldensTest,  type="prob") ## predict test-set
#confusionMatrix ## print validation summaries
confusionMatrix(wld.ens)
wld.test <- cbind(wldensTest, wldens.pred)
wldp <- subset(wld.test, WLD=="Y", select=c(Y))
wlda <- subset(wld.test, WLD=="N", select=c(Y))
wld.eval <- evaluate(p=wldp[,1], a=wlda[,1]) ## calculate ROC's on test set <dismo>
wld.eval
plot(wld.eval, 'ROC') ## plot ROC curve
wld.thld <- threshold(wld.eval, 'spec_sens') ## TPR+TNR threshold for classification
wldens.pred <- predict(pred, wld.ens, type="prob") ## spatial prediction
plot(1-wldens.pred, axes=F)
wldensmask <- 1-wldens.pred > wld.thld
plot(wldensmask, axes = F, legend = F, main= "Woodland Ensemble Prediction")
plot(varImp(wld.ens,scale=F))

rf=writeRaster(1-wldens.pred,filename="./NG_OCP_cover_results/OCP_1km_woodens.tif", format= "GTiff", overwrite = TRUE)

# human settlements

#for buildings presence
hsppresabs=cbind(ngcov$HSP, ngcovex)
hsppresabs=na.omit(hsppresabs)
colnames(hsppresabs)[1]="HSP"
hsppresabs$HSP=as.factor(hsppresabs$HSP)
summary(hsppresabs)
#to test if cover is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(hsppresabs$HSP))

#+ Split data into train and test sets ------------------------------------
# for settlements
hspIndex <- createDataPartition(hsppresabs$HSP, p = 2/3, list = FALSE, times = 1)
hspTrain <- hsppresabs[ hspIndex,]
hspTest  <- hsppresabs[-hspIndex,]
hspTest= na.omit(hspTest)

#____________
#set up data for caret
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for cropland
hsp.glm=train(HSP ~ ., data=hspTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(hsp.glm, hspTest[,2:23], type="prob")
#confusionMatrix on cross validation
confusionMatrix(hsp.glm)
#variable importance
plot(varImp(hsp.glm,scale=F))
hsptest=cbind(hspTest, predictions)
hspp=subset(hsptest, HSP=="Y", select=c(Y) )
hspa=subset(hsptest, HSP=="N", select=c(Y))
hsp.eval=evaluate(p=hspp[,1],a=hspa[,1])
hsp.eval
plot(hsp.eval, 'ROC') ## plot ROC curve
hsp.thld <- threshold(hsp.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
hspglm.pred <- predict(t,hsp.glm, type="prob") 
hspglm.pred=1-hspglm.pred
hspmask=hspglm.pred>hsp.thld
plot(hspmask, legend=F,axes=F, main= "Settlements")

#GBM settlements

hsp.gbm=train(HSP ~ ., data=hspTrain, 
              #family= "binomial",
              method="gbm",metric="Accuracy", trControl=objControl)

predictions <- predict(hsp.gbm, hspTest[,2:23], type="prob")
#confusionMatrix on cross validation
confusionMatrix(hsp.gbm)
#variable importance
plot(varImp(hsp.gbm,scale=F))
hsptest=cbind(hspTest, predictions)
hspp=subset(hsptest, HSP=="Y", select=c(Y) )
hspa=subset(hsptest, HSP=="N", select=c(Y))
hsp.eval=evaluate(p=hspp[,1],a=hspa[,1])
hsp.eval
plot(hsp.eval, 'ROC') ## plot ROC curve
hsp.thld <- threshold(hsp.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
hspgbm.pred <- predict(t,hsp.gbm, type="prob") 
hspgbm.pred=1-hspgbm.pred
hspmask=hspgbm.pred>hsp.thld
plot(hspmask, legend=F,axes=F, main= "Settlements")

#random Forest
hsp.rf=train(HSP ~ ., data=hspTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(hsp.rf, hspTest[,2:23], type="prob")
#confusionMatrix on cross validation
confusionMatrix(hsp.rf)
#variable importance
plot(varImp(hsp.rf,scale=F))
hsptest=cbind(hspTest, predictions)
hspp=subset(hsptest, HSP=="Y", select=c(Y) )
hspa=subset(hsptest, HSP=="N", select=c(Y))
hsp.eval=evaluate(p=hspp[,1],a=hspa[,1])
hsp.eval
plot(hsp.eval, 'ROC') ## plot ROC curve
hsp.thld <- threshold(hsp.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
hsprf.pred <- predict(t,hsp.rf, type="prob") 
hsprf.pred=1-hsprf.pred
hspmask=hsprf.pred>hsp.thld
plot(hspmask, legend=F,axes=F, main= "Settlements")

#Ensemble settlements
#+ Ensemble predictions <glm> <rf>, <gbm>, -------------------------------
# Ensemble set-up
pred <- stack(hspglm.pred, hsprf.pred, hspgbm.pred)
names(pred) <- c("hspglm","hsprf","hspgbm")
geospred <- extract(pred, ngcov.proj)
# presence/absence of maize (present = Y, absent = N)
hspens <- cbind.data.frame(ngcov$HSP, geospred)
hspens <- na.omit(hspens)
hspensTest <- hspens[-wldIndex,] ## replicate previous test set
names(hspensTest)[1]= "HSP"


# Regularized ensemble weighting on the test set <gbm>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of woodland (WLD, present = Y, absent = N)
hsp.ens <- train(HSP ~. , data = hspensTest,
                 #family = "binomial", 
                 method = "gbm",
                 trControl = ens)

hspens.pred <- predict(hsp.ens, hspensTest,  type="prob") ## predict test-set
#confusionMatrix ## print validation summaries
confusionMatrix(hsp.ens)
hsp.test <- cbind(hspensTest, hspens.pred)
hspp <- subset(hsp.test, HSP=="Y", select=c(Y))
hspa <- subset(hsp.test, HSP=="N", select=c(Y))
hsp.eval <- evaluate(p=hspp[,1], a=hspa[,1]) ## calculate ROC's on test set <dismo>
hsp.eval
plot(hsp.eval, 'ROC') ## plot ROC curve
hsp.thld <- threshold(hsp.eval, 'spec_sens') ## TPR+TNR threshold for classification
hspens.pred <- predict(pred, hsp.ens, type="prob") ## spatial prediction
plot(1-hspens.pred, axes=F)
hspens.pred=1-hspens.pred
hspensmask <- 1-hspens.pred > hsp.thld
plot(hspensmask, axes = F, legend = F, main= "Settlement Ensemble Prediction")
plot(varImp(hsp.ens,scale=F))

rf=writeRaster(hspens.pred,filename="./NG_OCP_cover_results/OCP_1km_settlens.tif", 
               format= "GTiff", overwrite = TRUE)

#for crop presence now including woodland and settlement layers
OCP_woodens=raster("./NG_OCP_cover_results/OCP_1km_woodens.tif", exdir=dat_dir, overwrite=T)
OCP_settlens=raster("./NG_OCP_cover_results/OCP_1km_settlens.tif", exdir=dat_dir, overwrite=T)

glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist,OCP_woodens, OCP_settlens)
t=scale(grid,center=TRUE, scale=TRUE)

ngcovex <- data.frame(coordinates(ngcov.proj), extract(t, ngcov.proj))
ngcovex=  ngcovex[,3:26]
covpresabs=cbind(ngcov$CRP, ngcovex)
covppresabs=na.omit(covpresabs)
colnames(covpresabs)[1]="Crop"
covpresabs$Crop=as.factor(covpresabs$Crop)
summary(covpresabs)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(covpresabs$Crop))


###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# for crops
covIndex <- createDataPartition(covpresabs$Crop, p = 2/3, list = FALSE, times = 1)
covTrain <- covpresabs[ covIndex,]
covTest  <- covpresabs[-covIndex,]
covTest= na.omit(covTest)

#____________
#set up data for caret
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for cropland
crop.glm=train(Crop ~ ., data=covTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(crop.glm, covTest[,2:25], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.glm)
#variable importance
plot(varImp(crop.glm,scale=F))
croptest=cbind(covTest, predictions)
cropp=subset(croptest, Crop=="Y", select=c(Y) )
cropa=subset(croptest, Crop=="N", select=c(Y))
crop.eval=evaluate(p=cropp[,1],a=cropa[,1])
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cropglm.pred <- predict(t,crop.glm, type="prob") 
cropglm.pred=1-cropglm.pred
cropmask=cropglm.pred>crop.thld
plot(cropmask, axes=F,legend=F, main= "Cropland")

#randomforest
crop.rf=train(Crop ~ ., data=covTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)
predictions <- predict(crop.rf, covTest[,2:25], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.rf)
plot(varImp(crop.rf,scale=F))
croptest=cbind(covTest, predictions)
cropp=subset(croptest, Crop=="Y", select=c(Y) )
cropa=subset(croptest, Crop=="N", select=c(Y))
crop.eval=evaluate(p=cropp[,1],a=cropa[,1])
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
croprf.pred <- predict(t,crop.rf, type="prob") 
croprf.pred=1-croprf.pred
plot(croprf.pred, main= "Cropland prediction, RandomForest")
cropmask=croprf.pred>crop.thld
plot(cropmask, legend=F, main= "Cropland")

#output for cropland rf
dir.create("NG_OCP_cover_results", showWarnings=F)
rf=writeRaster(cropmask, filename="./NG_OCP_cover_results/NG_OCP_crop_rf.tif", format= "GTiff", overwrite=TRUE)

#gbm on cropland
# using binomial distribution
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
# family= "binomial",
crop.gbm=train(Crop ~ ., data=covTrain,method="gbm",metric="Accuracy", trControl=gbm)

predictions <- predict(crop.gbm, covTest[,2:25], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.gbm)
#variable importance
plot(varImp(crop.gbm,scale=F))

croptest=cbind(covTest, predictions)
cropp=subset(croptest, Crop=="Y", select=c(Y) )
cropa=subset(croptest, Crop=="N", select=c(Y))
crop.eval=evaluate(p=cropp[,1],a=cropa[,1])
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cropgbm.pred <- predict(t,crop.gbm, type="prob") 
cropgbm.pred=1-cropgbm.pred
cropmask=cropgbm.pred>crop.thld
plot(cropmask, legend=F, main = "Cropland GBM")
points(ngcov.proj, cex=0.1)

#deep neural net
tc <- trainControl(method = "cv", number = 10, repeats= 3)
mc <- makeCluster(detectCores())
registerDoParallel(mc)
crop.dnn <- train(Crop ~ ., data=covTrain, 
                  method = "dnn", 
                  trControl = tc,
                  tuneGrid = expand.grid(layer1 = 2:7,
                                         layer2 = 0:3,
                                         layer3 = 0:3,
                                         hidden_dropout = 0,
                                         visible_dropout = 0))
print(crop.dnn)
crop.imp <- varImp(crop.dnn, useModel = FALSE)
plot(crop.imp)
predictions <- predict(crop.dnn, covTest[,2:25], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.dnn)
croptest=cbind(covTest, predictions)
cropp=subset(croptest, Crop=="Y", select=c(Y) )
cropa=subset(croptest, Crop=="N", select=c(Y))
crop.eval=evaluate(p=cropp[,1],a=cropa[,1])
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cropdnn.pred <- predict(t,crop.dnn, type="prob") 
cropdnn.pred=1-cropdnn.pred
cropmask=cropdnn.pred>crop.thld
plot(cropmask, legend=F, main = "Cropland Dnn")
points(ngcrop.proj, cex=0.1)


#+ Ensemble predictions <glm> <rf>, <gbm>,<dnn> -------------------------------
# Ensemble set-up
pred <- stack(cropglm.pred, croprf.pred, cropgbm.pred, cropdnn.pred)
names(pred) <- c("cropglm","croprf","cropgbm", "cropdnn")
geospred <- extract(pred, ngcov.proj)
# presence/absence of maize (present = Y, absent = N)
cropens <- cbind.data.frame(ngcov$CRP, geospred)
cropens <- na.omit(cropens)
cropensTest <- cropens[-covIndex,] ## replicate previous test set
names(cropensTest)[1]= "crop"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Cropland (CRP, present = Y, absent = N)
crop.ens <- train(crop ~. , data = cropensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)

cropens.pred <- predict(crop.ens, cropensTest,  type="prob") ## predict test-set
#confusionMatrix(cropens.pred, cropensTest$crop, "Y") ## print validation summaries
confusionMatrix(crop.ens)
crop.test <- cbind(cropensTest, cropens.pred)
cropp <- subset(crop.test, crop=="Y", select=c(Y))
cropa <- subset(crop.test, crop=="N", select=c(Y))
crop.eval <- evaluate(p=cropp[,1], a=cropa[,1]) ## calculate ROC's on test set <dismo>
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
cropens.pred <- predict(pred, crop.ens, type="prob") ## spatial prediction
plot(1-cropens.pred, axes=F)
cropens.pred=1-cropens.pred
cropensmask <- cropens.pred > crop.thld
plot(cropensmask, axes = F, legend = F, main= "Cropland Ensemble Prediction")
plot(varImp(crop.ens,scale=F))

rf=writeRaster(cropens.pred,filename="./NG_OCP_cover_results/OCP_1km_cropens.tif", format= "GTiff", overwrite = TRUE)
rf=writeRaster(cropensmask,filename="./NG_OCP_cover_results/OCP_1km_cropensmask.tif", format= "GTiff", overwrite = TRUE)


#+ Data setup for NG crops OCP--------------------------------------------------------------
# Project crop data to grid CRS
ngcrop.proj <- as.data.frame(project(cbind(ngcrop$X_gps_longitude, ngcrop$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ngcrop.proj) <- c("x","y")
coordinates(ngcrop.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ngcrop.proj) <- projection(grid)
#for the crop distribution we now load cropland, woodland and settlement predictions
OCP_cropens=raster("./NG_OCP_cover_results/OCP_1km_cropens.tif", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(OCP_cropens, OCP_woodens, OCP_settlens, glist)
t=scale(grid,center=TRUE, scale=TRUE)

# Extract gridded variables for NG data observation after adding new cropland predictions
ngcropex <- data.frame(coordinates(ngcrop.proj), extract(t, ngcrop.proj))
ngcropex=  ngcropex[,3:27]#exclude coordinates
#ngcropex=na.omit(ngcropex)
# now bind crop species column to the covariates
# this has to change with every new crop
#use names (ngcrop) to check crop name
#crop presence
croppresabs=cbind(ngcrop$crop_pa, ngcropex)
colnames(croppresabs)="crop"
croppresabs$crop=as.factor(croppresabs$crop)
prop.table(table(croppresabs$crop))
#for Yam
yampresabs=cbind(ngcrop$Yam, ngcropex)
yampresabs=na.omit(yampresabs)
colnames(yampresabs)[1]="Yam"
yampresabs$Yam=as.factor(yampresabs$Yam)
summary(yampresabs)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(yampresabs$Yam))

#for pigeon pea
pigpresabs=cbind(ngcrop$legume.pigeonpea, ngcropex)
pigpresabs=na.omit(pigpresabs)
colnames(pigpresabs)[1]="pig_pea"
pigpresabs$pig_pea=as.factor(pigpresabs$pig_pea)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(pigpresabs$pig_pea))
# for maize
maizepresabs=cbind(ngcrop$cereal.maize, ngcropex)
maizepresabs=na.omit(maizepresabs)
colnames(maizepresabs)[1]="maize"
maizepresabs$maize=as.factor(maizepresabs$maize)
prop.table(table(maizepresabs$maize))

#for wheat
whtpresabs=cbind(ngcrop$cereal.wheat, ngcropex)
whtpresabs=na.omit(whtpresabs)
colnames(whtpresabs)[1]="wheat"
whtpresabs$wheat=as.factor(whtpresabs$wheat)
prop.table(table(whtpresabs$wheat))
#for Finger Millet
finpresabs=cbind(ngcrop$Finger_millet, ngcropex)
finpresabs=na.omit(finpresabs)
colnames(finpresabs)[1]="Finger_millet"
finpresabs$Finger_millet=as.factor(finpresabs$Finger_millet)
prop.table(table(finpresabs$Finger_millet))

#for Green beans
beanpresabs=cbind(ngcrop$legume.beans, ngcropex)
beanpresabs=na.omit(beanpresabs)
colnames(beanpresabs)[1]="bean"
beanpresabs$bean=as.factor(beanpresabs$bean)
prop.table(table(beanpresabs$bean))
#for cowpea
cowpresabs=cbind(ngcrop$legume.cowpea, ngcropex)
cowpresabs=na.omit(cowpresabs)
colnames(cowpresabs)[1]="cowpea"
cowpresabs$cowpea=as.factor(cowpresabs$cowpea)
prop.table(table(cowpresabs$cowpea))
#for soybean
soypresabs=cbind(ngcrop$legume.soybeans, ngcropex)
soypresabs=na.omit(soypresabs)
colnames(soypresabs)[1]="soybean"
soypresabs$soybean=as.factor(soypresabs$soybean)
prop.table(table(soypresabs$soybean))
#for cassava
caspresabs=cbind(ngcrop$root.cassava, ngcropex)
caspresabs=na.omit(caspresabs)
colnames(caspresabs)[1]="cassava"
caspresabs$cassava=as.factor(caspresabs$cassava)
prop.table(table(caspresabs$cassava))
#for sweet potatoes
spotpresabs=cbind(ngcrop$root.spotatoes, ngcropex)
spotpresabs=na.omit(spotpresabs)
colnames(spotpresabs)[1]="sweet_potatoes"
spotpresabs$sweet_potatoes=as.factor(spotpresabs$sweet_potatoes)
prop.table(table(spotpresabs$sweet_potatoes))
#for irish potatoes
potpresabs=cbind(ngcrop$root.potatoes, ngcropex)
potpresabs=na.omit(potpresabs)
colnames(potpresabs)[1]="irish_potatoes"
potpresabs$irish_potatoes=as.factor(potpresabs$irish_potatoes)
prop.table(table(potpresabs$irish_potatoes))
#for rice
ricepresabs=cbind(ngcrop$cereal.rice, ngcropex)
ricepresabs=na.omit(ricepresabs)
colnames(ricepresabs)[1]="rice"
ricepresabs$rice=as.factor(ricepresabs$rice)
prop.table(table(ricepresabs$rice))
#for sorghum
sgpresabs=cbind(ngcrop$cereal.sorgum, ngcropex)
sgpresabs=na.omit(sgpresabs)
colnames(sgpresabs)[1]="sorghum"
sgpresabs$sorghum=as.factor(sgpresabs$sorghum)
prop.table(table(sgpresabs$sorghum))

#for millet
milpresabs=cbind(ngcrop$cereal.millet, ngcropex)
milpresabs=na.omit(milpresabs)
colnames(milpresabs)[1]="millet"
milpresabs$millet=as.factor(milpresabs$millet)
prop.table(table(milpresabs$millet))

#download cropmask
#download.file("https://www.dropbox.com/s/nyvzq5a5v6v4io9/TZ_cropmask.zip?dl=0","./TZ_crops/TZ_cropmask.zip",  mode="wb")
#unzip("./TZ_crops/TZ_cropmask.zip", exdir=dat_dir, overwrite=T)

cropmask=raster("./NG_crops_OCP/OCP_1km_cropensmask.tif")

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split
#Yam
yamIndex <- createDataPartition(yampresabs$Yam, p = 2/3, list = FALSE, times = 1)
yamTrain <- yampresabs[ yamIndex,]
yamTest  <- yampresabs[-yamIndex,]
yamTest= na.omit(yamTest)

#pigeon pea
pigIndex=createDataPartition(pigpresabs$pig_pea, p = 2/3, list = FALSE, times = 1)
pigTrain <- pigpresabs[ pigIndex,]
pigTest  <- pigpresabs[-pigIndex,]
pigTest= na.omit(pigTest)

#soybean
soyIndex=createDataPartition(soypresabs$soybean, p = 2/3, list = FALSE, times = 1)
soyTrain <- soypresabs[ soyIndex,]
soyTest  <- soypresabs[-soyIndex,]
soyTest= na.omit(soyTest)

#maize
maizeIndex=createDataPartition(maizepresabs$maize, p = 2/3, list = FALSE, times = 1)
maizeTrain <- maizepresabs[ maizeIndex,]
maizeTest  <- maizepresabs[-maizeIndex,]
maizeTest= na.omit(maizeTest)

#wheat
whtIndex=createDataPartition(whtpresabs$wheat, p = 2/3, list = FALSE, times = 1)
whtTrain <- whtpresabs[ whtIndex,]
whtTest  <- whtpresabs[-whtIndex,]
whtTest= na.omit(whtTest)

#millet (pearl millet)
milIndex=createDataPartition(milpresabs$millet, p = 2/3, list = FALSE, times = 1)
milTrain <- milpresabs[ milIndex,]
milTest  <- milpresabs[-milIndex,]
milTest= na.omit(milTest)

#Bean
beanIndex=createDataPartition(beanpresabs$bean, p = 2/3, list = FALSE, times = 1)
beanTrain <- beanpresabs[ beanIndex,]
beanTest  <- beanpresabs[-beanIndex,]
beanTest= na.omit(beanTest)
#cassava
casIndex=createDataPartition(caspresabs$cassava, p = 2/3, list = FALSE, times = 1)
casTrain <- caspresabs[ casIndex,]
casTest  <- caspresabs[-casIndex,]
casTest= na.omit(casTest)
#rice
riceIndex=createDataPartition(ricepresabs$rice, p = 2/3, list = FALSE, times = 1)
riceTrain <- ricepresabs[ riceIndex,]
riceTest  <- ricepresabs[-riceIndex,]
riceTest= na.omit(riceTest)
#sorghum
sgIndex=createDataPartition(sgpresabs$sorghum, p = 2/3, list = FALSE, times = 1)
sgTrain <- sgpresabs[ sgIndex,]
sgTest  <- sgpresabs[-sgIndex,]
sgTest= na.omit(sgTest)
#cowpea
cowIndex=createDataPartition(cowpresabs$cowpea, p = 2/3, list = FALSE, times = 1)
cowTrain <- cowpresabs[ cowIndex,]
cowTest  <- cowpresabs[-cowIndex,]
cowTest= na.omit(cowTest)

#____________
#set up data for caret
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for yam
yam.glm=train(Yam ~ ., data=yamTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(yam.glm, yamTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(yam.glm)
#variable importance
plot(varImp(yam.glm,scale=F), main = "variable contribution Yam glmnet")
yamtest=cbind(yamTest, predictions)
yamp=subset(yamtest, Yam=="Y", select=c(Y) )
yama=subset(yamtest, Yam=="N", select=c(Y))
yam.eval=evaluate(p=yamp[,1],a=yama[,1])
yam.eval
plot(yam.eval, 'ROC') ## plot ROC curve
yam.thld <- threshold(yam.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
yamglm.pred <- predict(t,yam.glm, type="prob") 
yamglm.pred=1-yamglm.pred
yammask=yamglm.pred>yam.thld
yammask2=yammask*cropmask
plot(yammask2, legend=F, main= "Yam")

#randomforest
yam.rf=train(Yam ~ ., data=yamTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)
predictions <- predict(yam.rf, yamTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(yam.rf)
plot(varImp(yam.rf,scale=F), main= "variable contribution Yam RForest")
yamtest=cbind(yamTest, predictions)
yamp=subset(yamtest, Yam=="Y", select=c(Y) )
yama=subset(yamtest, Yam=="N", select=c(Y))
yam.eval=evaluate(p=yamp[,1],a=yama[,1])
yam.eval
plot(yam.eval, 'ROC') ## plot ROC curve
yam.thld <- threshold(yam.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
yamrf.pred <- predict(t,yam.rf, type="prob") 
yamrf.pred=1-yamrf.pred
plot(yamrf.pred, main= "Yam prediction, RandomForest")
yammask=yamrf.pred>yam.thld
yammask2=yammask*cropmask
plot(yammask2, axes=F, legend=F, main= "Yam")

#gbm

yam.gbm=train(Yam ~ ., data=yamTrain, 
              #family= "binomial",
              method="gbm",metric="Accuracy", trControl=objControl)
predictions <- predict(yam.gbm, yamTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(yam.gbm)
plot(varImp(yam.gbm,scale=F))
yamtest=cbind(yamTest, predictions)
yamp=subset(yamtest, Yam=="Y", select=c(Y) )
yama=subset(yamtest, Yam=="N", select=c(Y))
yam.eval=evaluate(p=yamp[,1],a=yama[,1])
yam.eval
plot(yam.eval, 'ROC') ## plot ROC curve
yam.thld <- threshold(yam.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
yamgbm.pred <- predict(t,yam.gbm, type="prob") 
yamgbm.pred=1-yamgbm.pred
plot(yamgbm.pred, main= "Yam prediction, GBM")
yammask=yamgbm.pred>yam.thld
yammask2=yammask*cropmask
plot(yammask2, axes=F, legend=F, main= "Yam")

#deep neural net
tc <- trainControl(method = "cv", number = 10, repeats= 3)
mc <- makeCluster(detectCores())
registerDoParallel(mc)
yam.dnn <- train(Yam ~ ., data=yamTrain, 
                method = "dnn", 
                trControl = tc,
                tuneGrid = expand.grid(layer1 = 2:7,
                                       layer2 = 0:3,
                                       layer3 = 0:3,
                                       hidden_dropout = 0,
                                       visible_dropout = 0))
print(yam.dnn)
yam.imp <- varImp(yam.dnn, useModel = FALSE)
plot(yam.imp)
predictions <- predict(yam.dnn, yamTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(yam.dnn)
yamtest=cbind(yamTest, predictions)
yamp=subset(yamtest, Yam=="Y", select=c(Y) )
yama=subset(yamtest, Yam=="N", select=c(Y))
yam.eval=evaluate(p=yamp[,1],a=yama[,1])
yam.eval
plot(yam.eval, 'ROC') ## plot ROC curve
yam.thld <- threshold(yam.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
yamdnn.pred <- predict(t,yam.dnn, type="prob") 
yamdnn.pred=1-yamdnn.pred
yammask=yamdnn.pred>yam.thld
plot(yammask, axes=F, legend=F, main = "Yam Dnn")
points(ngcrop.proj, cex=0.1)

#+ Ensemble predictions <glm> <rf>, <gbm>, <dnn>  -------------------------------
# Ensemble set-up
pred <- stack(yamglm.pred, 
              yamrf.pred, yamgbm.pred, yamdnn.pred)
names(pred) <- c("yamglm",
                 "yamrf","yamgbm", "yamdnn")
geospred <- extract(pred, ngcrop.proj)
# presence/absence of yam (present = Y, absent = N)
yamens <- cbind.data.frame(ngcrop$Yam, geospred)
yamens <- na.omit(yamens)
yamensTest <- yamens[-yamIndex,] ## replicate previous test set
names(yamensTest)[1]= "yam"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of yam (present = Y, absent = N)
yam.ens <- train(yam ~. , data = yamensTest,
                family = "binomial", 
                method = "glmnet",
                trControl = ens)

yamens.pred <- predict(yam.ens, yamensTest,  type="prob") ## predict test-set
confusionMatrix(yam.ens) ## print validation summaries
yam.test <- cbind(yamensTest, yamens.pred)
yamp <- subset(yam.test, yam=="Y", select=c(Y))
yama <- subset(yam.test, yam=="N", select=c(Y))
yam.eval <- evaluate(p=yamp[,1], a=yama[,1]) ## calculate ROC's on test set <dismo>
yam.eval
plot(yam.eval, 'ROC') ## plot ROC curve
yam.thld <- threshold(yam.eval, 'spec_sens') ## TPR+TNR threshold for classification
yamens.pred <- predict(pred, yam.ens, type="prob") ## spatial prediction
plot((1-yamens.pred)*cropmask, axes=F, main ="yam probability ensemble in cropland")
yamensmask <- 1-yamens.pred >yam.thld
yamensmask= yamensmask*cropmask
plot(yamensmask, axes = F, legend = F, main= "Ensemble distribution prediction of Yam")
plot(varImp(yam.ens,scale=F))

rf=writeRaster(yamens.pred, filename="./NG_OCP_results/NG_yam_2015_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(yamensmask, filename="./NG_OCP_results/NG_yam_2015_mask.tif", format= "GTiff", overwrite=TRUE)


#output for yam
dir.create("NG_OCP_results", showWarnings=F)
rf=writeRaster(yammask2, filename="./NG_OCP_results/NG_OCP_yam_glm.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for pigeon pea
pig.glm=train(pig_pea ~ ., data=pigTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(pig.glm, pigTest[,2:18], type="prob")
#confusionMatrix on cross validation
confusionMatrix(pig.glm)
#variable importance
plot(varImp(pig.glm,scale=F))

pigtest=cbind(pigTest, predictions)
pigp=subset(pigtest, pig_pea=="Y", select=c(Y) )
piga=subset(pigtest, pig_pea=="N", select=c(Y))
pig.eval=evaluate(p=pigp[,1],a=piga[,1])
pig.eval
plot(pig.eval, 'ROC') ## plot ROC curve
pig.thld <- threshold(pig.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
pigglm.pred <- predict(grid,pig.glm, type="prob") 
pigglmnet.pred=1-pigglm.pred
pigmask=pigglmnet.pred>pig.thld
pigmask2=pigmask*crp
plot(pigmask2, legend=F, main = "Pigeon Pea")

#write pigeon pea results
rf=writeRaster(pigmask2.pred, filename="./NG_OCP_results/NG_pigeon_pea_2015_glm.tif", format= "GTiff", overwrite=TRUE)

#maize
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')

#glmnet using binomial distribution for maize
mz.glm=train(maize ~ ., data=maizeTrain, family= "binomial",
             method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(mz.glm, maizeTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mz.glm)
#variable importance
plot(varImp(mz.glm,scale=F), main= "variable importance Maize glmnet")

maizetest=cbind(maizeTest, predictions)
maizep=subset(maizetest, maize=="Y", select=c(Y) )
maizea=subset(maizetest, maize=="N", select=c(Y))
mz.eval=evaluate(p=maizep[,1],a=maizea[,1])
mz.eval
plot(mz.eval, 'ROC') ## plot ROC curve
mz.thld <- threshold(mz.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
mzglm.pred <- predict(t,mz.glm, type="prob") 
mzglm.pred=1-mzglm.pred
mzmask=mzglm.pred>mz.thld
#mzmask2=mzmask*cropmask
plot(mzmask, legend=F, main = "Maize")

#write maize results
#rf=writeRaster(mzmask2.pred, filename="./NG_OCP_results/NG_maize_2015_glm.tif", format= "GTiff", overwrite=TRUE)

#random forest using binomial distribution for maize
mz.rf=train(maize ~ ., data=maizeTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(mz.rf, maizeTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mz.rf)
#variable importance
plot(varImp(mz.rf,scale=F), main="variable contribution Maize RForest")

maizetest=cbind(maizeTest, predictions)
maizep=subset(maizetest, maize=="Y", select=c(Y) )
maizea=subset(maizetest, maize=="N", select=c(Y))
mz.eval=evaluate(p=maizep[,1],a=maizea[,1])
mz.eval
plot(mz.eval, 'ROC') ## plot ROC curve
mz.thld <- threshold(mz.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
mzrf.pred <- predict(t,mz.rf, type="prob") 
mzrf.pred=1-mzrf.pred
mzmask=mzrf.pred>mz.thld
#mzmask2=mzmask*cropmask
#plot(mzmask2, legend=F, main = "Maize")
points(ngcrop.proj, cex=0.1)

#extract probabilities for maize
mzptspred=extract(mzrf.pred, ngcrop.proj)
mzptspred=cbind(mzptspred, ngcrop)
write.csv(mzptspred, file="./NG_OCP_results/NG_maize.probs.csv")
#gbm on maize
# using binomial distribution for maize
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
# family= "binomial",
mz.gbm=train(as.factor(maize) ~ ., data=maizeTrain,method="gbm",metric="Accuracy", trControl=gbm)

predictions <- predict(mz.gbm, maizeTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mz.gbm)
#variable importance
plot(varImp(mz.gbm,scale=F))

maizetest=cbind(maizeTest, predictions)
maizep=subset(maizetest, maize=="Y", select=c(Y) )
maizea=subset(maizetest, maize=="N", select=c(Y))
mz.eval=evaluate(p=maizep[,1],a=maizea[,1])
mz.eval
plot(mz.eval, 'ROC') ## plot ROC curve
mz.thld <- threshold(mz.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
mzgbm.pred <- predict(t,mz.gbm, type="prob") 
mzgbm.pred=1-mzgbm.pred
mzmask=mzgbm.pred>mz.thld
#mzmask2=mzmask*NATPA
plot(mzmask2, legend=F, main = "Maize")
points(ngcrop.proj, cex=0.1)

#deep neural net
tc <- trainControl(method = "cv", number = 10, repeats= 3)
mc <- makeCluster(detectCores())
registerDoParallel(mc)
mz.dnn <- train(maize ~ ., data=maizeTrain, 
                  method = "dnn", 
                  trControl = tc,
                  tuneGrid = expand.grid(layer1 = 2:7,
                                         layer2 = 0:3,
                                         layer3 = 0:3,
                                         hidden_dropout = 0,
                                         visible_dropout = 0))
print(mz.dnn)
mz.imp <- varImp(mz.dnn, useModel = FALSE)
plot(mz.imp)
predictions <- predict(mz.dnn, maizeTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mz.dnn)
mztest=cbind(maizeTest, predictions)
mzp=subset(mztest, maize=="Y", select=c(Y) )
mza=subset(mztest, maize=="N", select=c(Y))
mz.eval=evaluate(p=mzp[,1],a=mza[,1])
mz.eval
plot(mz.eval, 'ROC') ## plot ROC curve
mz.thld <- threshold(mz.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
mzdnn.pred <- predict(t,mz.dnn, type="prob") 
mzdnn.pred=1-mzdnn.pred
mzmask=mzdnn.pred>mz.thld
plot(mzmask, legend=F, main = "Maize Dnn")
points(ngcrop.proj, cex=0.1)

#+ Ensemble predictions <glm> <rf>, <gbm>, <dnn>  -------------------------------
# Ensemble set-up
pred <- stack(mzglm.pred, 
              mzrf.pred, mzgbm.pred, mzdnn.pred)
names(pred) <- c("elastic net",
                 "random forest","gradient boosting", "deep neural net")
geospred <- extract(pred, ngcrop.proj)
# presence/absence of maize (present = Y, absent = N)
maizeens <- cbind.data.frame(ngcrop$cereal.maize, geospred)
maizeens <- na.omit(maizeens)
maizeensTest <- maizeens[-maizeIndex,] ## replicate previous test set
names(maizeensTest)[1]= "maize"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of maize ( present = Y, absent = N)
mz.ens <- train(maize ~. , data = maizeensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)

mzens.pred <- predict(mz.ens, maizeensTest,  type="prob") ## predict test-set
confusionMatrix(mz.ens) ## print validation summaries
mz.test <- cbind(maizeensTest, mzens.pred)
mzp <- subset(mz.test, maize=="Y", select=c(Y))
mza <- subset(mz.test, maize=="N", select=c(Y))
mz.eval <- evaluate(p=mzp[,1], a=mza[,1]) ## calculate ROC's on test set <dismo>
mz.eval
plot(mz.eval, 'ROC') ## plot ROC curve
mz.thld <- threshold(mz.eval, 'spec_sens') ## TPR+TNR threshold for classification
mzens.pred <- predict(pred, mz.ens, type="prob") ## spatial prediction
mzens.pred= (1-mzens.pred)
plot(mzens.pred*cropmask, axes=F, main= "Probality map of Maize in cropland")
mzensmask <-mzens.pred > mz.thld
mzensmask= mzensmask*cropmask
plot(mzensmask, axes = F, legend = F, main= "Predicted Maize distribution")
plot(varImp(mz.ens,scale=F))

rf=writeRaster(mzens.pred, filename="./NG_OCP_results/NG_maize_2015_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(mzensmask, filename="./NG_OCP_results/NG_maize_2015_mask.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for beans
bean.glm=train(bean ~ ., data=beanTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(bean.glm, beanTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(bean.glm)
#variable importance
plot(varImp(bean.glm,scale=F))

beantest=cbind(beanTest, predictions)
beanp=subset(beantest, bean=="Y", select=c(Y) )
beana=subset(beantest, bean=="N", select=c(Y))
bean.eval=evaluate(p=beanp[,1],a=beana[,1])
bean.eval
plot(bean.eval, 'ROC') ## plot ROC curve
bean.thld <- threshold(bean.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
beanglm.pred <- predict(grid,bean.glm, type="prob") 
beanglmnet.pred=1-beanglm.pred
beanmask=beanglmnet.pred>bean.thld
beanmask2=beanmask*crp
plot(beanmask2, legend=F, main = "Bean")

#write bean results
rf=writeRaster(beanmask2.pred, filename="./NG_OCP_results/NG_bean_2015_glm.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for cassava
cas.glm=train(cassava ~ ., data=casTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(cas.glm, casTest[,2:18], type="prob")
#confusionMatrix on cross validation
confusionMatrix(cas.glm)
#variable importance
plot(varImp(cas.glm,scale=F))

castest=cbind(casTest, predictions)
casp=subset(castest, cassava=="Y", select=c(Y) )
casa=subset(castest, cassava=="N", select=c(Y))
cas.eval=evaluate(p=casp[,1],a=casa[,1])
cas.eval
plot(cas.eval, 'ROC') ## plot ROC curve
cas.thld <- threshold(cas.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
casglm.pred <- predict(grid,cas.glm, type="prob") 
casglmnet.pred=1-casglm.pred
casmask=casglmnet.pred>cas.thld
casmask2=casmask*crp
plot(casmask2, legend=F, main ="Cassava")

#write cassava results
rf=writeRaster(casmask2.pred, filename="./NG_OCP_results/NG_cassava_2015_glm.tif", format= "GTiff", overwrite=TRUE)


#glmnet using binomial distribution for rice
rice.glm=train(rice ~ ., data=riceTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(rice.glm, riceTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(rice.glm)
#variable importance
plot(varImp(rice.glm,scale=F), main="variable importance Rice glmnet")

ricetest=cbind(riceTest, predictions)
ricep=subset(ricetest, rice=="Y", select=c(Y) )
ricea=subset(ricetest, rice=="N", select=c(Y))
rice.eval=evaluate(p=ricep[,1],a=ricea[,1])
rice.eval
plot(rice.eval, 'ROC') ## plot ROC curve
rice.thld <- threshold(rice.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
riceglm.pred <- predict(t, rice.glm, type="prob") 
riceglm.pred=1-riceglm.pred
ricemask=riceglm.pred>rice.thld
ricemask2=ricemask*cropmask
plot(ricemask2, legend=F, main = "Rice")

#RF using binomial distribution for rice
rice.rf=train(rice ~ ., data=riceTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(rice.rf, riceTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(rice.rf)
#variable importance
plot(varImp(rice.rf,scale=F))

ricetest=cbind(riceTest, predictions)
ricep=subset(ricetest, rice=="Y", select=c(Y) )
ricea=subset(ricetest, rice=="N", select=c(Y))
rice.eval=evaluate(p=ricep[,1],a=ricea[,1])
rice.eval
plot(rice.eval, 'ROC') ## plot ROC curve
rice.thld <- threshold(rice.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
ricerf.pred <- predict(t, rice.rf, type="prob") 
ricerf.pred=1-ricerf.pred
ricemask=ricerf.pred>rice.thld
ricemask2=ricemask*cropmask
plot(ricemask2, legend=F, main = "Rice")

#GBM using binomial distribution for rice
rice.gbm=train(rice ~ ., data=riceTrain, 
               #family= "binomial",
               method="gbm",metric="Accuracy", trControl=objControl)

predictions <- predict(rice.gbm, riceTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(rice.gbm)
#variable importance
plot(varImp(rice.gbm,scale=F))

ricetest=cbind(riceTest, predictions)
ricep=subset(ricetest, rice=="Y", select=c(Y) )
ricea=subset(ricetest, rice=="N", select=c(Y))
rice.eval=evaluate(p=ricep[,1],a=ricea[,1])
rice.eval
plot(rice.eval, 'ROC') ## plot ROC curve
rice.thld <- threshold(rice.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
ricegbm.pred <- predict(t, rice.gbm, type="prob") 
ricegbm.pred=1-ricegbm.pred
ricemask=ricegbm.pred>rice.thld
ricemask2=ricemask*cropmask
plot(ricemask2, legend=F, main = "Rice")

#deep neural net
tc <- trainControl(method = "cv", number = 10, repeats= 3)
mc <- makeCluster(detectCores())
registerDoParallel(mc)
rice.dnn <- train(rice ~ ., data=riceTrain, 
                method = "dnn", 
                trControl = tc,
                tuneGrid = expand.grid(layer1 = 2:7,
                                       layer2 = 0:3,
                                       layer3 = 0:3,
                                       hidden_dropout = 0,
                                       visible_dropout = 0))
print(rice.dnn)
rice.imp <- varImp(rice.dnn, useModel = FALSE)
plot(rice.imp)
predictions <- predict(rice.dnn, riceTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(rice.dnn)
ricetest=cbind(riceTest, predictions)
ricep=subset(ricetest, rice=="Y", select=c(Y) )
ricea=subset(ricetest, rice=="N", select=c(Y))
rice.eval=evaluate(p=ricep[,1],a=ricea[,1])
rice.eval
plot(rice.eval, 'ROC') ## plot ROC curve
rice.thld <- threshold(rice.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
ricednn.pred <- predict(t,rice.dnn, type="prob") 
ricednn.pred=1-ricednn.pred
ricemask=ricednn.pred>rice.thld
plot(ricemask, legend=F, main = "Rice Dnn")
points(ngcrop.proj, cex=0.1)

#+ Ensemble predictions <glm> <rf>, <gbm>, <dnn>  -------------------------------
# Ensemble set-up
pred <- stack(riceglm.pred, 
              ricerf.pred, ricegbm.pred, ricednn.pred)
names(pred) <- c("riceglm",
                 "ricerf","ricegbm", "ricednn")
geospred <- extract(pred, ngcrop.proj)
# presence/absence of maize (present = Y, absent = N)
riceens <- cbind.data.frame(ngcrop$cereal.rice, geospred)
riceens <- na.omit(riceens)
riceensTest <-riceens[-riceIndex,] ## replicate previous test set
names(riceensTest)[1]= "rice"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Cropland (CRP, present = Y, absent = N)
rice.ens <- train(rice ~. , data = riceensTest,
                family = "binomial", 
                method = "glmnet",
                trControl = ens)

riceens.pred <- predict(rice.ens, riceensTest,  type="prob") ## predict test-set
confusionMatrix(rice.ens) ## print validation summaries
rice.test <- cbind(riceensTest, riceens.pred)
ricep <- subset(rice.test, rice=="Y", select=c(Y))
ricea <- subset(rice.test, rice=="N", select=c(Y))
rice.eval <- evaluate(p=ricep[,1], a=ricea[,1]) ## calculate ROC's on test set <dismo>
rice.eval
plot(rice.eval, 'ROC') ## plot ROC curve
rice.thld <- threshold(rice.eval, 'spec_sens') ## TPR+TNR threshold for classification
riceens.pred <- predict(pred, rice.ens, type="prob") ## spatial prediction
plot(1-riceens.pred, axes=F, main= "Probability of Rice presence")
riceensmask <- 1-riceens.pred > mz.thld
riceensmask= riceensmask*cropmask
plot(riceensmask, axes = F, legend = F, main = "Rice ensemble prediction")
plot(varImp(rice.ens,scale=F))

#write rice results
rf=writeRaster(1-riceens.pred, filename="./NG_OCP_results/NG_rice_2015_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(1-riceensmask, filename="./NG_OCP_results/NG_rice_2015_mask.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for sorghum
sg.glm=train(sorghum ~ ., data=sgTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(sg.glm, sgTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(sg.glm)
#variable importance
plot(varImp(sg.glm,scale=F), main= "variable contribution sorghum glmnet")

sgtest=cbind(sgTest, predictions)
sgp=subset(sgtest, sorghum=="Y", select=c(Y) )
sga=subset(sgtest, sorghum=="N", select=c(Y))
sg.eval=evaluate(p=sgp[,1],a=sga[,1])
sg.eval
plot(sg.eval, 'ROC') ## plot ROC curve
sg.thld <- threshold(sg.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
sgglm.pred <- predict(t, sg.glm, type="prob") 
sgglm.pred=1-sgglm.pred
sgmask=sgglm.pred>sg.thld
sgmask=sgmask*cropmask
plot(sgmask, legend=F)

#random forest using binomial distribution for maize
sg.rf=train(sorghum ~ ., data=sgTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(sg.rf, sgTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(sg.rf)
#variable importance
plot(varImp(sg.rf,scale=F), main = "variable contribution sorghum RForest")

sgtest=cbind(sgTest, predictions)
sgp=subset(sgtest, sorghum=="Y", select=c(Y) )
sga=subset(sgtest, sorghum=="N", select=c(Y))
sg.eval=evaluate(p=sgp[,1],a=sga[,1])
sg.eval
plot(sg.eval, 'ROC') ## plot ROC curve
sg.thld <- threshold(sg.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
sgrf.pred <- predict(t,sg.rf, type="prob") 
sgrf.pred=1-sgrf.pred
sgmask=sgrf.pred>sg.thld
plot(sgmask, axes=F, legend=F)

#gradient boosting using binomial distribution for sorghum
sg.gbm=train(sorghum ~ ., data=sgTrain, 
             #family= "binomial",
             method="gbm",metric="Accuracy", trControl=objControl)

predictions <- predict(sg.gbm, sgTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(sg.gbm)
#variable importance
plot(varImp(sg.gbm,scale=F))

sgtest=cbind(sgTest, predictions)
sgp=subset(sgtest, sorghum=="Y", select=c(Y) )
sga=subset(sgtest, sorghum=="N", select=c(Y))
sg.eval=evaluate(p=sgp[,1],a=sga[,1])
sg.eval
plot(sg.eval, 'ROC') ## plot ROC curve
sg.thld <- threshold(sg.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
sggbm.pred <- predict(t,sg.gbm, type="prob") 
sggbm.pred=1-sggbm.pred
sgmask=sggbm.pred>sg.thld
plot(sgmask, axes=F, legend=F)

#deep neural net
sg.dnn <- train(sorghum ~ ., data=sgTrain, 
                  method = "dnn", 
                  trControl = tc,
                  tuneGrid = expand.grid(layer1 = 2:7,
                                         layer2 = 0:3,
                                         layer3 = 0:3,
                                         hidden_dropout = 0,
                                         visible_dropout = 0))
print(sg.dnn)
sg.imp <- varImp(sg.dnn, useModel = FALSE)
plot(sg.imp)
predictions <- predict(sg.dnn, sgTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(sg.dnn)
sgtest=cbind(sgTest, predictions)
sgp=subset(sgtest, sorghum=="Y", select=c(Y) )
sga=subset(sgtest, sorghum=="N", select=c(Y))
sg.eval=evaluate(p=sgp[,1],a=sga[,1])
sg.eval
plot(sg.eval, 'ROC') ## plot ROC curve
sg.thld <- threshold(sg.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
sgdnn.pred <- predict(t,sg.dnn, type="prob") 
sgdnn.pred=1-sgdnn.pred
sgmask=sgdnn.pred>sg.thld
plot(sgmask, legend=F, main = "Sorghum Dnn")
points(ngcrop.proj, cex=0.1)

#+ Ensemble predictions <glm> <rf>, <gbm>, <dnn>  -------------------------------
# Ensemble set-up
pred <- stack(sgglm.pred, 
              sgrf.pred, sggbm.pred, sgdnn.pred)
names(pred) <- c("sgglm",
                 "sgrf","sggbm", "sgdnn")
geospred <- extract(pred, ngcrop.proj)
# presence/absence of sorghum (present = Y, absent = N)
sgens <- cbind.data.frame(ngcrop$cereal.sorgum, geospred)
sgens <- na.omit(sgens)
sgensTest <-sgens[-sgIndex,] ## replicate previous test set
names(sgensTest)[1]= "sorghum"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Cropland (CRP, present = Y, absent = N)
sg.ens <- train(sorghum ~. , data = sgensTest,
                  family = "binomial", 
                  method = "rf",
                  trControl = ens)

sgens.pred <- predict(sg.ens, sgensTest,  type="prob") ## predict test-set
confusionMatrix(sg.ens) ## print validation summaries
sg.test <- cbind(sgensTest, sgens.pred)
sgp <- subset(sg.test, sorghum=="Y", select=c(Y))
sga <- subset(sg.test, sorghum=="N", select=c(Y))
sg.eval <- evaluate(p=sgp[,1], a=sga[,1]) ## calculate ROC's on test set <dismo>
sg.eval
plot(sg.eval, 'ROC') ## plot ROC curve
sg.thld <- threshold(sg.eval, 'spec_sens') ## TPR+TNR threshold for classification
sgens.pred <- predict(pred, sg.ens, type="prob") ## spatial prediction
plot((1-sgens.pred)*cropmask, axes=F, main= "Probability map of Sorghum in cropland")
sgensmask <- 1-sgens.pred > sg.thld
sgensmask= sgensmask*cropmask
plot(sgensmask, axes = F, legend = F, main = "Sorghum distribution prediction")
plot(varImp(sg.ens,scale=F))

#write sorghum results
rf=writeRaster(1-sgens.pred, filename="./NG_OCP_results/NG_sorghum_2015_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(sgensmask, filename="./NG_OCP_results/NG_sorghum_2015_mask.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for cowpea
cow.glm=train(cowpea ~ ., data=cowTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(cow.glm, cowTest[,2:18], type="prob")
#confusionMatrix on cross validation
confusionMatrix(cow.glm)
#variable importance
plot(varImp(cow.glm,scale=F))

cowtest=cbind(cowTest, predictions)
cowp=subset(cowtest, cowpea=="Y", select=c(Y) )
cowa=subset(cowtest, cowpea=="N", select=c(Y))
cow.eval=evaluate(p=cowp[,1],a=cowa[,1])
cow.eval
plot(cow.eval, 'ROC') ## plot ROC curve
cow.thld <- threshold(cow.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cowglm.pred <- predict(grid, cow.glm, type="prob") 
cowglmnet.pred=1-cowglm.pred
cowmask=cowglmnet.pred>cow.thld
cowmask2=cowmask*crp
plot(cowmask2, legend=F)

#write cowpea results
rf=writeRaster(cowmask2.pred, filename="./NG_OCP_results/NG_cowpea_2015_glm.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for soybean
soy.glm=train(soybean ~ ., data=soyTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(soy.glm, soyTest[,2:18], type="prob")
#confusionMatrix on cross validation
confusionMatrix(soy.glm)
#variable importance
plot(varImp(soy.glm,scale=F))

soytest=cbind(soyTest, predictions)
soyp=subset(soytest, soybean=="Y", select=c(Y) )
soya=subset(soytest, soybean=="N", select=c(Y))
soy.eval=evaluate(p=soyp[,1],a=soya[,1])
soy.eval
plot(soy.eval, 'ROC') ## plot ROC curve
soy.thld <- threshold(soy.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
soyglm.pred <- predict(grid, soy.glm, type="prob") 
soyglmnet.pred=1-soyglm.pred
soymask=soyglmnet.pred>soy.thld
soymask2=soymask*crp
plot(soymask2, legend=F)

#write soybean results
rf=writeRaster(soymask2.pred, filename="./NG_OCP_results/NG_soybean_2015_glm.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for wheat
wht.glm=train(wheat ~ ., data=whtTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(wht.glm, whtTest[,2:18], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wht.glm)
#variable importance
plot(varImp(wht.glm,scale=F))

whttest=cbind(whtTest, predictions)
whtp=subset(whttest, wheat=="Y", select=c(Y) )
whta=subset(whttest, wheat=="N", select=c(Y))
wht.eval=evaluate(p=whtp[,1],a=whta[,1])
wht.eval
plot(wht.eval, 'ROC') ## plot ROC curve
wht.thld <- threshold(wht.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
whtglm.pred <- predict(grid, wht.glm, type="prob") 
whtglmnet.pred=1-whtglm.pred
whtmask=whtglmnet.pred>wht.thld
whtmask2=whtmask*crp
plot(whtmask2, legend=F)

#write wheat results
rf=writeRaster(whtmask2.pred, filename="./NG_OCP_results/NG_wheat_2015_glm.tif", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for millet
mil.glm=train(millet ~ ., data=milTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(mil.glm, milTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mil.glm)
#variable importance
plot(varImp(mil.glm,scale=F), main= "variable importance Millet glmnet")

miltest=cbind(milTest, predictions)
milp=subset(miltest, millet=="Y", select=c(Y) )
mila=subset(miltest, millet=="N", select=c(Y))
mil.eval=evaluate(p=milp[,1],a=mila[,1])
mil.eval
plot(mil.eval, 'ROC') ## plot ROC curve
mil.thld <- threshold(mil.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
milglm.pred <- predict(t, mil.glm, type="prob") 
milglm.pred=1-milglm.pred
milmask=milglm.pred>mil.thld
milmask=milmask*cropmask
plot(milmask, axes=F, legend=F)
#random Forest millet
mil.rf=train(millet ~ ., data=milTrain, family= "binomial",method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(mil.rf, milTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mil.rf)
#variable importance
plot(varImp(mil.rf,scale=F), main = "variable importance Millet RForest")

miltest=cbind(milTest, predictions)
milp=subset(miltest, millet=="Y", select=c(Y) )
mila=subset(miltest, millet=="N", select=c(Y))
mil.eval=evaluate(p=milp[,1],a=mila[,1])
mil.eval
plot(mil.eval, 'ROC') ## plot ROC curve
mil.thld <- threshold(mil.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
milrf.pred <- predict(t, mil.rf, type="prob") 
milrf.pred=1-milrf.pred
milmask=milrf.pred>mil.thld
milmask=milmask*cropmask
plot(milmask, legend=F,axes=F, main = "Pearl Millet RF")

#Gradient Boosting millet
mil.gbm=train(millet ~ ., data=milTrain, 
             # family= "binomial",
              method="gbm",metric="Accuracy", trControl=objControl)

predictions <- predict(mil.gbm, milTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mil.gbm)
#variable importance
plot(varImp(mil.gbm,scale=F))

miltest=cbind(milTest, predictions)
milp=subset(miltest, millet=="Y", select=c(Y) )
mila=subset(miltest, millet=="N", select=c(Y))
mil.eval=evaluate(p=milp[,1],a=mila[,1])
mil.eval
plot(mil.eval, 'ROC') ## plot ROC curve
mil.thld <- threshold(mil.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
milgbm.pred <- predict(t, mil.gbm, type="prob") 
milgbm.pred=1-milgbm.pred
milmask=milgbm.pred>mil.thld
milmask=milmask*cropmask
plot(milmask, legend=F,axes=F, main = "Pearl Millet GBM")

#deepnet
mil.dnn <- train(millet ~ ., data=milTrain, 
                method = "dnn", 
                trControl = tc,
                tuneGrid = expand.grid(layer1 = 2:7,
                                       layer2 = 0:3,
                                       layer3 = 0:3,
                                       hidden_dropout = 0,
                                       visible_dropout = 0))
print(mil.dnn)
mil.imp <- varImp(mil.dnn, useModel = FALSE)
plot(mil.imp)
predictions <- predict(mil.dnn, milTest[,2:26], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mil.dnn)
miltest=cbind(milTest, predictions)
milp=subset(miltest, millet=="Y", select=c(Y) )
mila=subset(miltest, millet=="N", select=c(Y))
mil.eval=evaluate(p=milp[,1],a=mila[,1])
mil.eval
plot(mil.eval, 'ROC') ## plot ROC curve
mil.thld <- threshold(mil.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
mildnn.pred <- predict(t,mil.dnn, type="prob") 
mildnn.pred=1-mildnn.pred
milmask=mildnn.pred>mil.thld
plot(milmask, legend=F, main = "Millet Dnn")

#+ Ensemble predictions <glm> <rf>, <gbm>, <dnn>  -------------------------------
# Ensemble set-up
pred <- stack(milglm.pred, 
              milrf.pred, milgbm.pred, mildnn.pred)
names(pred) <- c("milglm",
                 "milrf","milgbm", "mildnn")
geospred <- extract(pred, ngcrop.proj)
# presence/absence of sorghum (present = Y, absent = N)
milens <- cbind.data.frame(ngcrop$cereal.millet, geospred)
milens <- na.omit(milens)
milensTest <-milens[-milIndex,] ## replicate previous test set
names(milensTest)[1]= "millet"


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of millet ( present = Y, absent = N)
mil.ens <- train(millet ~. , data = milensTest,
                #family = "binomial", 
                method = "gbm",
                trControl = ens)

milens.pred <- predict(mil.ens, milensTest,  type="prob") ## predict test-set
confusionMatrix(mil.ens) ## print validation summaries
mil.test <- cbind(milensTest, milens.pred)
milp <- subset(mil.test, millet=="Y", select=c(Y))
mila <- subset(mil.test, millet=="N", select=c(Y))
mil.eval <- evaluate(p=milp[,1], a=mila[,1]) ## calculate ROC's on test set <dismo>
mil.eval
plot(mil.eval, 'ROC') ## plot ROC curve
mil.thld <- threshold(mil.eval, 'spec_sens') ## TPR+TNR threshold for classification
milens.pred <- predict(pred, mil.ens, type="prob") ## spatial prediction
plot(1-milens.pred, axes=F)
milensmask <- 1-milens.pred > mil.thld
milensmask= milensmask*cropmask
plot(milensmask, axes = F, legend = F, main = "Pearl Millet ensemble prediction")
plot(varImp(mil.ens,scale=F))
#write millet results
rf=writeRaster(1-milens.pred, filename="./NG_OCP_results/NG_millet_2015_ens.tif", format= "GTiff", overwrite=TRUE)
rf=writeRaster(milensmask, filename="./NG_OCP_results/NG_millet_2015_mask.tif", format= "GTiff", overwrite=TRUE)

#now to GW model
#https://rpubs.com/chrisbrunsdon/99675  one reference
library(GWmodel)

#PCA on t for survey area
pca=princomp(as.matrix(na.omit(values(t)), cor=TRUE))  #gets rid of NAs
plot(pca)
pca$loadings
pcang=predict(t,pca, index=1:4)
plot (pcang)
#for robust PCA on variables for survey area
R.COV <- covMcd(as.matrix(na.omit(values(t)),cor = F, alpha = 0.75, scale= T))
pca.robust <- princomp(as.matrix(na.omit(values(t)), covmat = R.COV, cor = F))
pca.robust$loadings

#PCA predictions for survey area (first 4 components)
pca.robng=predict(t,pca.robust, index=1:4)
names(pca.robng)= c("Comp1", "Comp2", "Comp3", "Comp4")
plot(pca.robng)

# GW PCA for samples within survey area
# for crop soils samples
ngcropex <- SpatialPointsDataFrame(coordinates(ngcrop.proj), as.data.frame(extract(t, ngcrop.proj)))
ngcropex= sp.na.omit(ngcropex)
projection(ngcropex) <- projection(grid)
#basic
bw.gwpca.basic <- bw.gwpca(ngcropex,vars = colnames(ngcropex@data), k = 3, 
                           robust = FALSE, adaptive = TRUE)

gwpca.basic <- gwpca(ngcropex,vars = colnames(ngcropex@data), bw = bw.gwpca.basic, k = 10,
                     robust = FALSE, adaptive = TRUE)

#robust: make sure the ngcropex has last two columns created by basic deleted
ngcropex=ngcropex[,1:25]
bw.gwpca.robust <- bw.gwpca(ngcropex,vars = colnames(ngcropex@data), k = 3, 
                           robust = TRUE, adaptive = TRUE)
gwpca.robust <- gwpca(ngcropex,vars = colnames(ngcropex@data), bw = bw.gwpca.robust, 
                         robust = TRUE, adaptive = TRUE)
#function to get Percent Total Variance PTV
prop.var <- function(gwpca.obj, n.components) 
  { return((rowSums(gwpca.obj$var[, 1:n.components])/ rowSums(gwpca.obj$var))*100)
  }
# PVT for basic GWPCA
var.gwpca.basic <- prop.var(gwpca.basic, 3)
ngcropex$var.gwpca.basic <- var.gwpca.basic

#plot 1st component basic
spplot(ngcropex, "var.gwpca.basic", main= "Percentage Total Variance, basic")

#plot 1st component robust
var.gwpca.robust= prop.var(gwpca.robust,3)
ngcropex$var.gwpca.robust= var.gwpca.robust

spplot(ngcropex, "var.gwpca.robust", main= "Percentage Total Variance, robust")

#plot with variable names of "winning" variable
loadings.pc1.basic <- gwpca.basic$loadings[, , 1] # first component

lead.item <- colnames(loadings.pc1.basic)[max.col(abs(loadings.pc1.basic))]
ngcropex$win.item.basic= lead.item
spdf1=SpatialPointsDataFrame(coordinates(ngcropex), data.frame(lead=lead.item))
spplot(spdf1, "lead", main = "Winning variable: highest abs. loading on local Comp.1 (basic)")

#plot with variable names of "winning" variable robust
loadings.pc1.robust <- gwpca.robust$loadings[, , 1] # first component

lead.item2 <- colnames(loadings.pc1.robust)[max.col(abs(loadings.pc1.robust))]
ngcropex$win.item.robust= lead.item2
spdf2=SpatialPointsDataFrame(coordinates(ngcropex), data.frame(lead=lead.item2))
spplot(spdf2, "lead", main = "Highest abs. loading on local Comp.1 (robust)")
