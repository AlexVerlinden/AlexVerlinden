# Script for crop distribution  models NG using ensemble regressions
# basis for cropland mask is 5000 point survey for OCP 2016 conducted by AfSIS for Africa
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
require(foreach)
require(doMC)
require(doParallel)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("NG_250m_OCP", showWarnings=F)
dat_dir <- "./NG_250m_OCP"
# download crop, woodland, human settlements presence/absence locations
# these are data from 2016 5000 Geosurvey
download.file("https://www.dropbox.com/s/hgfqkiej2xzxsxn/NG_OCP_COV.csv?dl=0", 
              "./NG_250m_OCP/NG_OCP_COV.csv", mode="wb")
# note that 0 and 1 are not ok for Caret for classifications, should be N and Y or similar
ngcov <- read.csv(paste(dat_dir, "/NG_OCP_COV.csv", sep= ""), header=T, sep=",")
#download 2916 test sites, these are field sites that were also geosurveyed
# download crop presence/absence locations
# these are data from 2016 crop scout ODK forms n= 2915
download.file("https://www.dropbox.com/s/lz3h5mr488k6ine/crops_field_geos_short.csv?dl=0", "./NG_250m_OCP/crops_field_geos_short.csv", mode="wb")
# note that 0 and 1 are not ok for Caret for classifications, should be N and Y or similar
ngcrop <- read.csv(paste(dat_dir, "/crops_field_geos_short.csv", sep= ""), header=T, sep=",")

#download grids for NG cover OCP ~ 110MB !!!!!
download.file("https://www.dropbox.com/s/5dh85rnxbw22j5a/OCP_250mgrids.zip?dl=0","./NG_250m_OCP/OCP_250mgrids.zip",  mode="wb")
unzip("./NG_250m_OCP/OCP_250mgrids.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid,center=TRUE, scale=TRUE)
#+ Data setup for NG cover OCP--------------------------------------------------------------
# Project cover data to grid CRS
ngcov.proj <- as.data.frame(project(cbind(ngcov$Longitude, ngcov$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ngcov.proj) <- c("x","y")
coordinates(ngcov.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ngcov.proj) <- projection(grid)

# Project crop test data to grid CRS
ngcroptest.proj <- as.data.frame(project(cbind(ngcrop$Longitude, ngcrop$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ngcroptest.proj) <- c("x","y")
coordinates(ngcroptest.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ngcroptest.proj) <- projection(grid)

#restrict to only cropland
#dir.create("NG_OCP_cover_results", showWarnings=F)
#download.file("https://www.dropbox.com/s/0i11dkey8z5male/OCP_100m_cropmask.zip?dl=0","./NG_OCP_cover_results/OCP_100m_cropmask.zip", mode="wb")
#unzip("./NG_OCP_cover_results/OCP_100m_cropmask.zip")
#cropmask=raster("./NG_OCP_cover_results/OCP_100m_cropmask.tif", exdir=dat_dir, overwrite=T)
#cropmask[cropmask == 0] <- NA
#v=t*cropmask
# Extract gridded variables for NG cover data observations 
ngcovex <- data.frame(coordinates(ngcov.proj), extract(t, ngcov.proj))
ngcovex=  ngcovex[,3:30]#exclude coordinates

#test set
ngcropex <- data.frame(coordinates(ngcroptest.proj), extract(t, ngcroptest.proj))
ngcropex=  ngcropex[,3:30]#exclude coordinates

###### Forest cover
###!!!! note this layer is used later to refine cropland predictions
#for woodland presence on train set
wldpresabs=cbind(ngcov$WDL, ngcovex)
wldpresabs=na.omit(wldpresabs)
colnames(wldpresabs)[1]="WLD"
wldpresabs$WLD=as.factor(wldpresabs$WLD)

summary(wldpresabs)
#to test if woodland is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(wldpresabs$WLD))

#for woodland presence on test set
testwoodland=cbind(ngcrop$WDL, ngcropex)
testwoodland=na.omit(testwoodland)
colnames(testwoodland)[1]="WLD"
#testwoodland[!testwoodland$WLD == "U", ]

testwoodland$WLD=as.factor(testwoodland$WLD)

#____________
#set up data for caret
mc <- makeCluster(detectCores())
registerDoParallel(mc)


###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)


objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none', allowParallel = TRUE)
#glmnet using binomial distribution for cropland
wld.glm=train(WLD ~ ., data=wldpresabs, family= "binomial",
              method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(wld.glm, testwoodland[,2:29], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wld.glm)
#variable importance
plot(varImp(wld.glm,scale=F))
wldtest=cbind(testwoodland, predictions)
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
wld.rf=train(WLD ~ ., data=wldpresabs, family= "binomial",method="rf",
             metric="Accuracy", trControl=objControl )

predictions <- predict(wld.rf, testwoodland[,2:29], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wld.rf)
#variable importance
plot(varImp(wld.rf,scale=F))
wldtest=cbind(testwoodland, predictions)
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
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = TRUE)
wld.gbm=train(WLD ~ ., data=wldpresabs, method="gbm",metric="Accuracy", 
              trControl=gbm)

predictions <- predict(wld.gbm, testwoodland[,2:29], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wld.gbm)
#variable importance
plot(varImp(wld.gbm,scale=F))
wldtest=cbind(testwoodland, predictions)
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
#wldens <- cbind.data.frame(ngcov$WDL, geospred)
#wldens <- na.omit(wldens)

#test set
geotest=extract(pred, ngcroptest.proj)
wldensTest <- cbind.data.frame(ngcrop$WDL,geotest)

wldensTest$WDL=as.factor(ngcrop$WDL) ## replicate previous test set

names(wldensTest)[1]= "WLD"
wldensTest= na.omit(wldensTest)
wldensTest=subset(wldensTest, !WLD== "U")
wldensTest=droplevels(wldensTest)
wldensTest=wldensTest[,1:4]
# Regularized ensemble weighting on the test set <gbm>
# 10-fold CV

ens <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# presence/absence of woodland (WLD, present = Y, absent = N)
wld.ens <- train(WLD ~. , data = wldensTest,
                family = "binomial", 
                 method = "glmnet", metric= "Accuracy",
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

rf=writeRaster(1-wldens.pred,filename="./NG_OCP_cover_results/OCP_250m_woodens.tif", format= "GTiff", overwrite = TRUE)

# human settlements

#for buildings presence
hsppresabs=cbind(ngcov$HSP, ngcovex)
hsppresabs=na.omit(hsppresabs)
colnames(hsppresabs)[1]="HSP"
hsppresabs$HSP=as.factor(hsppresabs$HSP)
summary(hsppresabs)
#to test if cover is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(hsppresabs$HSP))


#for human settlement presence on test set
testhsp=cbind(ngcrop$HSP, ngcropex)
testhsp=na.omit(testhsp)
colnames(testhsp)[1]="HSP"
#____________
#set up data for caret
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none', allowParallel = TRUE)
#glmnet using binomial distribution for cropland
hsp.glm=train(HSP ~ ., data=hsppresabs, family= "binomial",
              method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(hsp.glm, testhsp[,2:29], type="prob")
#confusionMatrix on cross validation
confusionMatrix(hsp.glm)
#variable importance
plot(varImp(hsp.glm,scale=F))
hsptest=cbind(testhsp, predictions)
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

hsp.gbm=train(HSP ~ ., data=hsppresabs, 
              #family= "binomial",
              method="gbm",metric="Accuracy", trControl=objControl)

predictions <- predict(hsp.gbm, testhsp[,2:29], type="prob")
#confusionMatrix on cross validation
confusionMatrix(hsp.gbm)
#variable importance
plot(varImp(hsp.gbm,scale=F))
hsptest=cbind(testhsp, predictions)
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
hsp.rf=train(HSP ~ ., data=hsppresabs, family= "binomial",
             method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(hsp.rf, testhsp[,2:29], type="prob")
#confusionMatrix on cross validation
confusionMatrix(hsp.rf)
#variable importance
plot(varImp(hsp.rf,scale=F))
hsptest=cbind(testhsp, predictions)
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

#test set
geotest=extract(pred, ngcroptest.proj)
hspensTest <- cbind.data.frame(ngcrop$HSP,geotest)

hspensTest$HSP=as.factor(ngcrop$HSP) ## replicate previous test set

names(hspensTest)[1]= "HSP"
hspensTest= na.omit(hspensTest)
hspensTest=subset(hspensTest, !HSP== "U")
hspensTest=droplevels(hspensTest)
hspensTest=hspensTest[,1:4]

# Regularized ensemble weighting on the test set <gbm>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# presence/absence of settlements (HSP, present = Y, absent = N)
hsp.ens <- train(HSP ~. , data = hspensTest,
                 family = "binomial", 
                 method = "glmnet",
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
hspensmask <- 1-hspens.pred > hsp.thld
plot(hspensmask, axes = F, legend = F, main= "Settlement Ensemble Prediction")
plot(varImp(hsp.ens,scale=F))

rf=writeRaster(hspens.pred,filename="./NG_OCP_cover_results/OCP_250m_settlens.tif", 
               format= "GTiff", overwrite = TRUE)

#for crop presence now including woodland and settlement layers as covariates
OCP_woodens=raster("./NG_OCP_cover_results/OCP_250m_woodens.tif", exdir=dat_dir, overwrite=T)
OCP_settlens=raster("./NG_OCP_cover_results/OCP_250m_settlens.tif", exdir=dat_dir, overwrite=T)

glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist,OCP_woodens, OCP_settlens)
t=scale(grid,center=TRUE, scale=TRUE)

ngcovex <- data.frame(coordinates(ngcov.proj), extract(t, ngcov.proj))
ngcovex=  ngcovex[,3:32]
covpresabs=cbind(ngcov$CRP, ngcovex)
covppresabs=na.omit(covpresabs)
colnames(covpresabs)[1]="Crop"
covpresabs$Crop=as.factor(covpresabs$Crop)
summary(covpresabs)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(covpresabs$Crop))

#test set
ngcropex <- data.frame(coordinates(ngcroptest.proj), extract(t, ngcroptest.proj))
ngcropex=  ngcropex[,3:32]#exclude coordinates

testcrp=cbind(ngcrop$CRP, ngcropex)
testcrp=na.omit(testcrp)
colnames(testcrp)[1]="CRP"



###### Regressions


#____________
#set up data for caret
objControl <- trainControl(method='cv', number=5, classProbs = T,
                           returnResamp='none', allowParallel = TRUE)
#glmnet using binomial distribution for cropland
crop.glm=train(Crop ~ ., data=covpresabs, family= "binomial",
               method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(crop.glm, testcrp[,2:31], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.glm)
#variable importance
plot(varImp(crop.glm,scale=F))
croptest=cbind(testcrp, predictions)
cropp=subset(croptest, CRP=="Y", select=c(Y) )
cropa=subset(croptest, CRP=="N", select=c(Y))
crop.eval=evaluate(p=cropp[,1],a=cropa[,1])
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cropglm.pred <- predict(t,crop.glm, type="prob") 
cropglm.pred=1-cropglm.pred
cropmask=cropglm.pred>crop.thld
plot(cropmask, axes=F,legend=F, main= "Cropland")


#random forest using binomial distribution for cropland
crop.rf=train(Crop ~ ., data=covpresabs, family= "binomial",
               method="rf",metric="Accuracy", trControl=objControl)

predictions <- predict(crop.rf, testcrp[,2:31], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.rf)
#variable importance
plot(varImp(crop.rf,scale=F))
croptest=cbind(testcrp, predictions)
cropp=subset(croptest, CRP=="Y", select=c(Y) )
cropa=subset(croptest, CRP=="N", select=c(Y))
crop.eval=evaluate(p=cropp[,1],a=cropa[,1])
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
croprf.pred <- predict(t,crop.rf, type="prob") 
croprf.pred=1-croprf.pred
cropmask=croprf.pred>crop.thld
plot(cropmask, axes=F,legend=F, main= "Cropland RF")


#gradient boosting using binomial distribution for cropland
crop.gbm=train(Crop ~ ., data=covpresabs, #family= "binomial",
              method="gbm",metric="Accuracy", trControl=objControl)

predictions <- predict(crop.gbm, testcrp[,2:31], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.gbm)
#variable importance
plot(varImp(crop.gbm,scale=F))
croptest=cbind(testcrp, predictions)
cropp=subset(croptest, CRP=="Y", select=c(Y) )
cropa=subset(croptest, CRP=="N", select=c(Y))
crop.eval=evaluate(p=cropp[,1],a=cropa[,1])
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cropgbm.pred <- predict(t,crop.gbm, type="prob") 
cropgbm.pred=1-cropgbm.pred
cropmask=cropgbm.pred>crop.thld
plot(cropmask, axes=F,legend=F, main= "Cropland GBM")


#deep neural net
tc <- trainControl(method = "cv", number = 10, repeats= 3, allowParallel=TRUE)
crop.dnn <- train(Crop ~ ., data=covpresabs, 
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
predictions <- predict(crop.dnn, testcrp[,2:31], type="prob")
#confusionMatrix on cross validation
confusionMatrix(crop.dnn)
croptest=cbind(testcrp, predictions)
cropp=subset(croptest, CRP=="Y", select=c(Y) )
cropa=subset(croptest, CRP=="N", select=c(Y))
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


#+ Ensemble predictions <glm> <rf>, <gbm> dnn, -------------------------------
# Ensemble set-up
pred <- stack(cropglm.pred, croprf.pred, cropgbm.pred, cropdnn.pred)

names(pred) <- c("cropglm","croprf","cropgbm", "cropdnn")
plot(pred, axes=F)
geospred <- extract(pred, ngcroptest.proj)
# presence/absence of cropland (present = Y, absent = N)
cropens <- cbind.data.frame(ngcrop$CRP, geospred)
cropens <- na.omit(cropens)
cropens=subset(cropens, !crop== "U")
cropens=droplevels(cropens)



# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# presence/absence of Cropland (CRP, present = Y, absent = N)
crop.ens <- train(crop ~. , data = cropens,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)

cropens.pred <- predict(crop.ens, cropens,  type="prob") ## predict test-set
#confusionMatrix(cropens.pred, cropensTest$crop, "Y") ## print validation summaries
confusionMatrix(crop.ens)
crop.test <- cbind(cropens, cropens.pred)
cropp <- subset(crop.test, crop=="Y", select=c(Y))
cropa <- subset(crop.test, crop=="N", select=c(Y))
crop.eval <- evaluate(p=cropp[,1], a=cropa[,1]) ## calculate ROC's on test set <dismo>
crop.eval
plot(crop.eval, 'ROC') ## plot ROC curve
crop.thld <- threshold(crop.eval, 'spec_sens') ## TPR+TNR threshold for classification
cropens.pred <- predict(pred, crop.ens, type="prob") ## spatial prediction
plot(1-cropens.pred, axes=F)
cropensmask <- 1-cropens.pred > crop.thld
plot(cropensmask, axes = F, legend = F, main= "Cropland Ensemble Prediction")
plot(varImp(crop.ens,scale=F))

rf=writeRaster(1-cropens.pred,filename="./NG_OCP_cover_results/OCP_250m_cropens2.tif", format= "GTiff", overwrite = TRUE)
rf=writeRaster(cropensmask,filename="./NG_OCP_cover_results/OCP_250m_cropmask.tif", format= "GTiff", overwrite = TRUE)

