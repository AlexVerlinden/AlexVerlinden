#' Ensemble predictions of Ghana GeoSurvey fractional, tree , shrub, grass and bare ground cover,
#' at 100000 observations. using point intercept method on 16 points/ha
#' Alex Verlinden 2016 modified after M. Walsh, April 2014
# observations collected by crowdsourcing using "Geosurvey"  in October and November 2015 
# Required packages
# install.packages(c("downloader","raster","rgdal","caret","randomForest","gbm","nnet","glmnet","dismo")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(caret)
require(randomForest)
require(gbm)
require(nnet)
require(glmnet)
require(dismo)
require(deepnet)
require(doParallel)


#rm(list=ls()) # to clear memory
#+ Data downloads ----------------------------------------------------------
# Create a "Ghana Data" folder in your current working directory
dir.create("GH_data", showWarnings=F)
dat_dir <- "./GH_data"

# download GeoSurvey data
#https://www.dropbox.com/s/ktihfst0ogr3hac/Ghana_100k.zip?dl=0
download.file("https://www.dropbox.com/s/ktihfst0ogr3hac/Ghana_100k.zip?dl=0", 
              "./GH_data/Ghana_100k.zip", mode="wb")
unzip("./GH_data/Ghana_100k.zip", exdir="./GH_data", overwrite=T)
geofrac <- read.table(paste(dat_dir, "/Ghana_100000ha.csv", sep=""), header=T, sep=",")
geofrac <- na.omit(geofrac)
#for 100000 crop fractional samples
download.file("https://www.dropbox.com/s/f9uelyqmwdyg0n1/GH_100k_crop.zip?dl=0",
              "./GH_data/GH_100k_crop.zip", mode="wb")
unzip("./GH_data/GH_100k_crop.zip", exdir="./GH_data", overwrite = T)
cropfrac=read.table(paste(dat_dir, "/GH_100k_crop.csv", sep=""), header=T, sep= ",")
cropfrac=na.omit(cropfrac)

# download Ghana Gtifs (~165 Mb!!!) 250 m and stack in raster
download.file("https://www.dropbox.com/s/nbpi0l4utm32cgw/GH_250_grids.zip?dl=0", "./GH_data/GH_250_grids.zip", mode="wb")
unzip("./GH_data/GH_250_grids.zip", exdir="./GH_data", overwrite=T)
glist <- list.files(path="./GH_data/GH_250_grids", pattern="tif", full.names=T)
grid <- stack(glist)

t=scale(grid, center=TRUE,scale=TRUE) # scale all covariates
#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
geofrac.proj <- as.data.frame(project(cbind(geofrac$Longitude, geofrac$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geofrac.proj) <- c("x","y")
geofrac <- cbind(geofrac, geofrac.proj)
coordinates(geofrac) <- ~x+y
projection(geofrac) <- projection(grid)
#this is for crop fractional cover
cropfrac.proj=as.data.frame(project(cbind(cropfrac$Longitude, cropfrac$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(cropfrac.proj) <- c("x","y")
cropfrac <- cbind(cropfrac, cropfrac.proj)
coordinates(cropfrac) <- ~x+y
projection(cropfrac) <- projection(grid)

# Extract gridded variables at GeoSurvey locations
geosgrid <- extract(t, geofrac)
# Assemble dataframes

#first Water and Settlements
HSP <- geofrac$BLD
hspdat <- cbind.data.frame(HSP, geosgrid)
hspdat <- na.omit(hspdat)
#to test how imbalanced the data are
prop.table(table(HSP))
#surface water 
WAT= geofrac$WAT
WATdat <- cbind.data.frame(WAT, geosgrid)
WATdat <- na.omit(WATdat)
prop.table(table(WAT))

set.seed(12949)
# Settlement train/test split
hspIndex <- createDataPartition(hspdat$HSP, p = 2/3, list = FALSE, times = 1)
hspTrain <- hspdat[ hspIndex,]
hspTest  <- hspdat[-hspIndex,]

# water
WATIndex= createDataPartition(WATdat$WAT, p = 2/3, list = FALSE, times = 1)
WATTrain <- WATdat[ WATIndex,]
WATTest  <- WATdat[-WATIndex,]

#parallel computing
mc <- makeCluster(detectCores())
registerDoParallel(mc)

HSP.rf <- train(HSP ~ ., data = hspTrain,
                method = "rf",
                metric= "ROC",
                tuneGrid=data.frame(mtry=3),
                ntree=201,
                trControl = trainControl(method = "repeatedcv",
                                         repeats = 5,
                                         classProbs = TRUE,
                     summaryFunction = twoClassSummary))

hsprf.pred=predict (t, HSP.rf, type= "prob")
dir.create("GH_results", showWarnings=F)
writeRaster(1-hsprf.pred, filename="./GH_results/GH_hsppreds.tif", overwrite=TRUE)

WAT.rf=train(WAT ~ ., data = WATTrain,
             method = "rf",
             metric= "ROC",
             tuneGrid=data.frame(mtry=3),
             ntree=201,
             trControl = trainControl(method = "repeatedcv",
                                      repeats = 5,
                                      classProbs = TRUE,allowParallel = TRUE,
                                      summaryFunction = twoClassSummary))
watrf.pred=predict(t, WAT.rf, type= "prob")
writeRaster(1-watrf.pred,filename="./GH_results/GH_watpreds.tif")

#Bare ground (BRG 0-100)
BRG <- round(geofrac$BRG*6.25)
BRGdat <- cbind.data.frame(BRG, geosgrid)
BRGdat <- na.omit(BRGdat)

#Grass (GRS 0-100)
GRS=round(geofrac$GRS*6.25)
GRSdat=cbind.data.frame(GRS, geosgrid)
GRSdat <- na.omit(GRSdat)

#shrubs (SRB 0-100)
SRB=round(geofrac$SHRB*6.25)
SRBdat=cbind.data.frame(SRB, geosgrid)
SRBdat <- na.omit(SRBdat)

#trees (TRE 0-100)
TRE=round(geofrac$TREE*6.25)
TREdat=cbind.data.frame(TRE, geosgrid)
TREdat <- na.omit(TREdat)

# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# bare ground train/test split
brgIndex <- createDataPartition(BRGdat$BRG, p = 2/3, list = FALSE, times = 1)
brgTrain <- BRGdat[ brgIndex,]
brgTest  <- BRGdat[-brgIndex,]

# grass
grsIndex<- createDataPartition(GRSdat$GRS, p = 2/3, list = FALSE, times = 1)
grsTrain <- GRSdat[ grsIndex,]
grsTest  <- GRSdat[-grsIndex,]

# shrubs
srbIndex=createDataPartition(SRBdat$SRB, p = 2/3, list = FALSE, times = 1)
srbTrain <- SRBdat[ srbIndex,]
srbTest  <- SRBdat[-srbIndex,]

#Trees
treIndex=createDataPartition(TREdat$TRE, p = 2/3, list = FALSE, times = 1)
treTrain <- TREdat[ treIndex,]
treTest  <- TREdat[-treIndex,]

#Trees
#glmnet
objControl <- trainControl(method='cv', number=10, allowParallel = TRUE)

TRE.glm=train(TRE ~ ., data=treTrain, family= "gaussian", method= "glmnet",
              metric= "RMSE",
              trControl=objControl)

plot(varImp(TRE.glm,scale=F), main = "Variable contribution to Tree cover elasic net")
treglm.pred= predict(t, TRE.glm)

#Bare ground
BRG.glm=train(BRG ~ ., data=brgTrain, family= "gaussian", method= "glmnet",
              metric= "RMSE",
              trControl=objControl)
brgglm.pred=predict(t, BRG.glm)
# Deep neural net models --------------------------------------------------
# Start to parallelize model fitting  

mc <- makeCluster(detectCores())
registerDoParallel(mc)

#deepnet for bare ground
tc=trainControl(method = "cv", number = 10, allowParallel = TRUE)

BRG.dnn <- train(BRG ~., data=brgTrain, 
                method = "dnn",
               # preProc = c("center", "scale"), 
                trControl = tc,
               metric= "RMSE",
                tuneGrid = expand.grid(layer1 = 2:6,
                                       layer2 = 0:3,
                                       layer3 = 0:3,
                                       hidden_dropout = 0,
                                       visible_dropout = 0))
print(BRG.dnn)
brg.imp <- varImp(BRG.dnn, useModel = FALSE)
plot(brg.imp, top=27)
brgdnn.test <- predict(BRG.dnn, brgTest) # predict test set

brgdnn.pred <- predict(t, BRG.dnn)

#deepnet for grass
GRS.dnn <- train(GRS ~., data=grsTrain, 
                 method = "dnn",
                 # preProc = c("center", "scale"), 
                 trControl = tc,
                 tuneGrid = expand.grid(layer1 = 2:6,
                                        layer2 = 0:3,
                                        layer3 = 0:3,
                                        hidden_dropout = 0,
                                        visible_dropout = 0))
print(GRS.dnn)
grs.imp <- varImp(GRS.dnn, useModel = FALSE)
plot(grs.imp, top=27)
grsdnn.test <- predict(GRS.dnn, grsTest) # predict test set
grsdnn.pred <- predict(t, GRS.dnn)

#deepnet for shrubs
SRB.dnn <- train(SRB ~., data=srbTrain, 
                 method = "dnn",
                 # preProc = c("center", "scale"), 
                 trControl = tc,
                 tuneGrid = expand.grid(layer1 = 2:6,
                                        layer2 = 0:3,
                                        layer3 = 0:3,
                                        hidden_dropout = 0,
                                        visible_dropout = 0))
print(SRB.dnn)
srb.imp <- varImp(SRB.dnn, useModel = FALSE)
plot(srb.imp, top=27)
srbdnn.test <- predict(SRB.dnn, srbTest) # predict test set
srbdnn.pred <- predict(t, SRB.dnn)

#deepnet for Tree
TRE.dnn <- train(TRE ~., data=treTrain, 
                 method = "dnn",
                 # preProc = c("center", "scale"), 
                 metric= "RMSE",
                 trControl = tc)
                 
print(TRE.dnn)
tre.imp <- varImp(TRE.dnn, useModel = FALSE)
plot(tre.imp, top=27)
trednn.test <- predict(TRE.dnn, treTest) # predict test set
trednn.pred <- predict(t, TRE.dnn)

#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
trControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, allowParallel = TRUE)
oob=trainControl(method = "oob", allowParallel = TRUE)
# Bare ground (BRG, 0-100) including Importance vastly increases processing time
BRG.rf <- train(BRG ~ ., data = brgTrain,
                method = "rf",
                metric = "RMSE", maximize= FALSE,
                trControl = oob)

print(BRG.rf) #check RMSE
brgrf.test <- predict(BRG.rf, brgTest) ## predict test-set
brgrf.pred <- predict(t, BRG.rf) ## spatial predictions
plot(varImp(BRG.rf,scale=F), main = "Variable contribution to Bare ground RF")

#+ Random forests <randomForest>
# grass cover
GRS.rf <- train(GRS ~ ., data = grsTrain,
                method = "rf", 
                ntree = 201,
                #importance=T, 
                metric = "RMSE", maximize= FALSE,
                trControl = oob)
print(GRS.rf)
grsrf.test <- predict(GRS.rf, grsTest) ## predict test-set
grsrf.pred <- predict(t, GRS.rf) ## spatial predictions
plot(varImp(GRS.rf,scale=F), main="Variable contribution to Grass cover RF")
#RF for shrubs
SRB.rf <- train(SRB ~ ., data = srbTrain,
                method = "rf", 
                #importance=T, 
                metric = "RMSE", maximize= FALSE,
                trControl = oob)
print(SRB.rf) #check RMSE
srbrf.test <- predict(SRB.rf, srbTest) ## predict test-set
srbrf.pred <- predict(t, SRB.rf) ## spatial predictions
plot(varImp(SRB.rf,scale=F), main="Variable contribution to Shrub cover RF")

#trees RF
TRE.rf=train(TRE ~ ., data = treTrain,
              method = "rf",
             ntree=501,
             importance=T, 
             metric = "RMSE", maximize= FALSE,
              trControl = oob)
print(TRE.rf)
trerf.test <- predict(TRE.rf, treTest) ## predict test-set
trerf.pred <- predict(t, TRE.rf) ## spatial predictions
plot(varImp(TRE.rf, scale=F), main= "Variable contribution to Tree cover RF")
writeRaster(trerf.pred, filename = "./GH_results/trerfpred.tif")
#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, 
                    repeats = 5, allowParallel = TRUE)

# counts of bare ground (BRG)
BRG.gbm <- train(BRG ~ ., data = brgTrain,
                 method = "gbm",
                 metric= "RMSE",
                 trControl = gbm)
print(BRG.gbm) #check RMSE
brggbm.test <- predict(BRG.gbm, brgTest) ## predict test-set
brggbm.pred <- predict(t, BRG.gbm) ## spatial predictions
plot(varImp(BRG.gbm, scale=F))
# counts of grass
GRS.gbm <- train(GRS ~ ., data = grsTrain,
                 method = "gbm",
                 trControl = gbm)
print(GRS.gbm) #check RMSE
grsgbm.test <- predict(GRS.gbm, grsTest) ## predict test-set
grsgbm.pred <- predict(t, GRS.gbm) ## spatial predictions
plot(varImp(GRS.gbm, scale=F))

#counts for shrubs
SRB.gbm <- train(SRB ~ ., data = srbTrain,
                 method = "gbm",
                 trControl = gbm)
print(SRB.gbm)
srbgbm.test <- predict(SRB.gbm, srbTest) ## predict test-set
srbgbm.pred <- predict(t, SRB.gbm) ## spatial predictions
plot(varImp(SRB.gbm, scale=F))

#counts for Trees
TRE.gbm=train(TRE ~ ., data = treTrain,
              method = "gbm",
              metric= "RMSE",
              trControl = gbm)
print(TRE.gbm)
tregbm.test <- predict(TRE.gbm, treTest) ## predict test-set
tregbm.pred <- predict(t, TRE.gbm) ## spatial predictions
plot(varImp(TRE.gbm, scale=F))



#+ Ensemble predictions <rf>, <gbm>, <dnet> -------------------------------
# Ensemble set-up

pred <- stack(brgglm.pred, brgrf.pred, brggbm.pred, brgdnn.pred)
names(pred) <- c("BRGglm","BRGrf","BRGgbm", "BRGdnn")
geospred <- extract(pred, geofrac)

# count of bare ground
brgens <- cbind.data.frame(BRG, geospred)
brgens <- na.omit(brgens)
brgensTest <- brgens[-brgIndex,] ## replicate previous test set


# Regularized ensemble weighting on the test set <glmnet> for bare ground
# 10-fold CV
ens <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# counts of bare ground

BRG.ens <- train(BRG ~ BRGrf + BRGgbm +BRGdnn + BRGglm, 
                  data = brgensTest,
                 family = "gaussian", 
                 method = "glmnet",
                 metric="RMSE",
                 trControl = ens)
print(BRG.ens) #actually little improvement in RMSE
brg.pred <- predict(BRG.ens, brgensTest)
brg.test <- cbind(brgensTest, brg.pred)

brgens.pred <- predict(pred, BRG.ens) ## spatial prediction

plot(varImp(BRG.ens,scale=F))

# counts of grass 

# Ensemble set-up

predgrs <- stack(grsrf.pred, grsgbm.pred, grsdnn.pred)
names(predgrs) <- c("GRSrf","GRSgbm", "GRSdnn")
geospred <- extract(predgrs, geofrac)

# ensembles of grass cover
grsens <- cbind.data.frame(GRS, geospred)
grsens <- na.omit(grsens)
grsensTest <- grsens[-grsIndex,] ## replicate previous test set


GRS.ens <- train(GRS ~ GRSrf + GRSgbm + GRSdnn, 
                 data = grsensTest,
                 family = "gaussian", 
                 method = "glmnet",
                 trControl = ens)
print(GRS.ens) #actually no improvement in RMSE
grs.pred <- predict(GRS.ens, grsensTest)
grs.test <- cbind(grsensTest, grs.pred)

grsens.pred <- predict(predgrs, GRS.ens) ## spatial prediction

plot(varImp(GRS.ens,scale=F))

# counts of shrubs
# Ensemble set-up
predsrb <- stack(srbrf.pred, srbgbm.pred, srbdnn.pred)
names(predsrb) <- c("SRBrf","SRBgbm", "SRBdnn")
geospred <- extract(predsrb, geofrac)
# ensembles of shrub cover
srbens <- cbind.data.frame(SRB, geospred)
srbens <- na.omit(srbens)
srbensTest <- srbens[-srbIndex,] ## replicate previous test set
SRB.ens <- train(SRB ~ SRBrf + SRBgbm + SRBdnn, 
                 data = srbensTest,
                 family = "gaussian", 
                 method = "glmnet",
                 trControl = ens)
print(SRB.ens) #
srb.pred <- predict(SRB.ens, srbensTest)
srb.test <- cbind(srbensTest, srb.pred)
srbens.pred <- predict(predsrb, SRB.ens) ## spatial prediction
plot(varImp(SRB.ens,scale=F))

# counts of Trees
# Ensemble set-up
predtre <- stack(treglm.pred, trerf.pred, tregbm.pred, trednn.pred)
names(predtre) <- c("TREglm", "TRErf","TREgbm", "TREdnn")
geospred <- extract(predtre, geofrac)
# ensembles of Tree cover
treens <- cbind.data.frame(TRE, geospred)
treens <- na.omit(treens)
treensTest <- treens[-treIndex,] ## replicate previous test set
TRE.ens <- train(TRE ~ TREglm + TRErf + TREgbm + TREdnn, 
                 data = treensTest,
                 family = "gaussian", 
                 method = "glmnet",
                 metric= "RMSE",
                 trControl = ens)
print(TRE.ens) #
tre.pred <- predict(TRE.ens, treensTest)
tre.test <- cbind(treensTest, tre.pred)
treens.pred <- predict(predtre, TRE.ens) ## spatial prediction
plot(varImp(TRE.ens,scale=F))

#+ Plot predictions by GeoSurvey variables ---------------------------------

# bare ground prediction plots
brg.preds <- stack(brgrf.pred,brggbm.pred, brgdnn.pred)
names(brg.preds) <- c("randomForest","gbm", "deepnet")
plot(brg.preds, axes = F)

#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("GH_results", showWarnings=F)

download.file("https://www.dropbox.com/s/j6a444rna9y4usw/GH_cover_wdl_crop.zip?dl=0",
              "./GH_data/GH_cover_wdl_crop.zip", mode="wb")
unzip("./GH_data/GH_cover_wdl_crop.zip", exdir="./GH_data", overwrite = T)

# Export Gtif's to "./GH_results"
writeRaster(brgens.pred, filename="./GH_results/GH_brgpredens.tif", filetype="GTiff", overwrite=T)
writeRaster(grsens.pred, filename="./GH_results/GH_grspredens.tif", filetype="GTiff", overwrite=T)
writeRaster(srbens.pred, filename="./GH_results/GH_srbpredens.tif", filetype="GTiff", overwrite=T)
writeRaster(treens.pred, filename="./GH_results/GH_trepredens.tif", filetype="GTiff", overwrite=T)

#for cropland cover 
grid <- stack(glist)
GH_WDL250=raster("./GH_data/GH_wdl1_250.tif")
GH_CRP250=raster("./GH_data/GH_crp1_250_ens.tif")
GH_WAT250=raster("./GH_results/GH_WATpreds.tif")
GH_HSP250=raster("./GH_results/GH_HSPpreds.tif")
GH_BRG250=raster("./GH_results/GH_brgpredens.tif")
GH_TRE250=raster("./GH_results/GH_trepredens.tif")
GH_GRS250=raster("./GH_results/GH_grspredens.tif")
GH_SRB250=raster("./GH_results/GH_srbpredens.tif")
grid=addLayer(grid, GH_WDL250, GH_WAT250,GH_CRP250,GH_HSP250, GH_BRG250,GH_TRE250,GH_GRS250,
              GH_SRB250)

t=scale(grid, center=TRUE,scale=TRUE) # scale all covariates

#cropland
CRP=round(cropfrac$CRPCV*6.25)
CRPdat=cbind.data.frame(CRP, cropgrid)
CRPdat=na.omit(CRPdat)

cropgrid <- extract(t, cropfrac)

#crops

crpIndex=createDataPartition(CRPdat$CRP, p=2/3, list=FALSE, times=1)
crpTrain=CRPdat[crpIndex,]
crpTest=CRPdat[-crpIndex,]


#glmnet
objControl <- trainControl(method='cv', number=10, allowParallel = TRUE)

CRP.glm=train(CRP ~ ., data=crpTrain, family= "gaussian", method= "glmnet",
              metric= "RMSE",
              trControl=objControl)

plot(varImp(CRP.glm,scale=F), main = "Variable contribution to Crop cover elasic net")
crpglm.pred= predict(t, CRP.glm)

objControl <- trainControl(method='cv', number=10, allowParallel = TRUE)

CRP.rf=train(CRP ~ ., data=crpTrain, family= "gaussian", method= "rf",
              metric= "RMSE", ntree=101,
              trControl=objControl)

croprf.pred= predict(t, CRP.rf)

gbm <- trainControl(method = "repeatedcv", number = 10, 
                    repeats = 5, allowParallel = TRUE)

CRP.gbm=train(CRP ~ ., data=crpTrain, method= "gbm",
              metric= "RMSE",
              trControl=gbm)

crpgbm.pred=predict(t,CRP.gbm)
#deepnet for crop
tc=trainControl(method = "cv", number = 10, allowParallel = TRUE)

CRP.dnn <- train(CRP ~., data=crpTrain, 
                 method = "dnn",
                 # preProc = c("center", "scale"), 
                 trControl = tc,
                 metric= "RMSE",
                 tuneGrid = expand.grid(layer1 = 2:6,
                                        layer2 = 0:3,
                                        layer3 = 0:3,
                                        hidden_dropout = 0,
                                        visible_dropout = 0))

crpdnn.pred= predict(t, CRP.dnn)
# counts of Cropland
# Ensemble set-up
predcrp <- stack(crpglm.pred, croprf.pred, crpgbm.pred, crpdnn.pred)
names(predcrp) <- c("CRPglm", "CRPrf","CRPgbm", "CRPdnn")
geospred <- extract(predcrp, cropfrac)
# ensembles of Crop cover
crpens <- cbind.data.frame(CRP, geospred)
crpens <- na.omit(crpens)
crpensTest <- crpens[-crpIndex,] ## replicate previous test set
CRP.ens <- train(CRP ~ ., 
                 data = crpensTest,
                 family = "gaussian", 
                 method = "glmnet",
                 metric= "RMSE",
                 trControl = ens)
print(CRP.ens) #
crp.pred <- predict(CRP.ens, crpensTest)
crp.test <- cbind(crpensTest, crp.pred)
crpens.pred <- predict(predcrp, CRP.ens) ## spatial prediction
plot(varImp(CRP.ens,scale=F))

writeRaster(crpens.pred, filename="./GH_results/GH_crppredens.tif", filetype="GTiff", overwrite=T)
