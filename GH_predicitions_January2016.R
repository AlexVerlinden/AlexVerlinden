#' Ensemble predictions of Ghana GeoSurvey cropland, woody vegetation cover,
#' and rural settlement observations. 
#' Alex Verlinden 2016 after M. Walsh, April 2014
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

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("GH_data", showWarnings=F)
dat_dir <- "./GH_data"

# download GeoSurvey data
download.file("https://www.dropbox.com/s/5q9twth406rx3yi/GH_survey_20000.csv?dl=0", "./GH_data/GH_survey_20000.csv?", mode="wb")
geos <- read.table(paste(dat_dir, "/GH_survey_20000.csv?", sep=""), header=T, sep=",")
geos <- na.omit(geos)

# download Ghana Gtifs (~8 Mb) and stack in raster
download.file("https://www.dropbox.com/s/3eo70huv7s8d9e5/GH_preds_1km.zip?dl=0", "./GH_data/GH_preds_1km.zip", mode="wb")
unzip("./GH_data/GH_preds_1km.zip", exdir="./GH_data", overwrite=T)
glist <- list.files(path="./GH_data", pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
geos.proj <- as.data.frame(project(cbind(geos$Longitude, geos$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geos.proj) <- c("x","y")
geos <- cbind(geos, geos.proj)
coordinates(geos) <- ~x+y
projection(geos) <- projection(grid)

# Extract gridded variables at GeoSurvey locations
geosgrid <- extract(grid, geos)

# Assemble dataframes
# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP <- geos$CRP
crpdat <- cbind.data.frame(CRP, geosgrid)
crpdat <- na.omit(crpdat)

# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)
WCP <- geos$WDL
wcpdat <- cbind.data.frame(WCP, geosgrid)
wcpdat <- na.omit(wcpdat)

# presence/absence of Buildings/Human Settlements (HSP, present = Y, absent = N)
# note that this excludes large urban areas where MODIS fPAR = 0
HSP <- geos$BLD
hspdat <- cbind.data.frame(HSP, geosgrid)
hspdat <- na.omit(hspdat)

# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Cropland train/test split
crpIndex <- createDataPartition(crpdat$CRP, p = 0.75, list = FALSE, times = 1)
crpTrain <- crpdat[ crpIndex,]
crpTest  <- crpdat[-crpIndex,]

# Woody cover train/test split
wcpIndex <- createDataPartition(wcpdat$WCP, p = 0.75, list = FALSE, times = 1)
wcpTrain <- wcpdat[ wcpIndex,]
wcpTest  <- wcpdat[-wcpIndex,]

# Settlement train/test split
hspIndex <- createDataPartition(hspdat$HSP, p = 0.75, list = FALSE, times = 1)
hspTrain <- hspdat[ hspIndex,]
hspTest  <- hspdat[-hspIndex,]

#glmnet--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none')
#glmnet using binomial distribution presence/absence of Cropland (CRP, present = Y, absent = N)
CRP.glm=train(CRP ~ ., data=crpTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)
crpglm.test <- predict(CRP.glm, crpTest) # predict test set
confusionMatrix(crpglm.test, crpTest$CRP, "Y") ## print validation summaries
crpglm.pred <- predict(grid, CRP.glm, type="prob") ## spatial predictions
plot(varImp(CRP.glm,scale=F))

# Deep neural net models --------------------------------------------------
# Start foreach to parallelize model fitting
mc <- makeCluster(detectCores())
registerDoParallel(mc)
#deepnet
tc=trainControl(method = "cv", number = 10)

CRP.dnn <- train(CRP ~., data=crpTrain, 
                method = "dnn", 
                preProc = c("center", "scale"), 
                trControl = tc,
                tuneGrid = expand.grid(layer1 = 2:6,
                                       layer2 = 0:3,
                                       layer3 = 0:3,
                                       hidden_dropout = 0,
                                       visible_dropout = 0))
print(CRP.dnn)
CRP.imp <- varImp(CRP.dnn, useModel = FALSE)
plot(CRP.imp, top=27)
crpdnn.test <- predict(CRP.dnn, crpTest) # predict test set
confusionMatrix(crpdnn.test, crpTest$CRP, "Y") ## print validation summaries
crpdnn.pred <- predict(grid, CRP.dnn, type="prob")


#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP.rf <- train(CRP ~ ., data = crpTrain,
                method = "rf",
                trControl = oob)
crprf.test <- predict(CRP.rf, crpTest) ## predict test-set
confusionMatrix(crprf.test, crpTest$CRP, "Y") ## print validation summaries
crprf.pred <- predict(grid, CRP.rf, type = "prob") ## spatial predictions
plot(varImp(CRP.rf,scale=F))

# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)
WCP.rf <- train(WCP ~ ., data = wcpTrain,
                method = "rf",
                trControl = oob)
wcprf.test <- predict(WCP.rf, wcpTest) ## predict test-set
confusionMatrix(wcprf.test, wcpTest$WCP, "Y") ## print validation summaries
wcprf.pred <- predict(grid, WCP.rf, type = "prob") ## spatial predictions
plot(varImp(WCP.rf,scale=F))

# presence/absence of Buildings/Human Settlements (HSP, present = Y, absent = N)
HSP.rf <- train(HSP ~ ., data = hspTrain,
                method = "rf",
                trControl = oob)
hsprf.test <- predict(HSP.rf, hspTest) ## predict test-set
confusionMatrix(hsprf.test, hspTest$HSP, "Y") ## print validation summaries
hsprf.pred <- predict(grid, HSP.rf, type = "prob") ## spatial predictions
plot(varImp(HSP.rf,scale=F))

#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP.gbm <- train(CRP ~ ., data = crpTrain,
                 method = "gbm",
                 trControl = gbm)
crpgbm.test <- predict(CRP.gbm, crpTest) ## predict test-set
confusionMatrix(crpgbm.test, crpTest$CRP, "Y") ## print validation summaries
crpgbm.pred <- predict(grid, CRP.gbm, type = "prob") ## spatial predictions

# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)
WCP.gbm <- train(WCP ~ ., data = wcpTrain,
                 method = "gbm",
                 trControl = gbm)
wcpgbm.test <- predict(WCP.gbm, wcpTest) ## predict test-set
confusionMatrix(wcpgbm.test, wcpTest$WCP, "Y") ## print validation summaries
wcpgbm.pred <- predict(grid, WCP.gbm, type = "prob") ## spatial predictions

# presence/absence of Buildings/Human Settlements (HSP, present = Y, absent = N)
HSP.gbm <- train(HSP ~ ., data = hspTrain,
                 method = "gbm",
                 trControl = gbm)
hspgbm.test <- predict(HSP.gbm, hspTest) ## predict test-set
confusionMatrix(hspgbm.test, hspTest$HSP, "Y") ## print validation summaries
hspgbm.pred <- predict(grid, HSP.gbm, type = "prob") ## spatial predictions

#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP.nn <- train(CRP ~ ., data = crpTrain,
                method = "nnet",
                trControl = nn)
crpnn.test <- predict(CRP.nn, crpTest) ## predict test-set
confusionMatrix(crpnn.test, crpTest$CRP, "Y") ## print validation summaries
crpnn.pred <- predict(grid, CRP.nn, type = "prob") ## spatial predictions

# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)
WCP.nn <- train(WCP ~ ., data = wcpTrain,
                method = "nnet",
                trControl = nn)
wcpnn.test <- predict(WCP.nn, wcpTest) ## predict test-set
confusionMatrix(wcpnn.test, wcpTest$WCP, "Y") ## print validation summaries
wcpnn.pred <- predict(grid, WCP.nn, type = "prob") ## spatial predictions

# presence/absence of Buildings/Human Settlements (HSP, present = Y, absent = N)
HSP.nn <- train(HSP ~ ., data = hspTrain,
                method = "nnet",
                trControl = nn)
hspnn.test <- predict(HSP.nn, hspTest) ## predict test-set
confusionMatrix(hspnn.test, hspTest$HSP, "Y") ## print validation summaries
hspnn.pred <- predict(grid, HSP.nn, type = "prob") ## spatial predictions

#+ Ensemble predictions <rf>, <gbm>, <nnet> -------------------------------
# Ensemble set-up

pred <- stack(1-crprf.pred, 1-crpgbm.pred, 1-crpnn.pred, 1-crpglm.pred, 1-crpdnn.pred,
              1-wcprf.pred, 1-wcpgbm.pred, 1-wcpnn.pred,
              1-hsprf.pred, 1-hspgbm.pred, 1-hspnn.pred)
names(pred) <- c("CRPrf","CRPgbm","CRPnn", "CRPglm", "CRPdnn",
                 "WCPrf","WCPgbm","WCPnn",
                 "HSPrf","HSPgbm","HSPnn")
geospred <- extract(pred, geos)

# presence/absence of Cropland (CRP, present = Y, absent = N)
crpens <- cbind.data.frame(CRP, geospred)
crpens <- na.omit(crpens)
crpensTest <- crpens[-crpIndex,] ## replicate previous test set

# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)
wcpens <- cbind.data.frame(WCP, geospred)
wcpens <- na.omit(wcpens)
wcpensTest <- wcpens[-wcpIndex,] ## replicate previous test set

# presence/absence of Buildings/Human Settlements (HSP, present = Y, absent = N)
hspens <- cbind.data.frame(HSP, geospred)
hspens <- na.omit(hspens)
hspensTest <- hspens[-hspIndex,] ## replicate previous test set

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP.ens <- train(CRP ~ CRPrf + CRPgbm + CRPnn + CRPglm +CRPdnn, data = crpensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
crpens.test <- predict(CRP.ens, crpensTest,  type="prob") ## predict test-set
confusionMatrix(crpens.test, crpTest$CRP, "Y") ## print validation summaries

CRP.ens <- train(CRP ~ CRPrf + CRPgbm + CRPnn + CRPglm, data = crpensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
crp.pred <- predict(CRP.ens, crpensTest, type="prob")

crp.test <- cbind(crpensTest, crp.pred)
crp <- subset(crp.test, CRP=="Y", select=c(Y))
cra <- subset(crp.test, CRP=="N", select=c(Y))
crp.eval <- evaluate(p=crp[,1], a=cra[,1]) ## calculate ROC's on test set <dismo>
crp.eval
plot(crp.eval, 'ROC') ## plot ROC curve
crp.thld <- threshold(crp.eval, 'spec_sens') ## TPR+TNR threshold for classification
crpens.pred <- predict(pred, CRP.ens, type="prob") ## spatial prediction
plot(1-crpens.pred, axes=F)
crpmask <- 1-crpens.pred > crp.thld
plot(crpmask, axes = F, legend = F)
plot(varImp(CRP.ens,scale=F))

# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)
WCP.ens <- train(WCP ~ WCPrf + WCPgbm + WCPnn, data = wcpensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
wcp.pred <- predict(WCP.ens, wcpensTest, type="prob")
wcp.test <- cbind(wcpensTest, wcp.pred)
wcp <- subset(wcp.test, WCP=="Y", select=c(Y))
wca <- subset(wcp.test, WCP=="N", select=c(Y))
wcp.eval <- evaluate(p=wcp[,1], a=wca[,1]) ## calculate ROC's on test set <dismo>
wcp.eval
plot(wcp.eval, 'ROC') ## plot ROC curve
wcp.thld <- threshold(wcp.eval, 'spec_sens') ## TPR+TNR threshold for classification
wcpens.pred <- predict(pred, WCP.ens, type="prob") ## spatial prediction
plot(1-wcpens.pred, axes=F)
wcpmask <- 1-wcpens.pred > wcp.thld
plot(wcpmask, axes = F, legend = F)
plot(varImp(WCP.ens,scale=F))

# presence/absence of Buildings/Rural Settlements (HSP, present = Y, absent = N)
HSP.ens <- train(HSP ~ HSPrf + HSPgbm + HSPnn, data = hspensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
hsp.pred <- predict(HSP.ens, hspensTest, type="prob")
hsp.test <- cbind(hspensTest, hsp.pred)
hsp <- subset(hsp.test, HSP=="Y", select=c(Y))
hsa <- subset(hsp.test, HSP=="N", select=c(Y))
hsp.eval <- evaluate(p=hsp[,1], a=hsa[,1]) ## calculate ROC's on test set <dismo>
hsp.eval
plot(hsp.eval, 'ROC') ## plot ROC curve
hsp.thld <- threshold(hsp.eval, 'spec_sens') ## TPR+TNR threshold for classification
hspens.pred <- predict(pred, HSP.ens, type="prob") ## spatial prediction
plot(1-hspens.pred, axes=F)
hspmask <- 1-hspens.pred > hsp.thld
plot(hspmask, axes = F, legend = F)
plot(varImp(HSP.ens,scale=F))

#+ Plot predictions by GeoSurvey variables ---------------------------------
# Cropland prediction plots
crp.preds <- stack(1-crprf.pred, 1-crpgbm.pred, 1-crpnn.pred, 1-crpglm.pred,1-crpens.pred)
names(crp.preds) <- c("randomForest","gbm","nnet", "glm", "Ensemble")
plot(crp.preds, axes = F)

# Woody vegetation cover >60% prediction plots
wcp.preds <- stack(1-wcprf.pred, 1-wcpgbm.pred, 1-wcpnn.pred, 1-wcpens.pred)
names(wcp.preds) <- c("randomForest","gbm","nnet","Ensemble")
plot(wcp.preds, axes = F)

# Rural settlement prediction plots
hsp.preds <- stack(1-hsprf.pred, 1-hspgbm.pred, 1-hspnn.pred, 1-hspens.pred)
names(hsp.preds) <- c("randomForest","gbm","nnet","Ensemble")
plot(hsp.preds, axes = F)

#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("GH_results", showWarnings=F)

# Export Gtif's to "./GH_results"
writeRaster(crp.preds, filename="./GH_results/GH_crpreds.tif", datatype="FLT4S", options="INTERLEAVE=BAND", overwrite=T)
writeRaster(wcp.preds, filename="./GH_results/GH_wcpreds.tif", datatype="FLT4S", options="INTERLEAVE=BAND", overwrite=T)
writeRaster(hsp.preds, filename="./GH_results/GH_hspreds.tif", datatype="FLT4S", options="INTERLEAVE=BAND", overwrite=T)
