#' Ensemble predictions of Ghana GeoSurvey cropland, woody vegetation cover,
#' and rural settlement observations. 
#' Alex Verlinden 2016 after M. Walsh, April 2014
# observations collected by crowdsourcing using "Geosurvey"  in October and November 2015 
# Required packages
# install.packages(c("downloader","raster","doMC",rgdal","caret","randomForest","gbm","nnet","glmnet","dismo")), dependencies=TRUE)
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
require(foreach)

#rm(list=ls())
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("GH_data", showWarnings=F)
dat_dir <- "./GH_data"

# download GeoSurvey data
download.file("https://www.dropbox.com/s/p6god7sqwn4o96b/Ghana_100k.zip?dl=0", "./GH_data/Ghana_100k.zip", mode="wb")
unzip("./GH_data/Ghana_100k.zip", exdir="./GH_data", overwrite=T)
geofrac <- read.table(paste(dat_dir, "/Ghana_100000ha.csv", sep=""), header=T, sep=",")
geofrac <- na.omit(geofrac)

# download Ghana Gtifs (~258 Mb) 250 m and stack in raster
download.file("https://www.dropbox.com/s/7sqifq7bhwh826r/GH_250_grids.zip?dl=0", "./GH_data/GH_250_grids.zip", mode="wb")
unzip("./GH_data/GH_250_grids.zip", exdir="./GH_data", overwrite=T)
glist <- list.files(path="./GH_data", pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid, center=TRUE,scale=TRUE)
#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
geofrac.proj <- as.data.frame(project(cbind(geofrac$Longitude, geofrac$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geofrac.proj) <- c("x","y")
geofrac <- cbind(geofrac, geofrac.proj)
coordinates(geofrac) <- ~x+y
projection(geofrac) <- projection(grid)

# Extract gridded variables at GeoSurvey locations
geosgrid <- extract(t, geofrac)

# Assemble dataframes
#Bare ground (BRG 0-100)
BRG <- round(geofrac$BRG*6.25)
BRGdat <- cbind.data.frame(BRG, geosgrid)
BRGdat <- na.omit(BRGdat)

#Grass
GRS=round(geofrac$GRS*6.25)
GRSdat=cbind.data.frame(GRS, geosgrid)
GRSdat <- na.omit(GRSdat)

#shrubs
SRB=round(geofrac$SHRB*6.25)
SRBdat=cbind.data.frame(SRB, geosgrid)
SRBdat <- na.omit(SRBdat)

#trees
TRE=round(geofrac$TREE*6.25)
TREdat=cbind.data.frame(TRE, geosgrid)
TREdat <- na.omit(TREdat)


# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# bare ground train/test split
brgIndex <- createDataPartition(BRGdat$BRG, p = 0.75, list = FALSE, times = 1)
brgTrain <- BRGdat[ brgIndex,]
brgTest  <- BRGdat[-brgIndex,]

# grass
grsIndex<- createDataPartition(GRSdat$GRS, p = 0.75, list = FALSE, times = 1)
grsTrain <- GRSdat[ grsIndex,]
grsTest  <- GRSdat[-grsIndex,]

# shrubs
srbIndex=createDataPartition(SRBdat$SRB, p = 0.75, list = FALSE, times = 1)
srbTrain <- SRBdat[ srbIndex,]
srbTest  <- SRBdat[-srbIndex,]

#Trees
treIndex=createDataPartition(TREdat$TRE, p = 0.75, list = FALSE, times = 1)
treTrain <- TREdat[ treIndex,]
treTest  <- TREdat[-treIndex,]
#some initial explorations
par(mfrow=c(2,2))
hist(BRG)
hist(TRE)
hist(GRS)
hist(SRB)
par(mfrow=c(1,1))
#test for Poisson model fitting data
BRG.test=glm (BRG~., data=brgTrain, family= "poisson")
1 - pchisq(summary(BRG.test)$deviance,
           summary(BRG.test)$df.residual)
#test for zero inflation
install.packages("pscl")
library(pscl)
BRG.test2=zeroinfl(BRG~.|., data=brgTrain)
cbind(brgTest, 
      Count = predict(BRG.test2, newdata = brgTest, type = "count"),
      Zero = predict(BRG.test2, newdata = brgTest, type = "zero")
)
##We can test for overdispersion in the count part of the zero-inflated model by specifying a negative binomial distribution.
model.zip.3 = zeroinfl(BRG~ .|1, data = crpTrain, dist = "negbin")
summary(model.zip.3)
# this means the model should use zero inflated Poisson distribution ?

#glmnet can only be used after testing for Poisson distribution--------------------------------
#objControl <- trainControl(method='cv', number=10, returnResamp='none')
#glmnet using poisson distribution of Cropland counts 
#CRP.glm=train(CRP ~ ., data=crpTrain, family= "poisson",method="glmnet",metric="RMSE", trControl=objControl)
#crpglm.test <- predict(CRP.glm, crpTest) # predict test set
#crpglm.pred <- predict(t, CRP.glm) ## spatial predictions
#plot(varImp(CRP.glm,scale=F))

# Deep neural net models --------------------------------------------------
# Start foreach to parallelize model fitting  (I do not see improvement)
#detectCores()
registerDoMC(cores=4)
#getDoParWorkers()

#deepnet for bare ground
tc=trainControl(method = "cv", number = 10)

BRG.dnn <- train(BRG ~., data=brgTrain, 
                method = "dnn",
               # preProc = c("center", "scale"), 
                trControl = tc,
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
                 trControl = tc)
                 
print(TRE.dnn)
tre.imp <- varImp(TRE.dnn, useModel = FALSE)
plot(tre.imp, top=27)
trednn.test <- predict(TRE.dnn, treTest) # predict test set
trednn.pred <- predict(t, TRE.dnn)

#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# Bare ground (BRG, 0-100) including Importance vastly increases processing time
BRG.rf <- train(BRG ~ ., data = brgTrain,
                method = "rf", 
                importance=T, 
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
                importance=T, 
                metric = "RMSE", maximize= FALSE,
                trControl = oob)
print(GRS.rf)
grsrf.test <- predict(GRS.rf, grsTest) ## predict test-set
grsrf.pred <- predict(t, GRS.rf) ## spatial predictions
plot(varImp(GRS.rf,scale=F), main="Variable contribution to Grass cover RF")
#RF for shrubs
SRB.rf <- train(SRB ~ ., data = srbTrain,
                method = "rf", 
                importance=T, 
                metric = "RMSE", maximize= FALSE,
                trControl = oob)
print(SRB.rf) #check RMSE
srbrf.test <- predict(SRB.rf, srbTest) ## predict test-set
srbrf.pred <- predict(t, SRB.rf) ## spatial predictions
plot(varImp(SRB.rf,scale=F), main="Variable contribution to Shrub cover RF")

#trees RF
TRE.rf=train(TRE ~ ., data = treTrain,
              method = "rf",
             importance=T, 
             metric = "RMSE", maximize= FALSE,
              trControl = oob)
print(TRE.rf)
trerf.test <- predict(TRE.rf, treTest) ## predict test-set
trerf.pred <- predict(t, TRE.rf) ## spatial predictions
plot(varImp(TRE.rf, scale=F), main= "Variable contribution to Tree cover RF")

#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# counts of bare ground (BRG)
BRG.gbm <- train(BRG ~ ., data = brgTrain,
                 method = "gbm",
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
              trControl = gbm)
print(TRE.gbm)
tregbm.test <- predict(TRE.gbm, treTest) ## predict test-set
tregbm.pred <- predict(t, TRE.gbm) ## spatial predictions
plot(varImp(TRE.gbm, scale=F))



#+ Ensemble predictions <rf>, <gbm>, <dnet> -------------------------------
# Ensemble set-up

pred <- stack(brgrf.pred, brggbm.pred, brgdnn.pred)
names(pred) <- c("BRGrf","BRGgbm", "BRGdnn")
geospred <- extract(pred, geofrac)

# count of bare ground
brgens <- cbind.data.frame(BRG, geospred)
brgens <- na.omit(brgens)
brgensTest <- brgens[-brgIndex,] ## replicate previous test set


# Regularized ensemble weighting on the test set <glmnet> for bare ground
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# counts of bare ground

BRG.ens <- train(BRG ~ BRGrf + BRGgbm +BRGdnn, 
                  data = brgensTest,
                 family = "gaussian", 
                 method = "glmnet",
                 trControl = ens)
print(BRG.ens) #actually no improvement in RMSE
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
predtre <- stack(trerf.pred, tregbm.pred, trednn.pred)
names(predtre) <- c("TRErf","TREgbm", "TREdnn")
geospred <- extract(predtre, geofrac)
# ensembles of shrub cover
treens <- cbind.data.frame(TRE, geospred)
treens <- na.omit(treens)
treensTest <- treens[-treIndex,] ## replicate previous test set
TRE.ens <- train(TRE ~ TRErf + TREgbm + TREdnn, 
                 data = treensTest,
                 family = "gaussian", 
                 method = "glmnet",
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

# Export Gtif's to "./GH_results"
writeRaster(brgens.pred, filename="./GH_results/GH_brgpredens.tif", filetype="GTiff", overwrite=T)
writeRaster(grsens.pred, filename="./GH_results/GH_grspredens.tif", filetype="GTiff", overwrite=T)
writeRaster(srbens.pred, filename="./GH_results/GH_srbpredens.tif", filetype="GTiff", overwrite=T)
writeRaster(treens.pred, filename="./GH_results/GH_trepredens.tif", filetype="GTiff", overwrite=T)
