#' Prediction and local elastic net stacking of Namibia 10 k GeoSurvey woody cover classes predictions with additional Namibia 30 k GeoSurvey test data.
#' Modified by Alex Verlinden April 2016 after M.Walsh & J.Chen, April 2015
# This code models multiclasses of woody cover A=0 , B=0-10%, C= 10-30%, D=30-60%, E>60%
#+ Required packages
# install.packages(c("downloader","raster","rgdal","dismo","caret","glmnet", "DoMC")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(glmnet)
require(randomForest)
require(foreach)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("NAM_bush", showWarnings=F)
dat_dir <- "./NAM_bush"

# download GeoSurvey Namibia data
download.file("https://www.dropbox.com/s/fopvazaeathk941/NAM_woody_class2015_4cl.csv?dl=0", "./NAM_bush/NAM_woody_class2015_4cl.csv", mode="wb")
bushclass <- read.table(paste(dat_dir, "/NAM_woody_class2015_4cl.csv", sep=""), header=T, sep=",")
bushclass <- na.omit(bushclass)
# download Namibia 250 m Gtifs (~ Mb) and stack in raster
download.file("https://www.dropbox.com/s/8mpat9mohyst1fj/NAMGRIDS_250.zip?dl=0", "./NAM_bush/NAMGRIDS_250.zip", mode="wb")
unzip("./NAM_bush/NAMGRIDS_250.zip", exdir="./NAM_bush", overwrite=T)
glist <- list.files(path="./NAM_bush", pattern="tif", full.names=T)
grid <- stack(glist)
#scaling the grids
t=scale(grid, center=TRUE, scale=TRUE)
#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
bush.proj <- as.data.frame(project(cbind(bushclass$Longitude, bushclass$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(bush.proj) <- c("x","y")
coordinates(bush.proj) <- ~x+y
projection(bush.proj) <- projection(grid)

# Extract gridded variables at GeoSurvey test data locations (n~26k)
bushgrid=extract(t, bush.proj)

# Assemble dataframes

# woody plant classes
bush0=bushclass$CLASS
bush0dat <- cbind.data.frame(bush0, bushgrid)
bush0dat <- na.omit(bush0dat)

# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------

bush0Index=createDataPartition(bush0dat$bush0, p = 0.75, list = FALSE, times = 1)
bush0Train <- bush0dat[ bush0Index,]
bush0Test  <- bush0dat[-bush0Index,]


#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# use all cores (workers) 
library(doMC)
registerDoMC(cores=4)

# random Forest
bush.rf <- train(bush0 ~ ., data = bush0Train,
                method = "rf", metric= "Accuracy", importance=T,
                trControl = oob)
bushrf.test <- predict(bush.rf, bush0Test) ## predict test-set
confusionMatrix(bushrf.test, bush0Test$bush0) ## print validation summaries
bushrf.pred <- predict(t, bush.rf) ## spatial predictions
plot(bushrf.pred)

#importance of variables to RF
imprf=varImp(bush.rf)
plot(imprf, main = "Variables contribution to the RF regression, cover classes")

#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE)

# gradient boosting
bush.gbm <- train(bush0 ~ ., data = bush0Train,
                 method = "gbm", metric="Accuracy",
                 trControl = gbm )
bushgbm.test <- predict(bush.gbm, bush0Test) ## predict test-set
confusionMatrix(bushgbm.test, bush0Test$bush0) ## print validation summaries
bushgbm.pred=predict(t, bush.gbm)
#variable importance
impgbm=varImp(bush.gbm)
plot(impgbm, main = "Variables contribution to GBM cover classes")

#deepnet
tc <- trainControl(method = "cv", number = 5)

bush.dnn <- train(bush0 ~ ., data = bush0Train, 
                method = "dnn", 
                metric="Accuracy", 
                trControl = tc,
                tuneGrid = expand.grid(layer1 = 0:12,
                                       layer2 = 0:3,
                                       layer3 = 0:3,
                                       hidden_dropout = 0,
                                       visible_dropout = 0))
bushdnn.test <- predict(bush.dnn, bush0Test) ## predict test-set
confusionMatrix(bushdnn.test, bush0Test$bush0) ## print validation summaries
bushdnn.pred=predict(t, bush.dnn)
impdnn=varImp(bush.dnn)
plot(impgbm, main = "Variables contribution to DNN cover classes")


#+ Plot predictions by GeoSurvey variables ---------------------------------
# bush classes
bush.preds <- stack(bushrf.pred, bushgbm.pred, bushdnn.pred)
names(bush.preds) <- c("randomForest","gradientboosting","deepnet")
plot(bush.preds, axes = F)

bushpred=extract(bush.preds, bush.proj)
# 
bushens <- cbind.data.frame(bush0, bushpred)
bushens <- na.omit(bushens)
bushensTest <- bushens[-bush0Index,] ## replicate previous test set

# Regularized ensemble weighting on the test set
# 5-fold CV
ens <- trainControl(method = "cv", number = 5)

# ensembles of 3 classifiers
bush.ens <- train(bush0 ~ randomForest + gradientboosting + deepnet, data = bushensTest,
                 family="binomial",
                 method = "rf",
                 trControl = ens)

bushens.pred <- predict(bush.preds, bush.ens) ## spatial prediction
plot(bushens.pred, main= "Ensemble of low,medium and high woody plant cover", cex.main=0.8)
bushens.test <- predict(bush.ens, bushensTest) 
confusionMatrix(bushens.test, bushensTest$bush0) ## print validation summaries

impens=varImp(bush.ens)
plot(impens, main = "Regression contribution to Ensemble, for classes")


#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("NAM_results", showWarnings=F)

# Export Gtif's to "./NAM_results"

#write tiff
rf=writeRaster(bushrf.pred,filename="./NAM_Results/bushrf4cl.tif", format= "GTiff", overwrite = TRUE)
rf=writeRaster(bushens.pred,filename="./NAM_Results/bushens4cl.tif", format= "GTiff", overwrite = TRUE)
