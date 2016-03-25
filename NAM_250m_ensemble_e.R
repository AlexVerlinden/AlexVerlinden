##' Prediction and local elastic net stacking of Namibia 30 k GeoSurvey woody cover classes predictions with additional Namibia 30 k GeoSurvey test data.
#' Modified by Alex Verlinden September 2015   after M.Walsh & J.Chen, April 2015

#+ Required packages
# install.packages(c("downloader","raster","rgdal","dismo","caret","glmnet")), dependencies=TRUE)

require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(glmnet)
require(randomForest)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("NAM_bush", showWarnings=F)
dat_dir <- "./NAM_bush"
# download GeoSurvey Namibia data
download.file("https://www.dropbox.com/s/ptqj73kpcoy71ey/NAM_woody_cover2015_alex.csv?dl=0", "./NAM_bush/NAM_woody_cover2015_alex.csv", mode="wb")
bush <- read.table(paste(dat_dir, "/NAM_woody_cover2015_alex.csv", sep=""), header=T, sep=",")
bush <- na.omit(bush)
# download Namibia 250 m Gtifs (~ 320 Mb) and stack in raster
download.file("https://www.dropbox.com/s/8mpat9mohyst1fj/NAMGRIDS_250.zip?dl=0", "./NAM_bush/NAMGRIDS_250.zip", mode="wb")
unzip("./NAM_bush/NAMGRIDS_250.zip", exdir="./NAM_bush", overwrite=T)
#list covariates and stack in grid
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)
#scale covariates
t=scale(grid, center=TRUE, scale=TRUE)
#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
bush.proj <- as.data.frame(project(cbind(bush$Longitude, bush$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(bush.proj) <- c("x","y")
bush <- cbind(bush, bush.proj)
coordinates(bush) <- ~x+y
projection(bush) <- projection(grid)
#extract values at geosurvey locations
bushgrid=extract(t, bush)
# Assemble dataframes
# presence/absence of very high bush cover (60, present = Y, absent = N)
bush60 <- bush$S.T.60
bush6dat <- cbind.data.frame(bush60, bushgrid)
bush6dat <- na.omit(bush6dat)
# presence/absence of high bush cover (30-60, present = Y, absent = N)
bush30 <- bush$S.T_30.60
bush3dat <- cbind.data.frame(bush30, bushgrid)
bush3dat <- na.omit(bush3dat)
# presence/absence of medium bush cover (10-30, present = Y, absent = N)
bush15 <- bush$S.T_10.30
bush15dat <- cbind.data.frame(bush15, bushgrid)
bush15dat <- na.omit(bush15dat)
# presence/absence of low bush cover (<10, present = Y, absent = N)
bush10=bush$S.T.10
bush10dat <- cbind.data.frame(bush10, bushgrid)
bush10dat <- na.omit(bush10dat)
# presence/absence of woody plants
bush0=bush$S.T_pres
bush0dat <- cbind.data.frame(bush0, bushgrid)
bush0dat <- na.omit(bush0dat)
# set train/test set randomization seed
seed <- 12345
set.seed(seed)
#+ Split data into train and test sets ------------------------------------
# very high cover bushland train/test split
bush6Index <- createDataPartition(bush6dat$bush60, p = 0.75, list = FALSE, times = 1)
bush6Train <- bush6dat[ bush6Index,]
bush6Test  <- bush6dat[-bush6Index,]
bush3Index=createDataPartition(bush3dat$bush30, p = 0.75, list = FALSE, times = 1)
bush3Train <- bush3dat[ bush3Index,]
bush3Test  <- bush3dat[-bush3Index,]
bush15Index=createDataPartition(bush15dat$bush15, p = 0.75, list = FALSE, times = 1)
bush15Train <- bush15dat[ bush15Index,]
bush15Test  <- bush15dat[-bush15Index,]
bush1Index=createDataPartition(bush10dat$bush10, p = 0.75, list = FALSE, times = 1)
bush1Train <- bush10dat[ bush1Index,]
bush1Test  <- bush10dat[-bush1Index,]
bush0Index=createDataPartition(bush0dat$bush0, p = 0.75, list = FALSE, times = 1)
bush0Train <- bush0dat[ bush0Index,]
bush0Test  <- bush0dat[-bush0Index,]
oob <- trainControl(method = "oob")
# presence/absence of bushland (>60%, present = Y, absent = N)
bush6.rf <- train(bush60 ~ ., data = bush6Train,
method = "rf",
trControl = oob)
bushrf.test <- predict(bush6.rf, bush6Test, type="prob") ## predict test-set
confusionMatrix(bushrf.test, bush6Test$bush60, "Y") ## print validation summaries
bushrf.pred <- predict(t, bush6.rf, type = "prob") ## spatial predictions
plot(1-bushrf.pred)
#importance of variables to RF
imprf=varImp(bush6.rf)
plot(imprf, main = "Variables contribution to the RF regression (expert), cover >60%")
#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
# presence/absence of >60% bush cover (bush60, present = Y, absent = N)
bush6.gbm <- train(bush60 ~ ., data = bush6Train,
method = "gbm",
trControl = gbm)
bush6gbm.test <- predict(bush6.gbm, bush6Test) ## predict test-set
confusionMatrix(bush6gbm.test, bush6Test$bush60, "Y") ## print validation summaries
bush6gbm.pred=predict(t, bush6.gbm, type = "prob")
#variable importance
impgbm=varImp(bush6.gbm)
plot(impgbm, main = "Variables contribution to GBM (expert) cover >60%")
# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)
#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)
# presence/absence of >60% bush(bush60, present = Y, absent = N)
bush60.nn <- train(bush60 ~ ., data = bush6Train,
method = "nnet",
trControl = nn)
bush60nn.test <- predict(bush60.nn, bush6Test) ## predict test-set
confusionMatrix(bush60nn.test, bush6Test$bush60, "Y") ## print validation summaries
bush60nn.pred <- predict(t, bush60.nn, type = "prob") ## spatial predictions
#variables contribution
impnn=varImp(bush60.nn)
plot(impnn, main = "Variables contribution to Neural Net (expert) cover >60%")
#+ Plot predictions by GeoSurvey variables ---------------------------------
# bush encroachment >60 % prediction plots
bush60.preds <- stack(1-bushrf.pred, 1-bush6gbm.pred, 1-bush60nn.pred)
names(bush60.preds) <- c("randomForest","gradient boosting","neural net")
plot(bush60.preds, axes = F)
bush60pred=extract(bush60.preds, bush)
# presence/absence of bush>60 % (bush>60%, present = Y, absent = N)
bushens <- cbind.data.frame(bush60, bush60pred)
bushens <- na.omit(bushens)
bushensTest <- bushens[-bush6Index,] ## replicate previous test set
# Regularized ensemble weighting on the test set <glm>
# 5-fold CV
ens <- trainControl(method = "cv", number = 5)
# presence/absence of bushland (bush 60, present = Y, absent = N)
bush60.ens <- train(bush60 ~ randomForest + gradient.boosting + neural.net, data = bushensTest,
family = "binomial",
method = "glmnet",
trControl = ens)
bushens.pred <- predict(bush60.ens, bushensTest, type="prob")
bush.test <- cbind(bushensTest, bushens.pred)
bushp <- subset(bush.test, bush60=="Y", select=c(Y))
busha <- subset(bush.test, bush60=="N", select=c(Y))
bush.eval <- evaluate(p=bushp[,1], a=busha[,1]) ## calculate ROC's on test set <dismo>
bush.eval
plot(bush.eval, 'ROC') ## plot ROC curve
bush.thld <- threshold(bush.eval, 'spec_sens') ## TPR+TNR threshold for classification
bushens.pred <- predict(bush60.preds, bush60.ens, type="prob")
bush60ens=1-bushens.pred
plot(bush60ens)
bushmask=1-bushens.pred>bushens.thld

impens=varImp(bush60.ens)
plot(impens, main = "Regression contribution to Ensemble (expert) cover >60%")
dir.create("NAM_results", showWarnings=F)
rf=writeRaster(1-bushens.pred,filename="./NAM_Results/bush_60%_pred.tif", format= "GTiff", overwrite = TRUE)
rf=writeRaster(bushmask,filename="./NAM_Results/bush_60%.tif", format= "GTiff", overwrite = TRUE)
# presence/absence of bushland (>30%-60%, present = Y, absent = N)
bush3.rf <- train(bush30 ~ ., data = bush3Train,method = "rf",trControl = oob)
bushrf3.test <- predict(bush3.rf, bush3Test) ## predict test-set

confusionMatrix(bushrf3.test, bush3Test$bush30, "Y") ## print validation summaries
bushrf3.pred <- predict(t, bush3.rf, type = "prob") ## spatial predictions

imprf=varImp(bush3.rf)
plot(imprf, main = "Variable contribution to Random Forest (expert), 30-60%")
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
# presence/absence of 30-60% bush cover (bush30, present = Y, absent = N)
bush3.gbm <- train(bush30 ~ ., data = bush3Train, method = "gbm", trControl = gbm)
bush3gbm.test <- predict(bush3.gbm, bush3Test) ## predict test-set
confusionMatrix(bush3gbm.test, bush3Test$bush30, "Y") ## print validation summaries
bush3gbm.pred=predict(t, bush3.gbm, type = "prob")
impgbm=varImp(bush3.gbm)
plot(impgbm, main = "Variable contribution to Gradient Boosting (expert), 30-60%")
# presence/absence of Woody Vegetation Cover 30-60% (bush30, present = Y, absent = N)
#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)
# presence/absence of 30-60% bush(bush30, present = Y, absent = N)
bush30.nn <- train(bush30 ~ ., data = bush3Train,
method = "nnet",
trControl = nn)
bush30nn.test <- predict(bush30.nn, bush3Test) ## predict test-set
confusionMatrix(bush30nn.test, bush3Test$bush30, "Y") ## print validation summaries
bush30nn.pred <- predict(grid, bush30.nn, type = "prob") ## spatial predictions
#variable contribution
impnn=varImp(bush30.nn)
plot(impnn, main = "Variable contribution to Neural Net (expert), 30-60%")
# presence/absence of Woody Vegetation Cover 30-60% (bush30, present = Y, absent = N)
#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)
# presence/absence of 30-60% bush(bush30, present = Y, absent = N)
bush30.nn <- train(bush30 ~ ., data = bush3Train,
method = "nnet",
trControl = nn)
bush30nn.test <- predict(bush30.nn, bush3Test) ## predict test-set
confusionMatrix(bush30nn.test, bush3Test$bush30, "Y") ## print validation summaries
bush30nn.pred <- predict(t, bush30.nn, type = "prob") ## spatial predictions
#variable contribution
impnn=varImp(bush30.nn)
plot(impnn, main = "Variable contribution to Neural Net (expert), 30-60%")
bush30.preds <- stack(1-bushrf3.pred, 1-bush3gbm.pred, 1-bush30nn.pred)
names(bush30.preds) <- c("randomForest","gradient.boosting","neural.net")
plot(bush30.preds, axes = F)
bush30pred=extract(bush30.preds, bush)
# presence/absence of bush> 30-60 % (bush30-60%, present = Y, absent = N)
bushens <- cbind.data.frame(bush30, bush30pred)
bushens <- na.omit(bushens)
bushensTest <- bushens[-bush3Index,] ## replicate previous test set
# Regularized ensemble weighting on the test set <glm>
# 5-fold CV
ens <- trainControl(method = "cv", number = 5)
# presence/absence of bushland (bush 30-60%, present = Y, absent = N)
bush30.ens <- train(bush30 ~ randomForest + gradient.boosting + neural.net, data = bushensTest,
family = "binomial",
method = "glmnet",
trControl = ens)
bushens30.pred <- predict(bush30.ens, bushensTest, type="prob")
bush.test <- cbind(bushensTest, bushens30.pred)
bushp <- subset(bush.test, bush30=="Y", select=c(Y))
busha <- subset(bush.test, bush30=="N", select=c(Y))
bush.eval <- evaluate(p=bushp[,1], a=busha[,1]) ## calculate ROC's on test set <dismo>
bush.eval
plot(bush.eval, 'ROC') ## plot ROC curve
bush.thld <- threshold(bush.eval, 'spec_sens') ## TPR+TNR threshold for classification
bushens.pred <- predict(bush30.preds, bush30.ens, type="prob") ## spatial prediction
bush30ens=1-bushens.pred
plot(bush30ens)
rf=writeRaster(bush30ens,filename="./NAM_results/bush_30%.tif", format= "GTiff", overwrite = TRUE)
# presence/absence of bushland (10%-30%, present = Y, absent = N)
bush15.rf <- train(bush15 ~ ., data = bush15Train,
method = "rf",
trControl = oob)
bush15rf.test <- predict(bush15.rf, bush15Test) ## predict test-set
confusionMatrix(bush15rf.test, bush15Test$bush15, "Y") ## print validation summaries
bush15rf.pred <- predict(t, bush15.rf, type = "prob") ## spatial predictions
#variable contribution to RF 10-30%
imprf=varImp(bush15.rf)
plot(imprf, main = "Variable contribution to Random Forest (expert), 10-30%")
#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
# presence/absence of 10-30% bush cover (bush15, present = Y, absent = N)
bush15.gbm <- train(bush15 ~ ., data = bush15Train,
method = "gbm",
trControl = gbm)
bush15gbm.test <- predict(bush15.gbm, bush15Test) ## predict test-set
confusionMatrix(bush15gbm.test, bush15Test$bush15, "Y") ## print validation summaries
bush15gbm.pred=predict(t, bush15.gbm, type = "prob")
impgbm=varImp(bush15.gbm)
plot(impgbm, main = "Variable contribution to Gradient Boosting (expert), 10-30%")
# presence/absence of Woody Vegetation Cover 10-30% (bush15, present = Y, absent = N)
#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)
# presence/absence of 10-30% bush(bush15, present = Y, absent = N)
bush15.nn <- train(bush15 ~ ., data = bush15Train, method = "nnet", trControl = nn)
bush15nn.test <- predict(bush15.nn, bush15Test) ## predict test-set
confusionMatrix(bush15nn.test, bush15Test$bush15, "Y") ## print validation summaries
bush15nn.pred <- predict(t, bush15.nn, type = "prob") ## spatial predictions
impnn=varImp(bush15.nn)
plot(impnn, main = "Variable contribution to Neural Net (expert), 10-30%")
#+ Plot predictions by GeoSurvey variables ---------------------------------
# bush encroachment 10-30 % prediction plots
bush15.preds <- stack(1-bush15rf.pred, 1-bush15gbm.pred, 1-bush15nn.pred)
names(bush15.preds) <- c("randomForest","gradient boosting","neural net")
plot(bush15.preds, axes = F)
bush15pred=extract(bush15.preds, bush)
# presence/absence of bush> 10-30 % (bush10-30%, present = Y, absent = N)
bushens <- cbind.data.frame(bush15, bush15pred)
bushens <- na.omit(bushens)
bushensTest <- bushens[-bush15Index,] ## replicate previous test set
# Regularized ensemble weighting on the test set <glm>
# 5-fold CV
ens <- trainControl(method = "cv", number = 5)
# presence/absence of bushland (bush 15, present = Y, absent = N)
bush15.ens <- train(bush15 ~ randomForest + gradient.boosting + neural.net, data = bushensTest,
family= "binomial",
method = "glmnet",
trControl = ens)
bushens15.pred <- predict(bush15.ens, bushensTest, type="prob")
bush.test <- cbind(bushensTest, bushens15.pred)
bushp <- subset(bush.test, bush15=="Y", select=c(Y))
busha <- subset(bush.test, bush15=="N", select=c(Y))
bush.eval <- evaluate(p=bushp[,1], a=busha[,1]) ## calculate ROC's on test set <dismo>
bush.eval
plot(bush.eval, 'ROC') ## plot ROC curve
bush.thld <- threshold(bush.eval, 'spec_sens') ## TPR+TNR threshold for classification
bushens.pred <- predict(bush15.preds, bush15.ens, type="prob") ## spatial prediction
bushens15=1-bushens.pred
plot(bushens15)
bushmask=bushens15>bush.thld

# Export Gtif's to "./NAM_results"

#write tiff
rf=writeRaster(1-bushens.pred,filename="./NAM_Results/bush_15%_pred.tif", format= "GTiff", overwrite = TRUE)
rf=writeRaster(bushmask,filename="./NAM_Results/bush_15%.tif", format= "GTiff", overwrite = TRUE)


###___________________________________________________
#woody cover 0-10%
#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# presence/absence of bushland (0%-10%, present = Y, absent = N)
bush1.rf <- train(bush10 ~ ., data = bush1Train,
                  method = "rf",
                  trControl = oob)
bush1rf.test <- predict(bush1.rf, bush1Test) ## predict test-set
confusionMatrix(bush1rf.test, bush1Test$bush10, "Y") ## print validation summaries
bush1rf.pred <- predict(t, bush1.rf, type = "prob") ## spatial predictions

#regression contribution
imprf=varImp(bush1.rf)
plot(imprf, main = "Variable contribution to Random Forest (expert), >0-10%")



#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# presence/absence of 0-10% bush cover (bush10, present = Y, absent = N)
bush1.gbm <- train(bush10 ~ ., data = bush1Train,
                   method = "gbm",
                   trControl = gbm)
bush1gbm.test <- predict(bush1.gbm, bush1Test) ## predict test-set
confusionMatrix(bush1gbm.test, bush1Test$bush10, "Y") ## print validation summaries
bush1gbm.pred=predict(t, bush1.gbm, type = "prob")

impgbm=varImp(bush1.gbm)
plot(impgbm, main = "Variable contribution to Gradient Boosting (expert), >0-10%")



# presence/absence of Woody Vegetation Cover 0-10% (bush10, present = Y, absent = N)

#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# presence/absence of 0-10% bush(bush10, present = Y, absent = N)
bush10.nn <- train(bush10 ~ ., data = bush1Train,
                   method = "nnet",
                   trControl = nn)
bush10nn.test <- predict(bush10.nn, bush1Test) ## predict test-set
confusionMatrix(bush10nn.test, bush1Test$bush10, "Y") ## print validation summaries
bush10nn.pred <- predict(t, bush10.nn, type = "prob") ## spatial predictions
#variable contribution
impnn=varImp(bush10.nn)
plot(impnn, main = "Variable contribution to Neural Net (expert), >0-10%")


#+ Plot predictions by GeoSurvey variables ---------------------------------
# bush encroachment 0-10 % prediction plots
bush10.preds <- stack(1-bush1rf.pred, 1-bush1gbm.pred, 1-bush10nn.pred)
names(bush10.preds) <- c("randomForest","gradient boosting","neural net")
plot(bush10.preds, axes = F)

bush10pred=extract(bush10.preds, bush)


# presence/absence of bush> 0-10 % (bush 10, present = Y, absent = N)
bushens <- cbind.data.frame(bush10, bush10pred)
bushens <- na.omit(bushens)
bushensTest <- bushens[-bush1Index,] ## replicate previous test set


# Regularized ensemble weighting on the test set <glm>
# 5-fold CV
ens <- trainControl(method = "cv", number = 5)

# presence/absence of bushland (bush 10, present = Y, absent = N)
bush10.ens <- train(bush10 ~ randomForest + gradient.boosting + neural.net, data = bushensTest,
                    family = "binomial", 
                    method = "glmnet",
                    trControl = ens)
bushens10.pred <- predict(bush10.ens, bushensTest, type="prob")
bush.test <- cbind(bushensTest, bushens10.pred)
bushp <- subset(bush.test, bush10=="Y", select=c(Y))
busha <- subset(bush.test, bush10=="N", select=c(Y))
bush.eval <- evaluate(p=bushp[,1], a=busha[,1]) ## calculate ROC's on test set <dismo>
bush.eval
plot(bush.eval, 'ROC') ## plot ROC curve
bush.thld <- threshold(bush.eval, 'spec_sens') ## TPR+TNR threshold for classification
bushens.pred <- predict(bush10.preds, bush10.ens, type="prob") ## spatial prediction
bushmask <- 1-bushens.pred > bush.thld
plot(bushmask, axes = F, legend = F)


#contribution to ensemble
impens=varImp(bush10.ens)
plot(impens, main = "Contribution to Ensemble Regression (expert), >0-10%")

# Export Gtif's to "./NAM_results"

#write tiff
rf=writeRaster(1-bushens.pred,filename="./NAM_Results/bush_10%_pred.tif", format= "GTiff", overwrite = TRUE)
rf=writeRaster(bushmask,filename="./NAM_Results/bush_10%.tif", format= "GTiff", overwrite = TRUE)


###___________________________________________________
#woody cover 0%
#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# presence/absence of bushland (0%, present = Y, absent = N)
bush0.rf <- train(bush0 ~ ., data = bush0Train,
                  method = "rf",
                  trControl = oob)
bush0rf.test <- predict(bush0.rf, bush0Test) ## predict test-set
confusionMatrix(bush0rf.test, bush0Test$bush0, "Y") ## print validation summaries
bush0rf.pred <- predict(t, bush0.rf, type = "prob") ## spatial predictions

#variable contribution
imprf=varImp(bush0.rf)
plot(imprf, main = "Variable contribution to Random Forest (expert), no cover")



#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# presence/absence of  bush cover (bush0, present = Y, absent = N)
bush0.gbm <- train(bush0 ~ ., data = bush0Train,
                   method = "gbm",                   
                   trControl = gbm)
bush0gbm.test <- predict(bush0.gbm, bush0Test) ## predict test-set
confusionMatrix(bush0gbm.test, bush0Test$bush0, "Y") ## print validation summaries
bush0gbm.pred=predict(t, bush0.gbm, type = "prob")

impgbm=varImp(bush0.gbm)
plot(impgbm, main = "Variable contribution to Gradient Boosting (expert), no cover")


# presence/absence of Woody Vegetation Cover 0% (bush0, present = Y, absent = N)

#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# presence/absence of 0% bush(bush0, present = Y, absent = N)
bush0.nn <- train(bush0 ~ ., data = bush0Train,
                  method = "nnet",
                  trControl = nn)
bush0nn.test <- predict(bush0.nn, bush0Test) ## predict test-set
confusionMatrix(bush0nn.test, bush0Test$bush0, "Y") ## print validation summaries
bush0nn.pred <- predict(t, bush0.nn, type = "prob") ## spatial predictions

#variable contribution to neural network
impnn=varImp(bush0.nn)
plot(impnn, main = "Variable contribution to Neural Net (expert), no cover")


#+ Plot predictions by GeoSurvey variables ---------------------------------
# bush encroachment 0 % prediction plots
bush0.preds <- stack(bush0rf.pred, bush0gbm.pred, bush0nn.pred)
names(bush0.preds) <- c("randomForest","gradient boosting","neural net")
plot(bush0.preds, axes = F)

bush0pred=extract(bush0.preds, bush)

#ensemble regression preparation
# presence/absence of bush= 0% (bush0, present = Y =trees present, Trees absent = N)
bushens <- cbind.data.frame(bush0, bush0pred)
bushens <- na.omit(bushens)
bushensTest <- bushens[-bush0Index,] ## replicate previous test set


# Regularized ensemble weighting on the test set <glm>
# 5-fold CV
ens <- trainControl(method = "cv", number = 5)

# presence/absence of bushland (bush 0, present = Y, absent = N)

bush0.ens <- train(bush0 ~ randomForest + gradient.boosting + neural.net, data = bushensTest,
                   family = "binomial", 
                   method = "glmnet",
                   trControl = ens)
bushens0.pred <- predict(bush0.ens, bushensTest, type="prob")
bush.test <- cbind(bushensTest, bushens0.pred)
bushp <- subset(bush.test, bush0=="N", select=c(N))
busha <- subset(bush.test, bush0=="Y", select=c(N))
bush.eval <- evaluate(p=bushp[,1], a=busha[,1]) ## calculate ROC's on test set <dismo>
bush.eval
plot(bush.eval, 'ROC') ## plot ROC curve
bush.thld <- threshold(bush.eval, 'spec_sens') ## TPR+TNR threshold for classification
bushens.pred <- predict(bush0.preds, bush0.ens, type="prob") ## spatial prediction
bushmask <- bushens.pred > bush.thld
plot(bushmask, axes = F, legend = F)

#Regression contribution to ensemble
impens=varImp(bush0.ens)
plot(impens, main = "Regression contribution to Ensemble (expert), no cover")


# Export Gtif's to "./NAM_results"

#write tiff
rf=writeRaster(bushmask,filename="./NAM_Results/bush_0%.tif", format= "GTiff", overwrite = TRUE)
rf=writeRaster(1-bushens.pred,filename="./NAM_Results/bush_0%_pred.tif", format= "GTiff", overwrite = TRUE)




