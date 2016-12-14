#' Ensemble predictions of Ghana GeoSurvey cropland, woody vegetation cover,
#' and rural settlement observations. 
#' Alex Verlinden 2016 after M. Walsh, April 2014
# observations collected by crowdsourcing using "Geosurvey"  in October and November 2015 
# Required packages
# install.packages(c("devtools", "ROSE", "reshape2","doParallel", "downloader","raster","rgdal","caret","randomForest","gbm","nnet","glmnet","dismo")), dependencies=TRUE)
require(devtools)
require(doParallel)
require(downloader)
require(raster)
require(rgdal)
require(caret)
require(randomForest)
require(gbm)
require(nnet)
require(glmnet)
require(dismo)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("GH_data", showWarnings=F)
dat_dir <- "./GH_data"

#download Geosurvey data  13000 +
download.file("https://www.dropbox.com/s/ysh55mdy5ux9698/GH_class_obs.zip?dl=0", 
              "./GH_data/GH_class_obs.zip", mode="wb")
unzip("./GH_data/GH_class_obs.zip", exdir= "./GH_data", overwrite=T)
geos1 <- read.table(paste(dat_dir, "/GH_class_obs.csv", sep=""), header=T, sep=",")
geos1 <- na.omit(geos1)
# download Ghana Gtifs (!!!~200 Mb) and stack in raster
download.file("https://www.dropbox.com/s/nbpi0l4utm32cgw/GH_250_grids.zip?dl=0", "./GH_data/GH_250_grids.zip", mode="wb")
unzip("./GH_data/GH_250_grids.zip", exdir="./GH_data", overwrite=T)
glist <- list.files(path="./GH_data/GH_250_grids", pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid,center=TRUE, scale=TRUE)

#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
geos1.proj <- as.data.frame(project(cbind(geos1$Longitude, geos1$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geos1.proj) <- c("x","y")
geos1 <- cbind(geos1, geos1.proj)
coordinates(geos1) <- ~x+y
projection(geos1) <- projection(grid)
# Extract gridded variables at GeoSurvey locations
geos1grid=extract(t, geos1)

# Assemble dataframes
# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP1 <- geos1$CRP
#to test if cropland is a rare event as presences of much less than 15 % are difficult to model
#to test how imbalanced the data are
prop.table(table(CRP1))

#presence absence of Woodland
WDL1=geos1$WLD
prop.table(table(WDL1))
wdl1dat=cbind.data.frame(WDL1, geos1grid)
wdl1dat=na.omit(wdl1dat)

#First prediction of woodland
#is very low presence imbalanced, correction needed
#+ Split data into train and test sets ------------------------------------
seed=12345
set.seed=seed

# woodland train/test split
wdl1Index <- createDataPartition(wdl1dat$WDL1, p = 2/3, list = FALSE, times = 1)
wdl1Train <- wdl1dat[ wdl1Index,]
wdl1Test  <- wdl1dat[-wdl1Index,]

#for class imbalances in Random Forest

thresh_code <- getModelInfo("rf", regex = FALSE)[[1]]
thresh_code$type <- c("Classification")
## Add the threshold as another tuning parameter
thresh_code$parameters <- data.frame(parameter = c("mtry", "threshold"),
                                     class = c("numeric", "numeric"),
                                     label = c("#Randomly Selected Predictors",
                                               "Probability Cutoff"))

## The default tuning grid code:
thresh_code$grid <- function(x, y, len = NULL, search = "grid") {
  p <- ncol(x)
  if(search == "grid") {
    grid <- expand.grid(mtry = floor(sqrt(p)), 
                        threshold = seq(.01, .99, length = len))
  } else {
    grid <- expand.grid(mtry = sample(1:p, size = len),
                        threshold = runif(1, 0, size = len))
  }
  grid
}
## Here we fit a single random forest model (with a fixed mtry)
## and loop over the threshold values to get predictions from the same
## randomForest model.
thresh_code$loop = function(grid) {   
  library(plyr)
  loop <- ddply(grid, c("mtry"),
                function(x) c(threshold = max(x$threshold)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for(i in seq(along = loop$threshold)) {
    index <- which(grid$mtry == loop$mtry[i])
    cuts <- grid[index, "threshold"] 
    submodels[[i]] <- data.frame(threshold = cuts[cuts != loop$threshold[i]])
  }    
  list(loop = loop, submodels = submodels)
}

## Fit the model independent of the threshold parameter
thresh_code$fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
  if(length(levels(y)) != 2)
    stop("This works only for 2-class problems")
  randomForest(x, y, mtry = param$mtry, ...)
}
## Now get a probability prediction and use different thresholds to
## get the predicted class
thresh_code$predict = function(modelFit, newdata, submodels = NULL) {
  class1Prob <- predict(modelFit, 
                        newdata, 
                        type = "prob")[, modelFit$obsLevels[1]]
  ## Raise the threshold for class #1 and a higher level of
  ## evidence is needed to call it class 1 so it should 
  ## decrease sensitivity and increase specificity
  out <- ifelse(class1Prob >= modelFit$tuneValue$threshold,
                modelFit$obsLevels[1], 
                modelFit$obsLevels[2])
  if(!is.null(submodels)) {
    tmp2 <- out
    out <- vector(mode = "list", length = length(submodels$threshold))
    out[[1]] <- tmp2
    for(i in seq(along = submodels$threshold)) {
      out[[i+1]] <- ifelse(class1Prob >= submodels$threshold[[i]],
                           modelFit$obsLevels[1], 
                           modelFit$obsLevels[2])
    }
  } 
  out  
}

## The probabilities are always the same but we have to create
## mulitple versions of the probs to evaluate the data across
## thresholds
thresh_code$prob = function(modelFit, newdata, submodels = NULL) {
  out <- as.data.frame(predict(modelFit, newdata, type = "prob"))
  if(!is.null(submodels)) {
    probs <- out
    out <- vector(mode = "list", length = length(submodels$threshold)+1)
    out <- lapply(out, function(x) probs)
  } 
  out 
}
fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  ## This code will get use the area under the ROC curve and the
  ## sensitivity and specificity values using the current candidate
  ## value of the probability threshold.
  out <- c(twoClassSummary(data, lev = levels(data$obs), model = NULL))
  
  ## The best possible model has sensitivity of 1 and specificity of 1. 
  ## How far are we from that value?
  coords <- matrix(c(1, 1, out["Spec"], out["Sens"]), 
                   ncol = 2, 
                   byrow = TRUE)
  colnames(coords) <- c("Spec", "Sens")
  rownames(coords) <- c("Best", "Current")
  c(out, Dist = dist(coords)[1])
}
set.seed(949)
WDL1.rf <- train(WDL1 ~ ., data = wdl1Train,
                method = thresh_code,
                ## Minimize the distance to the perfect model
                metric = "Dist",
                maximize = FALSE,
                tuneLength = 20,
                ntree = 501,
                trControl = trainControl(method = "repeatedcv",
                                         repeats = 5,
                                         classProbs = TRUE,
                                         summaryFunction = fourStats))
wdl1.test <- predict(WDL1.rf, wdl1Test) # predict test set
confusionMatrix(wdl1.test, wdl1Test$WDL1, "Y") ## print validation summaries
wdl1.pred <- predict(t, WDL1.rf, type="prob") ## spatial predictions
plot(varImp(WDL1.rf,scale=F))

writeRaster(1-wdl1.pred, filename="./GH_results/GH_wdl1_250.tif", format="GTiff", overwrite=T)

#now add the new prediction for woodland

GH_wdl250m= raster("./GH_results/GH_wdl1_250.tif")
grid=addLayer(grid,GH_wdl250m)

t=scale(grid,center=TRUE, scale=TRUE)

#  n=13000 extract covariate values
geos1grid=extract(t, geos1)

# Assemble dataframes
# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP1 <- geos1$CRP


crp1dat <- cbind.data.frame(CRP1, geos1grid)
crp1dat <- na.omit(crp1dat)
# cropland train/test split
crp1Index <- createDataPartition(crp1dat$CRP1, p = 2/3, list = FALSE, times = 1)
crp1Train <- crp1dat[ crp1Index,]
crp1Test  <- crp1dat[-crp1Index,]

mc <- makeCluster(detectCores())
registerDoParallel(mc)


#glmnet--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE)
#glmnet using binomial distribution presence/absence of Cropland (CRP, present = Y, absent = N)
CRP1.glm=train(CRP1 ~ ., data=crp1Train, family= "binomial",method="glmnet",
               metric="ROC", 
               trControl=objControl)
crp1glm.test <- predict(CRP1.glm, crp1Test) # predict test set
confusionMatrix(crp1glm.test, crp1Test$CRP, "Y") ## print validation summaries
crp1glm.pred <- predict(t, CRP1.glm, type="prob") ## spatial predictions
plot(varImp(CRP1.glm,scale=F))

#rf
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE,
                           classProbs = TRUE,
                           summaryFunction= twoClassSummary)

CRP1.rf <- train(CRP1 ~ ., data = crp1Train,
                method = "rf",
                metric= "ROC",
                trControl = objControl)
crp1rf.tst <- predict(CRP1.rf, crp1Test, type="prob") ## predict test-set
crp1rf.test=cbind(crp1Test,crp1rf.tst)
confusionMatrix(crp1rf.test, crp1Test$CRP1, "Y") ## print validation summaries
confusionMatrix(CRP1.rf)
crp1rf.pred <- predict(t, CRP1.rf, type = "prob") ## spatial predictions
plot(varImp(CRP1.rf,scale=F))

#gbm
CRP1.gbm <- train(CRP1 ~ ., data = crp1Train,
                 method = "gbm",
                 metric = "ROC",
                 trControl = objControl)
crp1gbm.test <- predict(CRP1.gbm, crp1Test) ## predict test-set
confusionMatrix(crp1gbm.test, crp1Test$CRP1, "Y") ## print validation summaries
crp1gbm.pred <- predict(t, CRP1.gbm, type = "prob") ## spatial predictions
#neural net
CRP1.nn <- train(CRP1 ~., data=crp1Train, 
                 method = "nnet", 
                 metric= "ROC",
                 #preProc = c("center", "scale"), 
                 trControl = objControl)
print(CRP1.nn)
CRP1.imp <- varImp(CRP1.nn, useModel = FALSE)
plot(CRP1.imp, top=27)
crp1nn.test <- predict(CRP1.nn, crp1Test) # predict test set
confusionMatrix(crp1nn.test, crp1Test$CRP1, "Y") ## print validation summaries
crp1nn.pred <- predict(t, CRP1.nn, type="prob")

#ksvm
test=ksvm(WDL1~.,data=wdl1Train,kernel="rbfdot",kpar="automatic",C=60,cross=3,prob.model=TRUE)
ksvm.pred=predict(t, test,type="prob")

#+ Ensemble predictions <rf>, <gbm>, <nnet>, <glm>, <ksvm> -------------------------------
# Ensemble set-up
pred <- stack(1-crp1rf.pred, 1-crp1gbm.pred, 1-crp1nn.pred, 
              1-crp1glm.pred, 1-ksvm.pred)
names(pred) <- c("CRPrf","CRPgbm","CRPnn", "CRPglm", "CRPksvm")
geospred <- extract(pred, geos1)

# presence/absence of Cropland (CRP, present = Y, absent = N)
crp1ens <- cbind.data.frame(CRP1, geospred)
crp1ens <- na.omit(crp1ens)
crp1ensTest <- crp1ens[-crp1Index,] ## replicate previous test set

#control
ens <- trainControl(method = "cv", number = 10)
#elastic net
CRP1.ens <- train(CRP1 ~ ., data = crp1ensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
crp1ens.test <- predict(CRP1.ens, crp1ensTest,  type="prob") ## predict test-set
confusionMatrix(crp1ens.test, crp1Test$CRP1, "Y") ## print validation summaries
confusionMatrix(CRP1.ens)
crp1.test <- cbind(crp1ensTest, crp1ens.test)
crp <- subset(crp1.test, CRP1=="Y", select=c(Y))
cra <- subset(crp1.test, CRP1=="N", select=c(Y))
crp.eval <- evaluate(p=crp[,1], a=cra[,1]) ## calculate ROC's on test set <dismo>
crp.eval
plot(crp.eval, 'ROC') ## plot ROC curve
crp.thld <- threshold(crp.eval, 'spec_sens') ## TPR+TNR threshold for classification
crp1ens.pred <- predict(pred, CRP1.ens, type="prob") ## spatial prediction
plot(1-crp1ens.pred, axes=F, main= "Cropland, Ensemble prediction")
crpmask <- 1-crp1ens.pred > crp.thld
plot(crpmask, axes = F, legend = F)
plot(varImp(CRP1.ens,scale=F))

writeRaster(1-crp1ens.pred, filename="./GH_results/GH_crp1_250_ens.tif", format="GTiff", overwrite=T)
writeRaster(crpmask, filename="./GH_results/GH_crp1mask.tif", format="GTiff", overwrite=T)
