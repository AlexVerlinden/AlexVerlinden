#' Ensemble predictions of Ghana GeoSurvey cropland, woody vegetation cover,
#' and rural settlement observations. 
#' Alex Verlinden 2016 after M. Walsh, April 2014
# observations collected by crowdsourcing using "Geosurvey"  in October and November 2015 
# Required packages
# install.packages(c("downloader","raster","foreach",rgdal","caret","doMC","rpart", randomForest","gbm","nnet","glmnet","dismo")), dependencies=TRUE)
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
require(doMC)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("GH_data", showWarnings=F)
dat_dir <- "./GH_data"

# download GeoSurvey data
download.file("https://www.dropbox.com/s/ald83m8h3fc1hz2/GH_100k_crop.csv?dl=0", "./GH_data/GH_100k_crop.csv", mode="wb")
geocrop <- read.table(paste(dat_dir, "/GH_100k_crop.csv", sep=""), header=T, sep=",")
geocrop <- na.omit(geocrop)

# download Ghana Gtifs (~8 Mb) and stack in raster
download.file("https://www.dropbox.com/s/3eo70huv7s8d9e5/GH_preds_1km.zip?dl=0", "./GH_data/GH_preds_1km.zip", mode="wb")
unzip("./GH_data/GH_preds_1km.zip", exdir="./GH_data", overwrite=T)
glist <- list.files(path="./GH_data", pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid, center=TRUE,scale=TRUE)
#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
geocrop.proj <- as.data.frame(project(cbind(geocrop$Longitude, geocrop$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geocrop.proj) <- c("x","y")
geocrop <- cbind(geocrop, geocrop.proj)
coordinates(geocrop) <- ~x+y
projection(geocrop) <- projection(grid)

# Extract gridded variables at GeoSurvey locations
geosgrid <- extract(t, geocrop)

# Assemble dataframes
#Cropland (CRP, 0-16)
CRP <- round(geocrop$CRPCV*6.25)
crpdat <- cbind.data.frame(CRP, geosgrid)
crpdat <- na.omit(crpdat)


# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Cropland train/test split
crpIndex <- createDataPartition(crpdat$CRP, p = 0.75, list = FALSE, times = 1)
crpTrain <- crpdat[ crpIndex,]
crpTest  <- crpdat[-crpIndex,]


#some initial tests
hist(CRP)
#test for Poisson model fitting data
CRP.test=glm (CRP~., data=crpTrain, family= "poisson")
1 - pchisq(summary(CRP.test)$deviance,
           summary(CRP.test)$df.residual)
#test for zero inflation
install.packages("pscl")
library(pscl)
CRP.test2=zeroinfl(CRP~.|., data=crpTrain)
cbind(crpTest, 
      Count = predict(CRP.test2, newdata = crpTest, type = "count"),
      Zero = predict(CRP.test2, newdata = crpTest, type = "zero")
)
##We can test for overdispersion in the count part of the zero-inflated model by specifying a negative binomial distribution.
model.zip.3 = zeroinfl(CRP~ .|1, data = crpTrain, dist = "negbin")
summary(model.zip.3)
# this means the model should use zero inflated Poisson distribution ?

#glmnet can only be used after testing for Poisson distribution--------------------------------
#objControl <- trainControl(method='cv', number=10, returnResamp='none')
#glmnet using poisson distribution of Cropland counts 
#CRP.glm=train(CRP ~ ., data=crpTrain, family= "poisson",method="glmnet",metric="RMSE", trControl=objControl)
#crpglm.test <- predict(CRP.glm, crpTest) # predict test set
#crpglm.pred <- predict(t, CRP.glm) ## spatial predictions
#plot(varImp(CRP.glm,scale=F))

# Start foreach to parallelize model fitting
#mc <- makeCluster(detectCores())
registerDoMC(cores=4)

#rpart works better on classification
tc=trainControl(method = "cv", number = 10)

CRP.rp <- train(CRP ~., data=crpTrain, 
                 method = "rpart",
                 # preProc = c("center", "scale"), 
                 trControl = tc)
print(CRP.rp)
CRP.imp <- varImp(CRP.rp, useModel = FALSE)
plot(CRP.imp, top=27)
crprp.pred <- predict(t, CRP.rp)
#deepnet 
##### seems deepnet works mainly on classification
tc=trainControl(method = "cv", number = 10)

CRP.dnn <- train(CRP ~., data=crpTrain, 
                method = "dnn",
               # preProc = c("center", "scale"), 
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

crpdnn.pred <- predict(t, CRP.dnn)


#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# Cropland (CRP, 0-100) including Importance vastly increases processing time
CRP.rf <- train(CRP ~ ., data = crpTrain,
                method = "rf", importance=T, metric = "RMSE", maximize= FALSE,
                trControl = oob)
crprf.test <- predict(CRP.rf, crpTest) ## predict test-set
crprf.pred <- predict(t, CRP.rf) ## spatial predictions
plot(varImp(CRP.rf, scale=F))
#table(crpTest$CRP, crprf.test)
plot(crprf.pred, main="Fractional cover cropland RF")

#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# counts of Cropland (CRP)
CRP.gbm <- train(CRP ~ ., data = crpTrain,
                 method = "gbm",
                 trControl = gbm)
crpgbm.test <- predict(CRP.gbm, crpTest) ## predict test-set
crpgbm.pred <- predict(t, CRP.gbm) ## spatial predictions
plot(varImp(CRP.gbm, scale=F))


#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# cropland fractional cover
CRP.nn <- train(CRP ~ ., data = crpTrain,
                method = "nnet",
                trControl = nn)
crpnn.test <- predict(CRP.nn, crpTest) ## predict test-set
crpnn.pred <- predict(t, CRP.nn) ## spatial predictions


#+ Ensemble predictions <rf>, <gbm>, <nnet> -------------------------------
# Ensemble set-up

pred <- stack(crprf.pred, crpgbm.pred, crpnn.pred, crpdnn.pred, crprp.pred)
names(pred) <- c("CRPrf","CRPgbm","CRPnn", "CRPdnn", "CRPrp")
geospred <- extract(pred, geocrop)

# Cropland fractional cover
crpens <- cbind.data.frame(CRP, geospred)
crpens <- na.omit(crpens)
crpensTest <- crpens[-crpIndex,] ## replicate previous test set


# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of Cropland (CRP, present = Y, absent = N)

CRP.ens <- train(CRP ~ CRPrf + CRPgbm +CRPdnn+CRPnn + CRPrp, data = crpensTest,
                 family = "gaussian", 
                 method = "glmnet",
                 trControl = ens)

crp.pred <- predict(CRP.ens, crpensTest)
crp.test <- cbind(crpensTest, crp.pred)
crpens.pred <- predict(pred, CRP.ens) ## spatial prediction

plot(varImp(CRP.ens,scale=F))


#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("GH_results", showWarnings=F)

# Export Gtif's to "./GH_results"
writeRaster(crpens.pred, filename="./GH_results/GH_crpfractpreds.tif", filetype="GEOTiff", overwrite=T)
