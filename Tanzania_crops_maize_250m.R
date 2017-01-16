# Script for crop distribution  models TZ using ensemble regression
# basis for cropland mask is the 12 k point survey for Tanzania of 2014-2015 conducted by AfSIS for Africa
# grids are from Africasoils.net
# over 10500 field data are collected by TanSIS in Tanzania 2015 and 2016
#script in development to test crop distribution model based on presence/absence from crop scout
# area of TZ = 941761 km2
# estimated cropland present from 12 k Geosurvey observations = 38.6 %
#around 81% of TanSIS soil and crop survey is on cropland
#around 67 % of cropland has maize
#Alex Verlinden January 2017 based on M. Walsh and J.Chen
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret", "doParallel")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(doParallel)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("TZ_crops", showWarnings=F)
dat_dir <- "./TZ_crops"
# download crop presence/absence locations
# these are data from 2017 crop scout ODK forms n= 10000+
download.file("https://www.dropbox.com/s/02g8dmzvr18nyx3/Crop_TZ_JAN_2017.csv.zip?dl=0", "./TZ_crops/Crop_TZ_JAN_2017.csv.zip", mode="wb")
unzip("./TZ_crops/Crop_TZ_JAN_2017.csv.zip", exdir=dat_dir, overwrite=T)
# note that 0 and 1 are not ok for Caret for classifications, should be N and Y or similar
ban <- read.csv(paste(dat_dir, "/Crop_TZ_JAN_2017.csv", sep= ""), header=T, sep=",")

#Download all test geosurvey data and select non crop areas
#Download all geosurvey data
download.file("https://www.dropbox.com/s/339k17oic3n3ju6/TZ_geos_012015.csv?dl=0", "./TZ_crops/TZ_geos_012015.csv", mode="wb")
geos <- read.csv(paste(dat_dir, "/TZ_geos_012015.csv", sep=""), header=T, sep=",")
geos <- geos[,1:7]
geos.no= subset.data.frame(geos, geos$CRP=="N") # select non crop areas
geos.no=na.omit(geos.no)
#download grids for TZ  ~ 530 MB
download.file("https://www.dropbox.com/s/r25qfm0yikiubeh/TZ_GRIDS250m.zip?dl=0","./TZ_crops/TZ_GRIDS250m.zip",  mode="wb")
unzip("./TZ_crops/TZ_GRIDS250m.zip", exdir=dat_dir, overwrite=T)


# load woodland pred, settlement pred and cropland pred at 250m
download.file("https://www.dropbox.com/s/6uxttpp5owrogpy/TZ_landcov.zip?dl=0","./TZ_crops/TZ_landcov.zip",  mode="wb")
unzip("./TZ_crops/TZ_landcov.zip", exdir=dat_dir, overwrite=T)
# stack all covariates
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid, center=TRUE,scale=TRUE) # scale all covariates

#+ Data setup for TZ crops--------------------------------------------------------------
# Project crop data to grid CRS
ban.proj <- as.data.frame(project(cbind(ban$X_gps_longitude, ban$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ban.proj) <- c("x","y")
coordinates(ban.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ban.proj) <- projection(grid)
#remove plotting formatting
dev.off()
#project no crop data to grid GRS
geos.no.proj= as.data.frame(project(cbind(geos.no$Lon, geos.no$Lat),"+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geos.no.proj)= c("x","y")
coordinates(geos.no.proj)= ~x+y # convert to Spatial data frame
projection(geos.no.proj)=projection(grid)
# add points from cropland to non cropland
allpts=rbind(ban.proj,geos.no.proj)
#only points from Tansis data with crops
ban.crops=subset.data.frame(ban, ban$crop_pres=="Y") #selects cropland from TanSIS survey
table(ban.crops$maize) # gives proportion of maize on cropland
# Extract gridded variables for TZ data observations cropland only
banex <- data.frame(coordinates(ban.proj), extract(t, ban.proj))
# Extract gridded variables for all TZ survey data
allex=data.frame(coordinates(allpts), extract(t, allpts))

mc <- makeCluster(detectCores())
registerDoParallel(mc)
banex=  banex[,3:40] #exclude coordinates
allex= allex[,3:40]
# now bind crop species column to the covariates
# this has to change with every new crop
#use names (ban) to check crop name

# for maize on TanSIS data
maizepresabs=cbind(ban$maize, banex)
maizepresabs=na.omit(maizepresabs)
colnames(maizepresabs)[1]="maize"
maizepresabs$maize=as.factor(maizepresabs$maize)
prop.table(table(maizepresabs$maize))
#for maize in crop and non crop points
geos.no$maize="N"
geos.no$maize=as.factor(geos.no$maize)
mz.list <- c( as.character(ban$maize) ,
                   as.character(geos.no$maize)  )
mz.list=as.factor(mz.list)
mzpresabs=cbind.data.frame(mz.list,allex) # bind all points with extracted covariates
prop.table(table(mzpresabs$mz.list))

#download cropmask 250m
dir.create("./TZ_cropmask")
download.file("https://www.dropbox.com/s/bjucbwpgexa3flc/TZ_cropmask_250m.zip?dl=0","./TZ_cropmask/TZ_cropmask_250m.zip",  mode="wb")
unzip("./TZ_cropmask/TZ_cropmask_250m.zip", exdir= "./TZ_cropmask", overwrite=T)

crp=raster("./TZ_cropmask/TZ_cropmask_250m.tif")
crp[crp==0]=NA

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split


#maize for all points about  31 %. proportion of cropland in TZ over 40%
mzIndex=createDataPartition(mzpresabs$mz.list, p=2/3, list = FALSE, times=1)
mzTrain=mzpresabs[mzIndex, ]
mzTest=mzpresabs[-mzIndex,]
mzTest=na.omit(mzTest)


#____________
#set up data for caret
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)

#maize glmnet for all points including non cropland
maize.glm=train(mz.list ~ ., data=mzTrain, family= "binomial",
                method="glmnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.glm)
#variable importance
plot(varImp(maize.glm,scale=F), main= "Variable Importance GLMnet")
#spatial predictions
mzglm.pred <- predict(t,maize.glm, type="prob") 
# maize Random Forest
maize.rf=train(mz.list ~ ., data=mzTrain, family= "binomial",
               method="rf",ntree= 501,metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.rf)
#variable importance
plot(varImp(maize.rf,scale=F), main = " Variable Importance Random Forest")
#spatial predictions
mzrf.pred <- predict(t,maize.rf, type="prob") 

#maize Gradient boosting for all points including non cropland
maize.gbm=train(mz.list ~ ., data=mzTrain,
                method="gbm", metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.gbm)
#variable importance
plot(varImp(maize.gbm,scale=F))
#spatial predictions
mzgbm.pred <- predict(t,maize.gbm, type="prob") 

#neural net maize
maize.nn=train(mz.list ~ ., data=mzTrain, family= "binomial",
               method="nnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.nn)
#variable importance
plot(varImp(maize.nn,scale=F))
#spatial predictions
mznn.pred <- predict(t,maize.nn, type="prob") 


#ensemble regression glmnet (elastic net)
pred <- stack(1-mzglm.pred, 1-mzrf.pred, 
              1-mzgbm.pred, 1-mznn.pred)
names(pred) <- c("mzglm","mzrf","mzgbm", "mznn")
geospred <- extract(pred, allpts)

# presence/absence of Cropland (present = Y, absent = N)
mzens <- cbind.data.frame(mz.list, geospred)
mzens <- na.omit(mzens)
mzensTest <- mzens[-mzIndex,] ## replicate previous test set
names(mzensTest)[1]= "maize"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of maize (present = Y, absent = N)
maize.ens <- train(maize ~. , data = mzensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)
confusionMatrix(maize.ens) # print validation summaries on crossvalidation
mzens.pred <- predict(maize.ens, mzensTest,  type="prob") ## predict test-set
mz.test <- cbind(mzensTest, mzens.pred)
mzp <- subset(mz.test, maize=="Y", select=c(Y))
mza <- subset(mz.test, maize=="N", select=c(Y))
mz.eval <- evaluate(p=mzp[,1], a=mza[,1]) ## calculate ROC's on test set <dismo>
mz.eval
plot(mz.eval, 'ROC')


#for all data incl test
#maize glmnet for all points including non cropland
maize.glm=train(mz.list ~ ., data=mzpresabs, family= "binomial",
                method="glmnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.glm)
#variable importance
plot(varImp(maize.glm,scale=F))
#spatial predictions
mzglm.pred <- predict(t,maize.glm, type="prob") 
# maize Random Forest
maize.rf=train(mz.list ~ ., data=mzpresabs, family= "binomial",
               method="rf",ntree= 501,metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.rf)
#variable importance
plot(varImp(maize.rf,scale=F))
#spatial predictions
mzrf.pred <- predict(t,maize.rf, type="prob") 

#maize Gradient boosting for all points including non cropland
maize.gbm=train(mz.list ~ ., data=mzpresabs,
                method="gbm", metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.gbm)
#variable importance
plot(varImp(maize.gbm,scale=F))
#spatial predictions
mzgbm.pred <- predict(t,maize.gbm, type="prob") 

#neural net maize
maize.nn=train(mz.list ~ ., data=mzpresabs, family= "binomial",
               method="nnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(maize.nn)
#variable importance
plot(varImp(maize.nn,scale=F))
#spatial predictions
mznn.pred <- predict(t,maize.nn, type="prob") 


#ensemble regression glmnet (elastic net)
pred <- stack(1-mzglm.pred, 1-mzrf.pred, 
              1-mzgbm.pred, 1-mznn.pred)
names(pred) <- c("mzglm","mzrf","mzgbm", "mznn")
geospred <- extract(pred, allpts)

# presence/absence of Cropland (present = Y, absent = N)
mzens <- cbind.data.frame(mz.list, geospred)
mzens <- na.omit(mzens)
mzensTest <- mzens[-mzIndex,] ## replicate previous test set
names(mzensTest)[1]= "maize"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of maize (present = Y, absent = N)
maize.ens <- train(maize ~. , data = mzensTest,
                   family = "binomial", 
                   method = "glmnet",
                   trControl = ens)
confusionMatrix(maize.ens) # print validation summaries on crossvalidation
mzens.pred <- predict(maize.ens, mzensTest,  type="prob") ## predict test-set
mz.test <- cbind(mzensTest, mzens.pred)
mzp <- subset(mz.test, maize=="Y", select=c(Y))
mza <- subset(mz.test, maize=="N", select=c(Y))
mz.eval <- evaluate(p=mzp[,1], a=mza[,1]) ## calculate ROC's on test set <dismo>
mz.eval
mz.thld <- threshold(mz.eval, 'spec_sens') 
#spatial predictions
mzens.pred <- predict(pred, maize.ens, type="prob") 
plot(1-mzens.pred, main = "Ensemble prediction Maize Tanzania 2016")
dir.create("TZ_results", showWarnings=F)
writeRaster(1-mzens.pred, filename = "./TZ_results/TZ_maizepred.tif", overwrite= TRUE )
mzmask=(1-mzens.pred)>mz.thld
plot(mzmask, main= "Predicted Maize distribution Tanzania 2016")
writeRaster(mzmask, filename="./TZ_results/TZ_maizemask_250m.tif", overwrite=TRUE )
cropmask2=cropmask*crp #exclude cropmask
plot(cropmask2, main="Predicted maize on predicted cropland in Tanzania 2016")
freq(mzmask)
freq(cropmask2)