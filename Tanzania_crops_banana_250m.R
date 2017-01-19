# Script for crop distribution  models TZ using ensemble regression
# basis for cropland mask is the 12 k point survey for Tanzania of 2014-2015 conducted by AfSIS for Africa
# grids are from Africasoils.net
# over 10500 field data are collected by TanSIS in Tanzania 2015 and 2016
#script in development to test crop distribution model based on presence/absence from crop scout
# area of TZ = 941761 km2
# estimated cropland present from 12 k Geosurvey observations = 38.6 %
#around 81% of TanSIS soil and crop survey is on cropland
#around 67 % of cropland has ban
#Alex Verlinden January 2017 based on M. Walsh and J.Chen
# wd: "/Users/alexverlinden/Documents/R-testing/nigeria"
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
table(ban.crops$Banana) # gives proportion of Banana on cropland
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

# for Banana on TanSIS data
banpresabs=cbind(ban$Banana, banex)
banpresabs=na.omit(banpresabs)
colnames(banpresabs)[1]="Banana"
banpresabs$Banana=as.factor(banpresabs$Banana)
prop.table(table(banpresabs$Banana))

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


#Banana for points about  10%. proportion of cropland in TZ over 40%
banIndex=createDataPartition(banpresabs$Banana, p=2/3, list = FALSE, times=1)
banTrain=banpresabs[banIndex, ]
banTest=banpresabs[-banIndex,]
banTest=na.omit(banTest)


#____________
#set up data for caret
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)

#banana glmnet for all TanSIS points
ban.glm=train(Banana ~ ., data=banTrain, family= "binomial",
                method="glmnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.glm)
#variable importance
plot(varImp(ban.glm,scale=F), main= "Variable Importance GLMnet")
#spatial predictions
banglm.pred <- predict(t,ban.glm, type="prob") 
# ban Random Forest
ban.rf=train(Banana ~ ., data=banTrain, family= "binomial",
               method="rf",ntree= 501,metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.rf)
#variable importance
plot(varImp(ban.rf,scale=F), main = " Variable Importance Random Forest")
#spatial predictions
banrf.pred <- predict(t,ban.rf, type="prob") 

#banana Gradient boosting for all TanSIS points 
ban.gbm=train(Banana ~ ., data=banTrain,
                method="gbm", metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.gbm)
#variable importance
plot(varImp(ban.gbm,scale=F))
#spatial predictions
bangbm.pred <- predict(t,ban.gbm, type="prob") 

#neural net ban
ban.nn=train(Banana ~ ., data=banTrain, family= "binomial",
               method="nnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.nn)
#variable importance
plot(varImp(ban.nn,scale=F))
#spatial predictions
bannn.pred <- predict(t,ban.nn, type="prob") 


#ensemble regression glmnet (elastic net)
pred <- stack(1-banglm.pred, 1-banrf.pred, 
              1-bangbm.pred, 1-bannn.pred)
names(pred) <- c("banglm","banrf","bangbm", "bannn")
geospred <- extract(pred, ban.proj)

# presence/absence of Banana (present = Y, absent = N)
banens <- cbind.data.frame(ban$Banana, geospred)
banens <- na.omit(banens)
banensTest <- banens[-banIndex,] ## replicate previous test set
names(banensTest)[1]= "Banana"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of ban (present = Y, absent = N)
ban.ens <- train(Banana ~. , data = banensTest,
                   family = "binomial", 
                   method = "glmnet",
                   trControl = ens)
confusionMatrix(ban.ens) # print validation summaries on crossvalidation
banens.pred <- predict(ban.ens, banensTest,  type="prob") ## predict test-set
ban.test <- cbind(banensTest, banens.pred)
banp <- subset(ban.test, Banana=="Y", select=c(Y))
bana <- subset(ban.test, Banana=="N", select=c(Y))
ban.eval <- evaluate(p=banp[,1], a=bana[,1]) ## calculate ROC's on test set <dismo>
ban.eval
plot(ban.eval, 'ROC')


#for all data incl test
#banana glmnet for all points 
ban.glm=train(Banana ~ ., data=banpresabs, family= "binomial",
                method="glmnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.glm)
#variable importance
plot(varImp(ban.glm,scale=F), main = "Variable Importance Banana Elastic net")
#spatial predictions
banglm.pred <- predict(t,ban.glm, type="prob") 
# banana Random Forest
ban.rf=train(Banana ~ ., data=banpresabs, family= "binomial",
               method="rf",ntree= 501,metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.rf)
#variable importance
plot(varImp(ban.rf,scale=F), main = "Variable Importance Banana Random Forest")
#spatial predictions
banrf.pred <- predict(t,ban.rf, type="prob") 

#ban Gradient boosting for all points including non cropland
ban.gbm=train(Banana ~ ., data=banpresabs,
                method="gbm", metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.gbm)
#variable importance
plot(varImp(ban.gbm,scale=F))
#spatial predictions
bangbm.pred <- predict(t,ban.gbm, type="prob") 

#neural net ban
ban.nn=train(Banana ~ ., data=banpresabs, family= "binomial",
               method="nnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(ban.nn)
#variable importance
plot(varImp(ban.nn,scale=F))
#spatial predictions
bannn.pred <- predict(t,ban.nn, type="prob") 


#ensemble regression glmnet (elastic net)
pred <- stack(1-banglm.pred, 1-banrf.pred, 
              1-bangbm.pred, 1-bannn.pred)
names(pred) <- c("banglm","banrf","bangbm", "bannn")
geospred <- extract(pred, ban.proj)

# presence/absence of Cropland (present = Y, absent = N)
banens <- cbind.data.frame(Banana, geospred)
banens <- na.omit(banens)
banensTest <- banens[-banIndex,] ## replicate previous test set
names(banensTest)[1]= "Banana"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of ban (present = Y, absent = N)
ban.ens <- train(Banana ~. , data = banensTest,
                   family = "binomial", 
                   method = "glmnet",
                   trControl = ens)
confusionMatrix(ban.ens) # print validation summaries on crossvalidation
banens.pred <- predict(ban.ens, banensTest,  type="prob") ## predict test-set
ban.test <- cbind(banensTest, banens.pred)
banp <- subset(ban.test, Banana=="Y", select=c(Y))
bana <- subset(ban.test, Banana=="N", select=c(Y))
ban.eval <- evaluate(p=banp[,1], a=bana[,1]) ## calculate ROC's on test set <dismo>
ban.eval
ban.thld <- threshold(ban.eval, 'spec_sens') 
#spatial predictions
banens.pred <- predict(pred, ban.ens, type="prob") 
plot(1-banens.pred, main = "Ensemble prediction Banana Tanzania 2016")
dir.create("TZ_results", showWarnings=F)
writeRaster(1-banens.pred, filename = "./TZ_results/TZ_banpred.tif", overwrite= TRUE )
banmask=(1-banens.pred)>ban.thld
plot(banmask, main= "Predicted banana distribution Tanzania 2016")
writeRaster(banmask, filename="./TZ_results/TZ_banmask_250m.tif", overwrite=TRUE )
banmask2=banmask*crp #exclude cropmask
plot(banmask2, main="Predicted banana on predicted cropland in Tanzania 2016")
freq(banmask)
freq(banmask2)