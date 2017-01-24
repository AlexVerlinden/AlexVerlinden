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
table(ban.crops$root.cassava) # gives proportion of cassava on cropland
# Extract gridded variables for TZ data observations cropland only
banex <- data.frame(coordinates(ban.proj), extract(t, ban.proj))
banex=  banex[,3:40] #exclude coordinates
# now bind crop species column to the covariates
# this has to change with every new crop
#use names (ban) to check crop name
casspresabs=cbind(ban$root.cassava, banex)
colnames(casspresabs)[1]="Cassava"
# ACAI cassava samples
download.file("https://www.dropbox.com/s/zldu22dnlrwquh5/Site%20description%20EZ%2007092016-PP.csv?dl=0", "./TZ_crops/Site%20description%20EZ%2007092016-PP.csv", mode="wb")
acai=read.csv(paste(dat_dir, "/Site%20description%20EZ%2007092016-PP.csv", sep="" ), header=T, sep=",")
#project acai data to grid TZ
acai.proj=as.data.frame(project(cbind(acai$Long.corr, acai$Lat.corr),"+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(acai.proj)=c("x", "y")
coordinates(acai.proj)= ~x+y
projection(acai.proj)=projection(grid)
#add points from cropland to non cropland and acai

# Extract gridded variables for all TZ ACAI FR data
acaiF=data.frame(coordinates(acai.proj), extract(t, acai.proj))
acaiF= acaiF[,3:40]# exclude coordinates
acai=data.frame("Y", acaiF)
colnames(acai)[1]="Cassava"
caspresabs=rbind(casspresabs,acai)

# for cassava on TanSIS data and ACAI RF
prop.table(table(caspresabs$Cassava))

#all coordinates
all.proj=ban.proj+acai.proj
#download cropmask 250m
dir.create("./TZ_cropmask")
download.file("https://www.dropbox.com/s/bjucbwpgexa3flc/TZ_cropmask_250m.zip?dl=0","./TZ_cropmask/TZ_cropmask_250m.zip",  mode="wb")
unzip("./TZ_cropmask/TZ_cropmask_250m.zip", exdir= "./TZ_cropmask", overwrite=T)

crp=raster("./TZ_cropmask/TZ_cropmask_250m.tif")
crp[crp==0]=NA

###### Regressions 
# set train/test set randomization seed
seed <- 1234
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split


#cassava for all TaNSIS and some ACAI points about  81% proportion of cropland in TZ over 38 %
cassIndex=createDataPartition(caspresabs$Cassava, p=2/3, list = FALSE, times=1)
cassTrain=caspresabs[cassIndex, ]
cassTest=caspresabs[-cassIndex,]
cassTest=na.omit(cassTest)


#____________
#set up data for caret
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)

#cassava glmnet for all points including non cropland
cassava.glm=train(Cassava ~ ., data=cassTrain, family= "binomial",
                method="glmnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.glm)
#variable importance
plot(varImp(cassava.glm,scale=F), main= "Variable Importance GLMnet")
#spatial predictions
cassglm.pred <- predict(t,cassava.glm, type="prob") 

mc <- makeCluster(detectCores())
registerDoParallel(mc)
# Cassava Random Forest
cassava.rf=train(Cassava ~ ., data=cassTrain, family= "binomial",
               method="rf",ntree= 501,metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.rf)
#variable importance
plot(varImp(cassava.rf,scale=F), main = " Variable Importance Random Forest")
#spatial predictions
cassrf.pred <- predict(t,cassava.rf, type="prob") 

#Cassava Gradient boosting 
cassava.gbm=train(Cassava ~ ., data=cassTrain,
                method="gbm", metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.gbm)
#variable importance
plot(varImp(cassava.gbm,scale=F))
#spatial predictions
cassgbm.pred <- predict(t,cassava.gbm, type="prob") 

#neural net cassava
cassava.nn=train(Cassava ~ ., data=cassTrain, family= "binomial",
               method="nnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.nn)
#variable importance
plot(varImp(cassava.nn,scale=F))
#spatial predictions
cassnn.pred <- predict(t,cassava.nn, type="prob") 


#ensemble regression glmnet (elastic net)
pred <- stack(1-cassglm.pred, 1-cassrf.pred, 
              1-cassgbm.pred, 1-cassnn.pred)
names(pred) <- c("cassglm","cassrf","cassgbm", "cassnn")
geospred <- extract(pred, all.proj)

# presence/absence of Cropland (present = Y, absent = N)
cassens <- cbind.data.frame(caspresabs$Cassava, geospred)
cassens <- na.omit(cassens)
cassensTest <- cassens[-cassIndex,] ## replicate previous test set
names(cassensTest)[1]= "cassava"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV

objControl <- trainControl(method='cv', number=10, 
                           classProbs = T,returnResamp='none')

# presence/absence of cassava (present = Y, absent = N)
cassava.ens <- train(cassava ~. , data = cassensTest,
                   family = "binomial", 
                   method = "glmnet", #glmnet gives an error
                   metric="Accuracy",
                   trControl = objControl)
confusionMatrix(cassava.ens) # print validation summaries on crossvalidation
cassens.pred <- predict(cassava.ens, cassensTest,  type="prob") ## predict test-set
cas.test <- cbind(cassensTest,cassens.pred)
casp <- subset(cas.test, cassava=="Y", select=c(Y))
casa <- subset(cas.test, cassava=="N", select=c(Y))
cas.eval <- evaluate(p=casp[,1], a=casa[,1]) ## calculate ROC's on test set <dismo>
cas.eval
plot(cas.eval, 'ROC')

objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
#for all data incl test
#cassava glmnet for all points including non cropland
cass.glm=train(Cassava ~ ., data=caspresabs, family= "binomial",
                method="glmnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cass.glm)
#variable importance
plot(varImp(cass.glm,scale=F))
#spatial predictions
cassglm.pred <- predict(t,cass.glm, type="prob") 
# cassava Random Forest
cass.rf=train(Cassava ~ ., data=caspresabs, family= "binomial",
               method="rf",ntree= 501,metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cass.rf)
#variable importance
plot(varImp(cass.rf,scale=F))
#spatial predictions
cassrf.pred <- predict(t,cass.rf, type="prob") 

#cassava Gradient boosting for all points including non cropland
cass.gbm=train(Cassava ~ ., data=caspresabs,
                method="gbm", metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cass.gbm)
#variable importance
plot(varImp(cass.gbm,scale=F))
#spatial predictions
cassgbm.pred <- predict(t,cass.gbm, type="prob") 

#neural net cassava
cass.nn=train(Cassava ~ ., data=caspresabs, family= "binomial",
               method="nnet",metric="ROC", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cass.nn)
#variable importance
plot(varImp(cass.nn,scale=F))
#spatial predictions
cassnn.pred <- predict(t,cass.nn, type="prob") 


#ensemble regression glmnet (elastic net)
pred <- stack( 1-cassglm.pred, 
              1-cassgbm.pred, 1-cassnn.pred)
names(pred) <- c("cassglm","cassgbm", "cassnn")
geospred <- extract(pred, all.proj)

# presence/absence of cassava (present = Y, absent = N)
cassens <- cbind.data.frame(caspresabs$Cassava, geospred)
cassens <- na.omit(cassens)
cassensTest <- cassens[-cassIndex,] ## replicate previous test set
names(cassensTest)[1]= "Cassava"
cassensTest=na.omit(cassensTest)
# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method='cv', number=10, 
                                  classProbs = T,returnResamp='none')

# presence/absence of cassava (present = Y, absent = N)
cass.ens <- train(Cassava ~. , data = cassensTest,
                   #family = "binomial", 
                   method = "gbm", #glmnet gives an error
                   trControl = ens)
confusionMatrix(cass.ens) # print validation summaries on crossvalidation
cassens.pred <- predict(cass.ens, cassensTest,  type="prob") ## predict test-set
cass.test <- cbind(cassensTest, cassens.pred)
cassp <- subset(cass.test, Cassava=="Y", select=c(Y))
cassa <- subset(cass.test, Cassava=="N", select=c(Y))
cass.eval <- evaluate(p=cassp[,1], a=cassa[,1]) ## calculate ROC's on test set <dismo>
cass.eval
cass.thld <- threshold(cass.eval, 'spec_sens') 
#spatial predictions
cassens.pred <- predict(pred,cass.ens, type="prob") 
plot(1-cassens.pred, main = "Ensemble prediction cassava Tanzania 2016")
dir.create("TZ_results", showWarnings=F)
writeRaster(1-cassens.pred, filename = "./TZ_results/TZ_cassavapred.tif", overwrite= TRUE )
cassmask=(1-cassens.pred)>cass.thld
plot(cassmask, main= "Predicted cassava distribution Tanzania 2016")
writeRaster(cassmask, filename="./TZ_results/TZ_cassavamask_250m.tif", overwrite=TRUE )
cassmask2=cassmask*crp #exclude cropmask
plot(cassmask2, main="Predicted Cassava on predicted cropland in Tanzania 2016")
freq(cassmask)
freq(cassmask2)