# Script for  cassava distribution  models TZ using ensemble regression
# grids are from Africasoils.net
# field data are collected by TanSIS in Tanzania 2015-16 
#Alex Verlinden November 2016 based on M. Walsh and J.Chen
# basis for cropland mask is the 12 k point survey for Tanzania of 2014-2015 conducted by AfSIS for Africa
# over 10500 field data are collected by TanSIS in Tanzania 2015 and 2016
#script in development to test crop distribution model based on presence/absence from crop scout
# area of TZ = 941761 km2
# estimated cropland present from 12 k Geosurvey observations = 38.6 %
#around 81% of TanSIS soil and crop survey is on cropland
#around 67 % of cropland has maize, only less than 5 % cassava
#using 1k m grids
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("TZ_crops1k", showWarnings=F)
dat_dir <- "./TZ_crops1k"
# download crop presence/absence locations
# these are data from 2015 crop scout ODK forms n= 10 k +
download.file("https://www.dropbox.com/s/02g8dmzvr18nyx3/Crop_TZ_JAN_2017.csv.zip?dl=0", "./TZ_crops/Crop_TZ_JAN_2017.csv.zip", mode="wb")
unzip("./TZ_crops/Crop_TZ_JAN_2017.csv.zip", exdir=dat_dir, overwrite=T)
# note that 0 and 1 are not ok for Caret for classifications, should be N and Y or similar
cas <- read.csv(paste(dat_dir, "/Crop_TZ_JAN_2017.csv", sep= ""), header=T, sep=",")

#download ACAI trial site data
download.file("https://www.dropbox.com/s/zldu22dnlrwquh5/Site%20description%20EZ%2007092016-PP.csv?dl=0", "./TZ_crops1k/Site%20description%20EZ%2007092016-PP.csv", mode="wb")
acai=read.csv(paste(dat_dir, "/Site%20description%20EZ%2007092016-PP.csv", sep="" ), header=T, sep=",")
#add Cassava presence
acai$Cassava= "Y"
#download grids for TZ  40 MB 1k m
download.file("https://www.dropbox.com/s/fwps69p6bl5747t/TZ_grids2.zip?dl=0","./TZ_crops/TZ_grids2.zip",  mode="wb")
unzip("./TZ_crops1k/TZ_grids2.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup for TZ crops--------------------------------------------------------------
# Project crop data to grid CRS
ban.proj <- as.data.frame(project(cbind(ban$X_gps_longitude, ban$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ban.proj) <- c("x","y")
coordinates(ban.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ban.proj) <- projection(grid)

#project acai data to grid TZ
acai.proj=as.data.frame(project(cbind(acai$Long.corr, acai$Lat.corr),"+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(acai.proj)=c("x", "y")
coordinates(acai.proj)= ~x+y
projection(acai.proj)=projection(grid)
# add points from TanSIS  and acai
allpts.proj=rbind(ban.proj,acai.proj)

# Extract gridded variables for all TZ survey data
allex=data.frame(coordinates(allpts.proj), extract(grid, allpts.proj))
allex= allex[,3:28]# exclude coordinates
# now bind crop species column to the covariates
# this has to change with every new crop
#use names (ban) to check crop name
#only mobile survey
banex=extract(grid,ban.proj)
caspresabs=cbind.data.frame(ban$root.cassava, banex)
colnames(caspresabs)[1]="cassava"
#all points
cas.list <- c(as.character(ban$root.cassava) ,
              as.character(acai$Cassava)  )
cas.list=as.factor(cas.list)
cassavapresabs=cbind.data.frame(cas.list,allex) # bind all points with extracted covariates
prop.table(table(cassavapresabs$cas.list))


#download cropmask
dir.create("./TZ_cropmask")
download.file("https://www.dropbox.com/s/nyvzq5a5v6v4io9/TZ_cropmask.zip?dl=0","./TZ_cropmask/TZ_cropmask.zip",  mode="wb")
unzip("./TZ_cropmask/TZ_cropmask.zip", exdir= "./TZ_cropmask", overwrite=T)

crp=raster("./TZ_cropmask/TZ_cropmask.tif")
crp[crp==0]=NA

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split

#cassava all points
cassavaIndex=createDataPartition(cassavapresabs$cas.list, p=2/3, list = FALSE, times=1)
cassavaTrain=cassavapresabs[cassavaIndex, ]
cassavaTest=cassavapresabs[-cassavaIndex,]
cassavaTest=na.omit(cassavaTest)


#____________
#set up data for caret
objControl <- trainControl(method='cv', number=10, 
                           classProbs = T,returnResamp='none')

#cassava for all points including non cropland elastic net
cassava.glm=train(cas.list ~ ., data=cassavaTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.glm)
#variable importance
plot(varImp(cassava.glm,scale=F))

#spatial predictions
cassavaglm.pred <- predict(grid,cassava.glm, type="prob") 
plot(1-cassavaglm.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
bancas=subset(ban, ban$root.cassava=="Y")
bancas.proj=as.data.frame(project(cbind(bancas$X_gps_longitude, bancas$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(bancas.proj) <- c("x","y")
coordinates(bancas.proj) <- ~x+y  #convert to Spatial DataFrame
projection(bancas.proj) <- projection(grid)
points(bancas.proj, cex=0.2, col="blue")

objControl <- trainControl(method='cv', number=10, 
                           classProbs = T,returnResamp='none')

#Random Forest
cassava.rf=train(cas.list ~ ., data=cassavaTrain, 
                 family= "binomial",method="rf",metric="Accuracy", 
                 ntree=501,trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.rf)
#variable importance
plot(varImp(cassava.rf,scale=F))
#spatial predictions
cassavarf.pred <- predict(grid,cassava.rf, type="prob") 
plot(1-cassavarf.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
points(bancas.proj, cex=0.12, col="blue")

#GBM
cassava.gbm=train(cas.list ~ ., data=cassavaTrain, 
                 #family= "binomial",
                 method="gbm",metric="Accuracy", 
                 trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.gbm)
#variable importance
plot(varImp(cassava.gbm,scale=F))
#spatial predictions
cassavagbm.pred <- predict(grid,cassava.gbm, type="prob") 
plot(1-cassavagbm.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
points(bancas.proj, cex=0.12, col="blue")

#neural net
cassava.nn=train(cas.list ~ ., data=cassavaTrain, 
                  family= "binomial",
                  method="nnet",metric="Accuracy", 
                  trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.nn)
#variable importance
plot(varImp(cassava.nn,scale=F))
#spatial predictions
cassavann.pred <- predict(grid,cassava.nn, type="prob") 
plot(1-cassavann.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
points(bancas.proj, cex=0.12, col="blue")

#ensemble regression glmnet (elastic net)
pred <- stack(1-cassavaglm.pred, 1-cassavarf.pred, 
              1-cassavagbm.pred, 1-cassavann.pred)
names(pred) <- c("casglm","casrf","casgbm", "casnn")
geospred <- extract(pred, allpts.proj)

# presence/absence of Cassava (present = Y, absent = N)
casens <- cbind.data.frame(cas.list, geospred)
casens <- na.omit(casens)
casensTest <- casens[-cassavaIndex,] ## replicate previous test set
names(casensTest)[1]= "Cassava"
casensTest=unique(na.omit(casensTest))
# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of cassava (present = Y, absent = N)
cas.ens <- train(Cassava ~. , data = casensTest,
                 #family = "binomial", 
                 method = "gbm", #glmnet has a bug here
                 trControl = ens)
confusionMatrix(cas.ens) # print validation summaries on crossvalidation
casens.pred <- predict(cas.ens, casensTest,  type="prob") ## predict test-set
cas.test <- cbind(casensTest, casens.pred)
casp <- subset(cas.test, Cassava=="Y", select=c(Y))
casa <- subset(cas.test, Cassava=="N", select=c(Y))
cas.eval <- evaluate(p=casp[,1], a=casa[,1]) ## calculate ROC's on test set <dismo>
cas.eval
plot(cas.eval, 'ROC')

#for all data

#set up data for caret
objControl <- trainControl(method='cv', number=10, 
                           classProbs = T,returnResamp='none')

#cassava for all points including non cropland elastic net
cassava.glm=train(cas.list ~ ., data=cassavapresabs, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.glm)
#variable importance
plot(varImp(cassava.glm,scale=F))

#spatial predictions
cassavaglm.pred <- predict(grid,cassava.glm, type="prob") 
plot(1-cassavaglm.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
bancas=subset(ban, ban$root.cassava=="Y")
bancas.proj=as.data.frame(project(cbind(bancas$X_gps_longitude, bancas$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(bancas.proj) <- c("x","y")
coordinates(bancas.proj) <- ~x+y  #convert to Spatial DataFrame
projection(bancas.proj) <- projection(grid)
points(bancas.proj, cex=0.2, col="blue")

objControl <- trainControl(method='cv', number=10, 
                           classProbs = T,returnResamp='none')

#Random Forest
cassava.rf=train(cas.list ~ ., data=cassavapresabs, 
                 family= "binomial",method="rf",metric="Accuracy", 
                 ntree=501,trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.rf)
#variable importance
plot(varImp(cassava.rf,scale=F))
#spatial predictions
cassavarf.pred <- predict(grid,cassava.rf, type="prob") 
plot(1-cassavarf.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
points(bancas.proj, cex=0.12, col="blue")

#GBM
cassava.gbm=train(cas.list ~ ., data=cassavapresabs, 
                  #family= "binomial",
                  method="gbm",metric="Accuracy", 
                  trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.gbm)
#variable importance
plot(varImp(cassava.gbm,scale=F))
#spatial predictions
cassavagbm.pred <- predict(grid,cassava.gbm, type="prob") 
plot(1-cassavagbm.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
points(bancas.proj, cex=0.12, col="blue")

#neural net
cassava.nn=train(cas.list ~ ., data=cassavapresabs, 
                 family= "binomial",
                 method="nnet",metric="Accuracy", 
                 trControl=objControl)
#confusionMatrix on cross validation
confusionMatrix(cassava.nn)
#variable importance
plot(varImp(cassava.nn,scale=F))
#spatial predictions
cassavann.pred <- predict(grid,cassava.nn, type="prob") 
plot(1-cassavann.pred)
points(acai.proj, cex=0.12, col= 'red')
points(ban.proj, cex=0.09, col="yellow")
points(bancas.proj, cex=0.12, col="blue")

#ensemble regression glmnet (elastic net)
pred <- stack(1-cassavaglm.pred, 1-cassavarf.pred, 
              1-cassavagbm.pred, 1-cassavann.pred)
names(pred) <- c("casglm","casrf","casgbm", "casnn")
geospred <- extract(pred, allpts.proj)

# presence/absence of Cassava (present = Y, absent = N)
casens <- cbind.data.frame(cas.list, geospred)
casens <- na.omit(casens)
casensTest <- casens[-cassavaIndex,] ## replicate previous test set
names(casensTest)[1]= "Cassava"
casensTest=unique(na.omit(casensTest))
# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of cassava (present = Y, absent = N)
cas.ens <- train(Cassava ~. , data = casensTest,
                 #family = "binomial", 
                 method = "gbm", #glmnet has a bug here
                 trControl = ens)
confusionMatrix(cas.ens) # print validation summaries on crossvalidation
casens.pred <- predict(cas.ens, casensTest,  type="prob") ## predict test-set
cas.test <- cbind(casensTest, casens.pred)
casp <- subset(cas.test, Cassava=="Y", select=c(Y))
casa <- subset(cas.test, Cassava=="N", select=c(Y))
cas.eval <- evaluate(p=casp[,1], a=casa[,1]) ## calculate ROC's on test set <dismo>
cas.eval
cas.thld <- threshold(cas.eval, 'spec_sens') 
#spatial predictions
casens.pred <- predict(pred, cas.ens, type="prob") 
plot(1-casens.pred, main = "Ensemble prediction Cassava Tanzania 2016")
dir.create("TZ_results", showWarnings=F)
writeRaster(1-casens.pred, filename = "./TZ_results/TZ_cassava1kpred.tif", overwrite= TRUE )
casmask=(1-casens.pred)>cas.thld
plot(casmask, main= "Predicted Cassava distribution Tanzania 2016")
writeRaster(casmask, filename="./TZ_results/TZ_casmask_1k.tif", overwrite=TRUE )
casmask2=casmask*crp #exclude cropmask
plot(casmask2, main="Predicted Cassava on predicted cropland in Tanzania 2016")
