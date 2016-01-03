# Script for crop distribution  models TZ using glmnet
# basis for cropland mask is the 1 Million point survey for Africa of 2014-2015 conducted by AfSIS for Africa
# grids are from Africasoils.net
# field data are collected by TanSIS in Tanzania 2015 
#script in development to test crop distribution model with glmnet based on presence/absence from crop scout
# Alex Verlinden November 2015 based on M. Walsh and J.Chen
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret", e1071")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(e1071)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("TZ_crops", showWarnings=F)
dat_dir <- "./TZ_crops"
# download crop presence/absence locations
# these are data from 2015 crop scout ODK forms n= 2087
download.file("https://www.dropbox.com/s/du2xye4njujypc1/Crop_NTZ_march_2016.csv?dl=0", "./TZ_crops/Crop_NTZ_march_2016.csv", mode="wb")

# note that 0 and 1 are not ok for Caret for classifications, should be N and Y or similar
ban <- read.csv(paste(dat_dir, "/Crop_NTZ_march_2016.csv", sep= ""), header=T, sep=",")

#download grids for TZ  40 MB
download.file("https://www.dropbox.com/s/fwps69p6bl5747t/TZ_grids2.zip?dl=0","./TZ_crops/TZ_grids2.zip",  mode="wb")
unzip("./TZ_crops/TZ_grids2.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup for TZ crops--------------------------------------------------------------
# Project crop data to grid CRS
ban.proj <- as.data.frame(project(cbind(ban$X_gps_longitude, ban$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ban.proj) <- c("x","y")
coordinates(ban.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ban.proj) <- projection(grid)

# Extract gridded variables for TZ data observations 
banex <- data.frame(coordinates(ban.proj), extract(grid, ban.proj))
banex=  banex[,3:28] #exclude coordinates
# now bind crop species column to the covariates
# this has to change with every new crop
#use names (ban) to check crop name
#for banana
banpresabs=cbind(ban$Banana, banex)
banpresabs=na.omit(banpresabs)
colnames(banpresabs)[1]="ban"
banpresabs$ban=as.factor(banpresabs$ban)
summary(banpresabs)
#for pigeon pea
pigpresabs=cbind(ban$legume.pigeonpea, banex)
pigpresabs=na.omit(pigpresabs)
colnames(pigpresabs)[1]="pig_pea"
pigpresabs$pig_pea=as.factor(pigpresabs$pig_pea)

# for maize
maizepresabs=cbind(ban$maize, banex)
maizepresabs=na.omit(maizepresabs)
colnames(maizepresabs)[1]="maize"
maizepresabs$maize=as.factor(maizepresabs$maize)

#for wheat
whtpresabs=cbind(ban$wheat, banex)
whtpresabs=na.omit(whtpresabs)
colnames(whtpresabs)[1]="wheat"
whtpresabs$wheat=as.factor(whtpresabs$wheat)

#for sunflower
sunpresabs=cbind(ban$Sunflower, banex)
sunpresabs=na.omit(sunpresabs)
colnames(sunpresabs)[1]="sunflower"
sunpresabs$sunflower=as.factor(sunpresabs$sunflower)

#for Green beans
beanpresabs=cbind(ban$legume.beans, banex)
beanpresabs=na.omit(beanpresabs)
colnames(beanpresabs)[1]="bean"
beanpresabs$bean=as.factor(beanpresabs$bean)

#for cowpea
cowpresabs=cbind(ban$legume.cowpea, banex)
cowpresabs=na.omit(cowpresabs)
colnames(cowpresabs)[1]="cowpea"
cowpresabs$cowpea=as.factor(cowpresabs$cowpea)

#for soybean
soypresabs=cbind(ban$legume.soybeans, banex)
soypresabs=na.omit(soypresabs)
colnames(soypresabs)[1]="soybean"
soypresabs$soybean=as.factor(soypresabs$soybean)

#for cassava
caspresabs=cbind(ban$root.cassava, banex)
caspresabs=na.omit(caspresabs)
colnames(caspresabs)[1]="cassava"
caspresabs$cassava=as.factor(caspresabs$cassava)

#for rice
ricepresabs=cbind(ban$rice, banex)
ricepresabs=na.omit(ricepresabs)
colnames(ricepresabs)[1]="rice"
ricepresabs$rice=as.factor(ricepresabs$rice)

#for sorghum
sgpresabs=cbind(ban$sorghum, banex)
sgpresabs=na.omit(sgpresabs)
colnames(sgpresabs)[1]="sorghum"
sgpresabs$sorghum=as.factor(sgpresabs$sorghum)

#for millet
milpresabs=cbind(ban$millet, banex)
milpresabs=na.omit(milpresabs)
colnames(milpresabs)[1]="millet"
milpresabs$millet=as.factor(milpresabs$millet)

#download cropmask
download.file("https://www.dropbox.com/s/nyvzq5a5v6v4io9/TZ_cropmask.zip?dl=0","./TZ_crops/TZ_cropmask.zip",  mode="wb")
unzip("./TZ_crops/TZ_cropmask.zip", exdir=dat_dir, overwrite=T)

crp=raster("./TZ_crops/TZ_cropmask.tif")
crp[crp==0]=NA

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Crop type train/test split
#banana
banIndex <- createDataPartition(banpresabs$ban, p = 2/3, list = FALSE, times = 1)
banTrain <- banpresabs[ banIndex,]
banTest  <- banpresabs[-banIndex,]
banTest= na.omit(banTest)
#to test if crop is a rare event as presences of much less than 15 % are difficult to model
prop.table(table(banpresabs$ban))
#print structure
print(str(banpresabs))
#pigeon pea
pigIndex=createDataPartition(pigpresabs$pig_pea, p = 2/3, list = FALSE, times = 1)
pigTrain <- pigpresabs[ pigIndex,]
pigTest  <- pigpresabs[-pigIndex,]
pigTest= na.omit(pigTest)

#soybean
soyIndex=createDataPartition(soypresabs$soybean, p = 2/3, list = FALSE, times = 1)
soyTrain <- soypresabs[ soyIndex,]
soyTest  <- soypresabs[-soyIndex,]
soyTest= na.omit(soyTest)

#maize
maizeIndex=createDataPartition(maizepresabs$maize, p = 2/3, list = FALSE, times = 1)
maizeTrain <- maizepresabs[ maizeIndex,]
maizeTest  <- maizepresabs[-maizeIndex,]
maizeTest= na.omit(maizeTest)

#wheat
whtIndex=createDataPartition(whtpresabs$wheat, p = 2/3, list = FALSE, times = 1)
whtTrain <- whtpresabs[ whtIndex,]
whtTest  <- whtpresabs[-whtIndex,]
whtTest= na.omit(whtTest)

#millet (pearl millet)
milIndex=createDataPartition(milpresabs$millet, p = 2/3, list = FALSE, times = 1)
milTrain <- milpresabs[ milIndex,]
milTest  <- milpresabs[-milIndex,]
milTest= na.omit(milTest)

# Sunflower
sunIndex=createDataPartition(sunpresabs$sunflower, p = 2/3, list = FALSE, times = 1)
sunTrain <- sunpresabs[ sunIndex,]
sunTest  <- sunpresabs[-sunIndex,]
sunTest= na.omit(sunTest)

#Bean
beanIndex=createDataPartition(beanpresabs$bean, p = 2/3, list = FALSE, times = 1)
beanTrain <- beanpresabs[ beanIndex,]
beanTest  <- beanpresabs[-beanIndex,]
beanTest= na.omit(beanTest)
#cassava
casIndex=createDataPartition(caspresabs$cassava, p = 2/3, list = FALSE, times = 1)
casTrain <- caspresabs[ casIndex,]
casTest  <- caspresabs[-casIndex,]
casTest= na.omit(casTest)
#rice
riceIndex=createDataPartition(ricepresabs$rice, p = 2/3, list = FALSE, times = 1)
riceTrain <- ricepresabs[ riceIndex,]
riceTest  <- ricepresabs[-riceIndex,]
riceTest= na.omit(riceTest)
#sorghum
sgIndex=createDataPartition(sgpresabs$sorghum, p = 2/3, list = FALSE, times = 1)
sgTrain <- sgpresabs[ sgIndex,]
sgTest  <- sgpresabs[-sgIndex,]
sgTest= na.omit(sgTest)
#cowpea
cowIndex=createDataPartition(cowpresabs$cowpea, p = 2/3, list = FALSE, times = 1)
cowTrain <- cowpresabs[ cowIndex,]
cowTest  <- cowpresabs[-cowIndex,]
cowTest= na.omit(cowTest)

#____________
#set up data for caret
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for banana
ban.glm=train(ban ~ ., data=banTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(ban.glm, banTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(ban.glm)
#variable importance
plot(varImp(ban.glm,scale=F))

bantest=cbind(banTest, predictions)
banp=subset(bantest, ban=="Y", select=c(Y) )
bana=subset(bantest, ban=="N", select=c(Y))
ban.eval=evaluate(p=banp[,1],a=bana[,1])
ban.eval
plot(ban.eval, 'ROC') ## plot ROC curve
ban.thld <- threshold(ban.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
banglm.pred <- predict(grid,ban.glm, type="prob") 
banglmnet.pred=1-banglm.pred
banmask=banglmnet.pred>ban.thld
banmask2=banmask*crp
plot(banmask2, legend=F)

#output for banana
dir.create("TZ_results", showWarnings=F)
rf=writeRaster(banmask2, filename="./TZ_results/TZ_banana_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for pigeon pea
pig.glm=train(pig_pea ~ ., data=pigTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(pig.glm, pigTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(pig.glm)
#variable importance
plot(varImp(pig.glm,scale=F))

pigtest=cbind(pigTest, predictions)
pigp=subset(pigtest, pig_pea=="Y", select=c(Y) )
piga=subset(pigtest, pig_pea=="N", select=c(Y))
pig.eval=evaluate(p=pigp[,1],a=piga[,1])
pig.eval
plot(pig.eval, 'ROC') ## plot ROC curve
pig.thld <- threshold(pig.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
pigglm.pred <- predict(grid,pig.glm, type="prob") 
pigglmnet.pred=1-pigglm.pred
pigmask=pigglmnet.pred>pig.thld
pigmask2=pigmask*crp
plot(pigmask2, legend=F)

#write pigeon pea results
rf=writeRaster(pigmask2.pred, filename="./TZ_results/TZ_pigeon_pea_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for maize
mz.glm=train(maize ~ ., data=maizeTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(mz.glm, maizeTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mz.glm)
#variable importance
plot(varImp(mz.glm,scale=F))

maizetest=cbind(maizeTest, predictions)
maizep=subset(maizetest, maize=="Y", select=c(Y) )
maizea=subset(maizetest, maize=="N", select=c(Y))
mz.eval=evaluate(p=maizep[,1],a=maizea[,1])
mz.eval
plot(mz.eval, 'ROC') ## plot ROC curve
mz.thld <- threshold(mz.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
mzglm.pred <- predict(grid,mz.glm, type="prob") 
mzglmnet.pred=1-mzglm.pred
mzmask=mzglmnet.pred>mz.thld
mzmask2=mzmask*crp
plot(mzmask2, legend=F)

#write maize results
rf=writeRaster(mzmask2.pred, filename="./TZ_results/TZ_maize_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for beans
bean.glm=train(bean ~ ., data=beanTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(bean.glm, beanTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(bean.glm)
#variable importance
plot(varImp(bean.glm,scale=F))

beantest=cbind(beanTest, predictions)
beanp=subset(beantest, bean=="Y", select=c(Y) )
beana=subset(beantest, bean=="N", select=c(Y))
bean.eval=evaluate(p=beanp[,1],a=beana[,1])
bean.eval
plot(bean.eval, 'ROC') ## plot ROC curve
bean.thld <- threshold(bean.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
beanglm.pred <- predict(grid,bean.glm, type="prob") 
beanglmnet.pred=1-beanglm.pred
beanmask=beanglmnet.pred>bean.thld
beanmask2=beanmask*crp
plot(beanmask2, legend=F)

#write bean results
rf=writeRaster(beanmask2.pred, filename="./TZ_results/TZ_bean_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for cassava
cas.glm=train(cassava ~ ., data=casTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(cas.glm, casTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(cas.glm)
#variable importance
plot(varImp(cas.glm,scale=F))

castest=cbind(casTest, predictions)
casp=subset(castest, cassava=="Y", select=c(Y) )
casa=subset(castest, cassava=="N", select=c(Y))
cas.eval=evaluate(p=casp[,1],a=casa[,1])
cas.eval
plot(cas.eval, 'ROC') ## plot ROC curve
cas.thld <- threshold(cas.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
casglm.pred <- predict(grid,cas.glm, type="prob") 
casglmnet.pred=1-casglm.pred
casmask=casglmnet.pred>cas.thld
casmask2=casmask*crp
plot(casmask2, legend=F)

#write cassava results
rf=writeRaster(casmask2.pred, filename="./TZ_results/TZ_cassava_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for sunflower
sun.glm=train(sunflower~., data=sunTrain, family="binomial", method = "glmnet", metric = "Accuracy", trControl=objControl)
predictions=predict(sun.glm, sunTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(sun.glm)
#variable importance
plot(varImp(sun.glm,scale=F))
#evaluate prediction
suntest=cbind(sunTest, predictions)
sunp=subset(suntest, sunflower=="Y", select=c(Y) )
suna=subset(suntest, sunflower=="N", select=c(Y))
sun.eval=evaluate(p=sunp[,1],a=suna[,1])
sun.eval
plot(sun.eval, 'ROC') ## plot ROC curve
sun.thld <- threshold(sun.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
sunglm.pred <- predict(grid,sun.glm, type="prob") 
sunglmnet.pred=1-sunglm.pred
sunmask=sunglmnet.pred>sun.thld
sunmask2=sunmask*crp
plot(sunmask2, legend=F)


#glmnet using binomial distribution for rice
rice.glm=train(rice ~ ., data=riceTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(rice.glm, riceTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(rice.glm)
#variable importance
plot(varImp(rice.glm,scale=F))

ricetest=cbind(riceTest, predictions)
ricep=subset(ricetest, rice=="Y", select=c(Y) )
ricea=subset(ricetest, rice=="N", select=c(Y))
rice.eval=evaluate(p=ricep[,1],a=ricea[,1])
rice.eval
plot(rice.eval, 'ROC') ## plot ROC curve
rice.thld <- threshold(rice.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
riceglm.pred <- predict(grid, rice.glm, type="prob") 
riceglmnet.pred=1-riceglm.pred
ricemask=riceglmnet.pred>rice.thld
ricemask2=ricemask*crp
plot(ricemask2, legend=F)

#write rice results
rf=writeRaster(ricemask2.pred, filename="./TZ_results/TZ_rice_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for sorghum
sg.glm=train(sorghum ~ ., data=sgTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(sg.glm, sgTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(sg.glm)
#variable importance
plot(varImp(sg.glm,scale=F))

sgtest=cbind(sgTest, predictions)
sgp=subset(sgtest, sorghum=="Y", select=c(Y) )
sga=subset(sgtest, sorghum=="N", select=c(Y))
sg.eval=evaluate(p=sgp[,1],a=sga[,1])
sg.eval
plot(sg.eval, 'ROC') ## plot ROC curve
sg.thld <- threshold(sg.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
sgglm.pred <- predict(grid, sg.glm, type="prob") 
sgglmnet.pred=1-sgglm.pred
sgmask=sgglmnet.pred>sg.thld
sgmask2=sgmask*crp
plot(sgmask2, legend=F)

#write sorghum results
rf=writeRaster(sgmask2.pred, filename="./TZ_results/TZ_sorghum_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for cowpea
cow.glm=train(cowpea ~ ., data=cowTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(cow.glm, cowTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(cow.glm)
#variable importance
plot(varImp(cow.glm,scale=F))

cowtest=cbind(cowTest, predictions)
cowp=subset(cowtest, cowpea=="Y", select=c(Y) )
cowa=subset(cowtest, cowpea=="N", select=c(Y))
cow.eval=evaluate(p=cowp[,1],a=cowa[,1])
cow.eval
plot(cow.eval, 'ROC') ## plot ROC curve
cow.thld <- threshold(cow.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cowglm.pred <- predict(grid, cow.glm, type="prob") 
cowglmnet.pred=1-cowglm.pred
cowmask=cowglmnet.pred>cow.thld
cowmask2=cowmask*crp
plot(cowmask2, legend=F)

#write cowpea results
rf=writeRaster(cowmask2.pred, filename="./TZ_results/TZ_cowpea_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for soybean
soy.glm=train(soybean ~ ., data=soyTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(soy.glm, soyTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(soy.glm)
#variable importance
plot(varImp(soy.glm,scale=F))

soytest=cbind(soyTest, predictions)
soyp=subset(soytest, soybean=="Y", select=c(Y) )
soya=subset(soytest, soybean=="N", select=c(Y))
soy.eval=evaluate(p=soyp[,1],a=soya[,1])
soy.eval
plot(soy.eval, 'ROC') ## plot ROC curve
soy.thld <- threshold(soy.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
soyglm.pred <- predict(grid, soy.glm, type="prob") 
soyglmnet.pred=1-soyglm.pred
soymask=soyglmnet.pred>soy.thld
soymask2=soymask*crp
plot(soymask2, legend=F)

#write soybean results
rf=writeRaster(soymask2.pred, filename="./TZ_results/TZ_soybean_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for wheat
wht.glm=train(wheat ~ ., data=whtTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(wht.glm, whtTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(wht.glm)
#variable importance
plot(varImp(wht.glm,scale=F))

whttest=cbind(whtTest, predictions)
whtp=subset(whttest, wheat=="Y", select=c(Y) )
whta=subset(whttest, wheat=="N", select=c(Y))
wht.eval=evaluate(p=whtp[,1],a=whta[,1])
wht.eval
plot(wht.eval, 'ROC') ## plot ROC curve
wht.thld <- threshold(wht.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
whtglm.pred <- predict(grid, wht.glm, type="prob") 
whtglmnet.pred=1-whtglm.pred
whtmask=whtglmnet.pred>wht.thld
whtmask2=whtmask*crp
plot(whtmask2, legend=F)

#write wheat results
rf=writeRaster(whtmask2.pred, filename="./TZ_results/TZ_wheat_2015_glm", format= "GTiff", overwrite=TRUE)

#glmnet using binomial distribution for millet
mil.glm=train(millet ~ ., data=milTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(mil.glm, milTest[,2:27], type="prob")
#confusionMatrix on cross validation
confusionMatrix(mil.glm)
#variable importance
plot(varImp(mil.glm,scale=F))

miltest=cbind(milTest, predictions)
milp=subset(whttest, millet=="Y", select=c(Y) )
mila=subset(whttest, millet=="N", select=c(Y))
mil.eval=evaluate(p=milp[,1],a=mila[,1])
mil.eval
plot(mil.eval, 'ROC') ## plot ROC curve
mil.thld <- threshold(mil.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
milglm.pred <- predict(grid, mil.glm, type="prob") 
milglmnet.pred=1-milglm.pred
milmask=milglmnet.pred>mil.thld
milmask2=milmask*crp
plot(milmask2, legend=F)

#write millet results
rf=writeRaster(milmask2.pred, filename="./TZ_results/TZ_millet_2015_glm", format= "GTiff", overwrite=TRUE)