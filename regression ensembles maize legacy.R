#Regression analysis for OFRA maize 0 and 50 kg N
rm(list=ls())
# install.packages(c("downloader","raster","rgdal","caret","MASS",randomForest","gbm","nnet")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(caret)
require(MASS)
require(randomForest)
require(gbm)
require(nnet)
library(rgdal)
library (maptools)
library(rgeos)
library(dismo)
library(sp)
library(gstat)
require(rpart)

#rm(list=ls()) #to remove what was previously loaded as there are issues when reloading modified shapefiles, 
#the old one remains and is not overwritten
#getwd()
#load tiffs of standardized variables
# Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("Data", showWarnings=F)
dat_dir <- "./Data"

# download Africa Gtifs (zipped ~ 850 Mb unzipped ~9  Gb) and stack in raster
download("https://www.dropbox.com/s/lp6sisnun4pw6nm/AF_grids_std.zip?dl=0", "./Data/AF_grids_std.zip", mode="wb")
unzip("./Data/AF_grids_std.zip", exdir="./Data", overwrite=T)
glist <- list.files(path="./Data", pattern="tif", full.names=T)
#stack grids 29 layers
grids <- stack(glist)

#read legacy data maize n=213 for N fertilizer only
download("https://www.dropbox.com/s/b5w0nn7bjhqll64/maize_responses.zip?dl=0", "./Data/maize_responses.zip", mode="wb")
unzip("./Data/maize_responses.zip", exdir="./Data", overwrite=T)
maize_resp=readShapeSpatial("./Data/maize_responses.shp") # n=213 samples
proj4string(maize_resp)=CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
plot(maize_resp, axes = TRUE)

#read legacy data maize n=956 for all trials with Y0 or control yield in T/ha
download("https://www.dropbox.com/s/0mh8jvc34yf47iw/Legacy_Maize_Y0.zip?dl=0", "./Data/Legacy_Maize_Y0.zip", mode="wb")
unzip("./Data/Legacy_Maize_Y0.zip", exdir="./Data", overwrite=TRUE)
maize_Y0=readShapeSpatial(paste(dat_dir,"/Legacy_Maize_Y0.shp", sep=""))#close to 1000 samples
proj4string(maize_Y0)=CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
maize_Y0_laea=spTransform(maize_Y0, CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")) # reproject vector file to LAEA
plot(maize_Y0_laea, axes=TRUE)

#data frame for all maize sites n=956
m2=as.data.frame(maize_Y0_laea)
#delete last field
m2=m2[,1:6]
mxy=m2[,1:2] #coords only
#+ Data setup --------------------------------------------------------------
# Project maize coords to grid CRS
coordinates(mxy) <- ~x+y
projection(mxy) <- projection(grids)

# Extract gridded variables at maize locations
mgrid <- extract(grids, mxy)

# Assemble dataframes

Y0 <- m2$Yo
mdat <- cbind.data.frame(Y0, mgrid)
mdat <- na.omit(mdat)
#dataset with coordinates
mdatc=cbind.data.frame(Y0,mxy, mgrid) 
mdatc=na.omit(mdatc)
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Cropland train/test split
index<- createDataPartition(mdatc$Y0, p = 0.75, list = FALSE, times = 1)
mTrain <- mdat [ index,]
mTest  <- mdat [-index,]
#Train and test sets with coordinates
mTrainc=mdatc[ index,]
mTestc=mdatc[-index,]
mTestlocs= mTestc[,2:3]
#+ Stepwise main effects GLM's <MASS> --------------------------------------
# 5-fold CV
step <- trainControl(method = "cv", number = 5)

# GLM all maize
maize.glm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                 family = gaussian, 
                 method = "glmStepAIC",
                 trControl = step)
maizeglm.pred <- predict(grids, maize.glm) ## spatial predictions
x=exp(maizeglm.pred)
x[x>=8]=8
plot (x)
points(m2, cex=0.2, col="red")
locs.train=mTrainc[,2:3]
points(locs.train, cex=0.2, col ="blue")

#for Malawi
#Y0test=test$Yo #selecting control yields 
#locstest=test[,1:2] #selecting coordinates for test sites

#check below not necessary I think should be for Malawi
#extract covariates from training set = control Yields
coordinates(mTrainc) <- ~x+y
projection(mTrainc) <- projection(grids)
traingrid <- extract(grids, mTrainc)
Y0 <- train$Yo
ycdat <- data.frame(cbind(Y0, traingrid))
ycdat <- na.omit(ycdat)

#for Y50/Y0
coordinates(train50) <- ~x+y
projection(train50) <- projection(grids)
traingrid50 <- extract(grids, train50)
Y50=train50$Y50.Y0
y50dat=data.frame(cbind(Y50, traingrid50))
y50dat <- na.omit(y50dat)
# Regression models Malawi-------------------------------------------------------
# Stepwise main effects GLM's this should be for Malawi, check data
# Control yield predictions (Yc) of training set
Yc.glm <- glm(Y0 ~ ., family=gaussian(link="log"), ycdat)
Yc.step <- stepAIC(Yc.glm)
summary(Yc.step)
ycglm <- predict(grids, Yc.step, type="response")
ycglm[ycglm>8]=8 #there are a few pixels with very high estimates, limit set to 8 tons/ha
plot(ycglm)

#predictions of 50kg N of training set with glm
Y50.glm <- glm(Y50 ~ ., family=gaussian(link="log"), y50dat)
Y50.step <- stepAIC(Y50.glm)
summary(Y50.step)
y50glm=predict(grids, Y50.step, type="response")
y50glm[y50glm>7]=7 #again pixels with high estimates, soils and GYGA covariates might be iffy
plot(y50glm)

# now compare model predictions with test data for control Yields (these are measured values in the field)
Ypred=extract(ycglm, locstest)
plot(Yctest, Ypred, xlim=c(0, 8), ylim=c(0,8))
x=lm(Yctest~Ypred+0)
abline(lm(Yctest~Ypred+0), col="red")
# test with test data on 50kg N
Y50pred=extract(y50glm, locstest)
plot(Y50test, Y50pred, xlim=c(0, 3), ylim=c(0,3))
x50=lm(Y50test~Y50pred+0)
abline(x50, col="red")
#reverse partitioning regression with rpart on Yc

# Regression trees for all maize
# Control yield predictions (Yc)
Yc.rt <- rpart(log(mTrainc$Y0) ~ ., data=mTrain)
ycrt <- predict(grids, Yc.rt)
plot(ycrt)
points(m2, cex =0.2, col="red")
points(locs.train, cex= 0.2, col= "blue")
# now confirm with test data
Ypred3=extract(ycrt, locstest)
plot(Yctest, Ypred3, xlim=c(0, 8), ylim=c(0,4))
xrt=lm(Ypred3~Yctest+0)
abline(xrt, col="red")
abline(lm(Ypred3~Yctest), col="blue")

#50kgN yield response ratios
Y50.rt= rpart(log(Y50) ~., data=y50dat)
Y50rt=predict(grids, Y50.rt)
#rt test with test sites not done, concentrate on ensembles
# Random forests (no tuning default) for all maize
# Control yield predictions (Yc)
Yc.rf <- randomForest(log(mTrainc$Y0) ~ ., importance=T, proximity=T, data=mTrain)
ycrf <- predict(grids, Yc.rf)
plot (ycrf)
points(m2, cex=0.2, col= "red")
points(locs.train, cex= 0.2, col= "blue")
#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# gbm
Yc.gbm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                 method = "gbm",
                 trControl = gbm)
Y0.gbm <- predict(grids,Yc.gbm) ## predict train-set
locs.train=mTrainc[,2:3]
points(m2, cex=0.2, col= "red")
points(locs.train,cex=0.2, col="blue")


#50kg N prediction with Random Forests
Y50.rf=randomForest(log(Y50)~., data=y50dat)
Y50rf=predict(grids, Y50.rf)
plot(Y50rf)
# now compare predictions of RandomForest with test data
Ypred2=extract(ycrf, locstest) #extraction of predictions at test locations
plot(Yctest, Ypred2, xlim=c(0, 8), ylim=c(0,2))
xrf=lm(Ypred2~Yctest+0) #regression through origin
abline(xrf, col="red")
abline(lm(Ypred2~Yctest), col="blue")



#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# neural network of control yields Maize
Y0.nn <- train(log(mTrain$Y0) ~ ., data = mTrain,
               method = "nnet",
               trControl = nn)
Yc.nn <- predict(grids,Y0.nn)
points(m2, cex=0.2, col= "red")
points(locs.train,cex=0.2, col="blue")

#regression ensembles
# Test set ensemble predictions for Yc
#largely based on Markus------------------------------------------
#bring first 5 regression predictions together in a stack
ycpred <- stack(maizeglm.pred, ycrt, ycrf, Y0.gbm, Yc.nn)
names(ycpred) <- c("ycglm", "ycrt", "ycrf", "ycgbm", "ycnn")
exyc <- extract(ycpred, locs.train) 
#exyc_train= extract(ycpred, traindat) #extract predictions from regressions
exyc=as.data.frame(exyc)
#exyc=na.omit(as.data.frame(exyc))
#test on means (really not necessary as weighting is preferred?)
exycmean <- transform(exyc, Col4 = rowMeans(exyc, na.rm = TRUE)) #taking mean predictions in a separate columns
plot(Yctest, exycmean$Col4, xlim=c(0, 7), ylim=c(0,4))
exycmn=lm(exycmean$Col4~Yctest+0)
abline(exycmn, col="red")

# do Ensemble control yield predictions (Yc) on train set
#Ycwgt.glm <- na.omit(glm(Y0~log(ycglm)+log(ycrt)+log(ycrf), family=gaussian(link="log"), data=exycmean))
#data=exycmean, start= 1,1,1,1))
# Regularized ensemble weighting on the test set <test>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)
#Ycwgt.glm <- train(glm($Y0~maizeglm.pred+ycrt+ycrf+Y0.gbm+Yc.nn,family=gaussian(link="log"), 
                         data = mTest, 
                         method = "glmnet",
                         trControl = ens))  

Ycwgt.step <- stepAIC(Ycwgt.glm)
summary(Ycwgt.step)
#plot(Yctest~fitted(Ycwgt.step), exycmean) # does not work - fitted function
ycwgt <- predict(ycpred, Ycwgt.step, type="response")
quantile(ycwgt, prob=c(0.025,0.25,0.5,0.75,0.975)) # don't really get this
plot(ycwgt)
Ycwgt_extr=extract(ycwgt, locstest)
plot(Yctest,Ycwgt_extr, xlim=c(0, 7), ylim=c(0,6))
ens=lm(Ycwgt_extr~Yctest+0)
abline(ens, col="red")

# do same test for Y50
y50pred <- stack(y50glm, Y50rt, Y50rf)
names(y50pred) <- c("y50glm", "Y50rt", "Y50rf")
exy50 <- extract(y50pred, locstest)

# Overlay training set predictions w. test data
exy50 <- data.frame(cbind(Y50test, exy50))
exy50 <- na.omit(exy50)
Y50wgt.glm=glm(Y50test ~ ., family= gaussian(link="log"), data=exy50)
Y50wgt.step=stepAIC(Y50wgt.glm)
y50wgt=predict(y50pred, Y50wgt.step, type="response")


#test
Y50wgtextr=extract(y50wgt, locstest)
plot(Y50test,Y50wgtextr, xlim=c(0, 3), ylim=c(0,3))
ens=lm(Y50wgtextr~Y50test+0)
abline(ens, col="red")

#following not part of the analysispoint
# test sites malawi
# Create a "Data" folder in your current working directory
dir.create("Data", showWarnings=F)
dat_dir <- "./Data"

# LREP fertilizer response data download to "./Data"
download("https://www.dropbox.com/s/i4dby04fl9j042a/MW_fert_trials.zip?dl=0", "./Data/MW_fert_trials.zip", mode="wb")
unzip("./Data/MW_fert_trials.zip", exdir="./Data", overwrite=T)
mwsite <- read.table(paste(dat_dir, "/Location.csv", sep=""), header=T, sep=",")
mtrial <- read.table(paste(dat_dir, "/Trial.csv", sep=""), header=T, sep=",")

# Georeference and specify site ID's --------------------------------------
# Project to Africa LAEA from UTM36S
mw <- cbind(mwsite$Easting, mwsite$Northing)
tr <- ptransform(mw, '+proj=utm +zone=36 +south +datum=WGS84 +units=m +no_defs', '+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs')
colnames(tr) <- c("x","y","z")
mwsite <- cbind(mwsite, tr)

#get average Yields for LIDs
trialsites=aggregate(mtrial$Yc~LID, mtrial, mean)
colnames(trialsites)= c("LID","Yc")
trialsites$Yc=trialsites$Yc/1000
# this links the control yields with trialsites thorugh LIDs
mwsite= cbind(mwsite, trialsites)
testlocs=mwsite[,4:5]
#overlay with test data
# this extract the estimates
exyc <- extract(maize, testlocs)
exyc <- data.frame(cbind(mwsite$Yc, exyc))

# Regression models -------------------------------------------------------

# GLM
require(MASS)
pres.glm <- glm(pb ~ ., family = binomial(link="logit"), data=presback)
step <- step(pres.glm)
pglm2=predict(grids, step, type="response", data=presback)
plot(xlab="Easting (m)", ylab="Northing (m)")
plot(pglm2, add=T)
points(maize_random, pch=3, col="black", cex=0.5)
points(m, pch=21, col="red", bg="red")
rf=writeRaster(maize, filename="maize_Yc", format= "GTiff", overwrite=TRUE)


#variogram for Y50.Y0  is response ratio N50/control yield
coordinates(regr2)=~x+y
yc.var=variogram(regr2$Y50.Y0~1, regr2)
yc.fit <- fit.variogram(yc.var, model = vgm(1, "Sph", 500000 , 0.3))
yc.var2=variogram(regr2$Y50.Y0~regr2$Lat+regr2$BSAN+regr2$seasons_AfSIS+regr2$BSAS+regr2$REF1+regr2$BSAV+regr2$TMFI+regr2$EVI+regr2$ELEV+regr2$REF7, regr2)
plot(variogram(maize.lm2$residuals~1, regr2))

#prediction with ordinary kriging
resp.k=krige(regr2$Y50.Y0, grids, yc.fit)

#write tiff
rf=writeRaster(maize50,filename="maize_50N", format= "GTiff", overwrite = TRUE)