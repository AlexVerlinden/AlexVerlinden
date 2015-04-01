#Regression analysis for OFRA maize no fertilizer
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
library(maptools)
library(rgeos)
library(dismo)
library(sp)
library(gstat)
require(rpart)
library(gam)

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

#read march 2015 legacy data
download("https://www.dropbox.com/s/50jl10nz0lqessu/maize_Y0_legacy_march2015.zip?dl=0", "./Data/maize_Y0_legacy_march2015.zip")
unzip("./Data/maize_Y0_legacy_march2015.zip", exdir="./Data", overwrite=TRUE)
maizeY0= readShapeSpatial(paste(dat_dir,"/Legacy_maize_2015_march.shp", sep=""))
proj4string(maizeY0)=CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
maize_Y0=spTransform(maizeY0, CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")) # reproject vector file to LAEA
plot(maize_Y0, axes=TRUE)

#data frame for all maize sites n=325
m2_2015=as.data.frame(maize_Y0)
m2=m2_2015[c("x","y", "Grain_Yiel")]
m2xy=m2[,1:2] #coords only


#for LREP data Malawi
#download LREP data
download("https://www.dropbox.com/s/rra8c3gcx8bjjnn/MW_fert_trials.zip?dl=0","./Data/MW_fert_trials.zip", mode="wb")

unzip("./Data/MW_fert_trials.zip", exdir="./Data", overwrite=TRUE)
mwsite <- read.table(paste(dat_dir, "/Location.csv", sep=""), header=TRUE, sep=",")
mwtrial <- read.table(paste(dat_dir, "/Trial.csv", sep=""), header=TRUE, sep=",")
test= merge(mwsite, mwtrial)
#get duplicate IDs = to remove treatments beyond y baseline yields
dups= duplicated(test["ID"])

#remove duplicate IDs
maize_LREP=test[!dups,]

#get duplicate coordinates
#dups <- duplicated(test[, c( 'Easting' ,  'Northing')])
#remove duplicate coordinates
#maize_LREP=test[!dups,]
#select 5% random sample
set.seed(12345)
samp=sample(nrow(maize_LREP), round(0.05*nrow(maize_LREP)))
maize_LREP=maize_LREP[samp,]
coordinates(maize_LREP)= ~Easting+Northing
proj4string(maize_LREP)= CRS('+proj=utm +zone=36 +south +datum=WGS84 +units=m +no_defs')
maize_LREP_laea=spTransform(maize_LREP, CRS("+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")) # reproject vector file to LAEA
maize_LREP_laea=as.data.frame(maize_LREP_laea)
maize_LREP= cbind(maize_LREP_laea$x,maize_LREP_laea$y, maize_LREP_laea$Yc/1000)
colnames(maize_LREP)=c("x", "y","Grain_Yiel")
maize_LREP= as.data.frame(maize_LREP)
# merge OFRA and LREP

OFRA_LREP=rbind(m2,maize_LREP)
#+ Data setup --------------------------------------------------------------
#for all samples baseline yield (control yields)

# Project maize coords to grid CRS for n=2002 # is projection used by AfSIS
OFRA_LREPxy= OFRA_LREP[,1:2]
#OFRA_LREPxy=as.data.frame(OFRA_LREPxy)
coordinates(OFRA_LREPxy) <- ~x+y
projection(OFRA_LREPxy) <- projection(grids)


# Extract gridded variables at maize locations for n =2002
mgrid = extract (grids, OFRA_LREPxy)

# Assemble dataframes for 
Y0=OFRA_LREP$Grain_Yiel
mdat <- cbind.data.frame(Y0, mgrid)
mdat <- na.omit(mdat)
#dataset with coordinates
mdatc=cbind.data.frame(Y0,OFRA_LREPxy, mgrid) 
mdatc=na.omit(mdatc)


# set train/test set randomization seed
seed <- 123456
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# maize control yields train/test split
index<- createDataPartition(mdat$Y0, p = 0.75, list = FALSE, times = 1)
mTrain <- mdat [ index,] # trainingData:'data.frame':  194 obs. of  31 variables
mTest  <- mdat [-index,]
mTrain =na.omit(mTrain)
#Train and test sets with coordinates
mTrainc=mdatc[ index,]
mTrainlocs=mTrainc[,2:3]
mTestc=mdatc[-index,]
mTestlocs= mTestc[,2:3]

correlationMatrix <- cor(mdat[,2:30])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# GAM maize control yields # not part of CARET
#on 302 training sites not cross validation
maize.gam= gam(log(mTrain$Y0) ~ ., data = mTrain,
               family = gaussian)
plot(maize.gam, residuals= TRUE)
preplot(maize.gam)
mgam.pred=predict(maize.gam, type="terms")
Ycgam.pred=predict(grids, maize.gam, type="response")

#plot estimates against 100 independent test yields
Ygamtestpred=extract(Ycgam.pred, mTestlocs)
plot(mTest$Y0, exp(Ygamtestpred))
xgam=lm(exp(Ygamtestpred)~mTest$Y0+0)
abline(xgam, col="red")

#+ Stepwise main effects GLM's <MASS> --------------------------------------
# 5-fold CV  cross validation
step <- trainControl(method = "cv", number = 5)
# GLM maize control yields on training samples
maize.glm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                   family = gaussian, 
                   method = "glmStepAIC",
                   trControl = step)
print(maize.glm)
ycglm.pred <- predict(grids, maize.glm) ## spatial predictions

Yglmtestpred=extract(ycglm.pred, mTestlocs)
plot(mTest$Y0, exp(Yglmtestpred))
xglm=lm(exp(Yglmtestpred)~mTest$Y0+0)
abline(xglm, col="red")

# Regression trees for training set maize
#reverse partioning rpart

#reverse partitioning regression with rpart on Yc
rt= trainControl(method= "cv", number = 10, repeats= 5)

# Control yield predictions (Yc) using reverese partitioning regression
Yc.rt <- train(log(mTrain$Y0) ~ ., method="rpart", data=mTrain)
ycrt.pred <- predict(grids, Yc.rt)

#plot importances
imp=varImp(Yc.rt, surrogates=FALSE, competes= TRUE)
plot(imp)
#print importances
print (imp)

Ycrttestpred=extract(ycrt.pred, mTestlocs)
plot(mTest$Y0, exp(Ycrttestpred))
xrt=lm(exp(Ycrttestpred)~mTest$Y0+0)
abline(xrt, col="red")

# Random forests (no tuning default) for training set maize
# out-of-bag predictions
oob <- trainControl(method = "oob")
# Control yield predictions (Yc) using random forests (out of bag)
Yc.rf <- train(log(mTrain$Y0) ~ ., method= "rf" ,
               data=mTrain, trControl = oob, importance=TRUE)
ycrf.pred <- predict(grids, Yc.rf)

#importance of variables
imprf=varImp(Yc.rf)
plot(imprf, main = "Variables contribution to the RF regression")
#test predictions with Test set
Ycrftestpred=extract(ycrf.pred, mTestlocs)
plot(mTest$Y0, exp(Ycrftestpred))
xrf=lm(exp(Ycrftestpred)~mTest$Y0+0)
abline(xrf, col="red")

#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's cross validation
gbm <- trainControl(method = "repeatedcv", number = 10, repeats= 5)

# gbm gradient boosting for training set control yields
Yc.gbm <- train(log(mTrain$Y0) ~ ., data = mTrain,
                method = "gbm",
                trControl = gbm)
ycgbm.pred <- predict(grids,Yc.gbm) ## predict train-set

#importances gbm
impgbm=varImp (Yc.gbm)
plot(impgbm, main = "Variables contribution to the GBM")

Ycgbmtestpred=extract(ycgbm.pred, mTestlocs)
plot(mTest$Y0, exp(Ycgbmtestpred))
xgbm=lm(exp(Ycgbmtestpred)~mTest$Y0+0)
abline(xgbm, col="red")

#+ Neural nets <nnet> this will change in future using h2o------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# neural network of control yields Maize
Y0.nn <- train(log(mTrain$Y0) ~ ., data = mTrain,
               method = "nnet",
               trControl = nn)
ycnn.pred <- predict(grids,Y0.nn)

Ycnntestpred=extract(ycnn.pred, mTestlocs)
plot(mTest$Y0, exp(Ycnntestpred))
xnn=lm(exp(Ycnntestpred)~mTest$Y0+0)
abline(xnn, col="red")

#regression ensembles

#plot ensembles
#largely based on Markus Walsh------------------------------------------
#bring first 6 regression predictions together in a stack
ycpred <- stack(Ycgam.pred, ycglm.pred, ycrt.pred, ycrf.pred, ycgbm.pred, ycnn.pred)
names(ycpred) <- c("ycgam", "ycglm", "ycrt", "ycrf", "ycgbm", "ycnn")
plot(ycpred, axes = F)

# Test set ensemble predictions for Yc

exyc <- extract(ycpred, OFRA_LREPxy) #extract predictions from stack regressions at all points
Ycens <- cbind.data.frame(Y0, exyc) #adds yields to covariates
Ycens <- na.omit(Ycens) # deletes NAs
YcensTest <- Ycens[-index,] # splits dataset to test set using same seed 12345

# do Ensemble control yield predictions (Yc) on test set

# Regularized ensemble weighting on the test set <test>
# 5 fold CV again cross validation to compensate for collinearity
ens <- trainControl(method = "cv", number = 5)
Yc.ens <- train(log(Y0)~ycgam+ycglm+ycrt+ycrf+ycgbm+ycnn,family=gaussian, 
                data = YcensTest, 
                method = "ridge",
                trControl = ens)
Ycimp= varImp(Yc.ens)
plot(Ycimp, main= "Relative weights of regressions")
Yc.pred=predict(ycpred,Yc.ens)
plot(exp(Yc.pred))
points(OFRA_LREPxy, cex=0.1, col= "red")
points(mTrainlocs,cex=0.1, col="blue")
# test predictions with observations on the Test set
Ycwgt_extr=extract(Yc.pred, mTestlocs)
plot(mTest$Y0,exp(Ycwgt_extr))
ens=lm(exp(Ycwgt_extr)~mTest$Y0+0)
abline(ens, col="red")

#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("AF_results", showWarnings=F)

x=exp(Yc.pred)
# Export Gtif's to "./AF_results"
writeRaster(x, filename="./AF_results/AF_maizepreds4.tif", format="GTiff", overwrite=T)


