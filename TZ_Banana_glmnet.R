# Script for banana crop distribution TZ
# basis for cropland mask is the 1 Million point survey for Africa of 2014-2015 conducted by AfSIS for Africa
# script in development to test mahalanobis model on presence/absence from crop scout
# Alex Verlinden November 2015
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "e1071")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("TZ_banana", showWarnings=F)
dat_dir <- "./TZ_banana"
# download banana presence/absence locations
# these are data from 2015 crop scout ODK forms
#download.file("https://www.dropbox.com/s/h58lay7iid52odp/Final_NOT.csv?dl=0", "./NOT_samples/Final_NOT.csv", mode="wb")
ban <- read.csv(paste(dat_dir, "/Crop_banana.csv", sep= ""), header=T, sep=",")

#download grids for TZ  ? MB
#download.file("https://www.dropbox.com/s/51k0ywnsnzmb78m/STZ_250m7.zip?dl=0","./NOT_samples/STZ_250m7.zip")
#unzip("./STZ_250m7.zip", exdir=dat_dir, overwrite=T)
#glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
glist=list.files("/Users/alexverlinden/Documents/R-testing/crops/TZ_data", pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup --------------------------------------------------------------
# Project test data to grid CRS
ban.proj <- as.data.frame(project(cbind(ban$X_gps_longitude, ban$X_gps_latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(ban.proj) <- c("x","y")
coordinates(ban.proj) <- ~x+y  #convert to Spatial DataFrame
projection(ban.proj) <- projection(grid)

# Extract gridded variables for TZ to test data observations 
banex <- data.frame(coordinates(ban.proj), extract(grid, ban.proj))
banex=  banex[,3:28]
banpresabs=cbind(ban$Banana, banex)
banpresabs=na.omit(banpresabs)
colnames(banpresabs)[1]="ban"
banpresabs$ban=as.factor(banpresabs$ban)
summary(banpresabs)

crp=raster("/Users/alexverlinden/Documents/R-testing/crops/TZ_banana/TZ_cropmask.tif")
crp[crp==0]=NA

###### Regressions 
# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# Cropland train/test split
banIndex <- createDataPartition(banpresabs$ban, p = 2/3, list = FALSE, times = 1)
banTrain <- banpresabs[ banIndex,]
banTest  <- banpresabs[-banIndex,]
banTest= na.omit(banTest)
#to test if banana is a rare event (in this case it is just below 15% so it is)
prop.table(table(banpresabs$ban))
#print structure
print(str(banpresabs))

objControl <- trainControl(method='cv', number=3, returnResamp='none')
#glmnet using binomial distribution
ban.glm=train(ban ~ ., data=banTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(ban.glm, banTest[,2:27], type="prob")
#AUC  
#install.packages(pROC)
auc <- roc(banTest$ban, predictions)
print(auc$auc) 
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
plot(banmask, legend=F)

dir.create("TZ_results", showWarnings=F)
rf=writeRaster(banglmnet.pred, filename="./TZ_results/TZ_banana_2015_glm", format= "GTiff", overwrite=TRUE)
rf=writeRaster(banmask2, filename="./TZ_results/TZ_banana_2015_mask2", format= "GTiff", overwrite=TRUE)

#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# presence/absence of Cropland (CRP, present = Y, absent = N)
CRP.rf <- train(CRP ~ ., data = crpTrain,
                method = "rf",
                trControl = oob)
crprf.test <- predict(CRP.rf, crpTest) ## predict test-set
confusionMatrix(crprf.test, crpTest$CRP, "Y") ## print validation summaries
crprf.pred <- predict(grid, CRP.rf, type = "prob") ## spatial predictions