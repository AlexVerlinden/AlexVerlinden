# UG Cropland predictions, including woodland and human settlements
# First woodland and human settlements are predicted and added as covariates for cropland
#ensemble predictions models based on Walsh 2016
# Alex Verlinden 2017
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret", "doParallel", "e1071")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(doParallel)
#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("UG_crops", showWarnings=F)
dat_dir <- "./UG_crops"

#Download all geosurvey data
download.file("https://www.dropbox.com/s/e4fd5wtvjdn0ncn/UG_geos_all.csv?dl=0", "./UG_crops/UG_geos_all.csv", mode="wb")
geos <- read.csv(paste(dat_dir, "/UG_geos_all.csv", sep=""), header=T, sep=",")

#download grids for UG  ~ 87 MB
download.file("https://www.dropbox.com/s/55cg9e95mdwd0ut/UG_grids_250m.zip?dl=0","./UG_crops/UG_GRIDS250m.zip",  mode="wb")
unzip("./UG_crops/UG_grids250m.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)

t=scale(grid, center=TRUE,scale=TRUE) # scale all covariates

#+ Data setup --------------------------------------------------------------
#drop UG protected
t=dropLayer(t, 23) # protected areas turn to too many NAs
names(t)= c("Al", "B", "BIO12", "BIOMFI", "BLD", "Ca", "CEC", "PRECavg", "PREC_2015", "EVI",
            "LSTD", "LSTN", "B1","B2", "B3", "B7","Mg", "Na", "NDVI", "Ntot", "ORC", "PH",
            "RDS", "SND", "FPARavg", "FPARstd", "FPARvar", "ELEV", "TWI")
#set projection of geosurvey
geo.proj=as.data.frame(project(cbind(geos$Longitude, geos$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geo.proj) <- c("x","y")
coordinates(geo.proj) <- ~x+y
projection(geo.proj) <- projection(grid)

#names geosurvey
geos=geos[,7:10]
colnames(geos)= c("Banana", "HSP", "CRP", "WDL")

# Extract gridded variables at GeoSurvey locations
geogrid=extract(t,geo.proj)#extract 19 k
# Assemble dataframes

#presence absence of Woodland >60%
WDL1=geos$WDL
prop.table(table(WDL1))
wdl1dat=cbind.data.frame(WDL1, geogrid)
wdl1dat=na.omit(wdl1dat)
colnames(wdl1dat)[1]= "WDL1"
#First prediction of woodland
#+ Split data into train and test sets ------------------------------------
seed=12345
set.seed=seed

# woodland train/test split
wdl1Index <- createDataPartition(wdl1dat$WDL1, p = 2/3, list = FALSE, times = 1)
wdl1Train <- wdl1dat[ wdl1Index,]
wdl1Test  <- wdl1dat[-wdl1Index,]


mc <- makeCluster(detectCores())
registerDoParallel(mc)


#RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
WDL1.rf <- train(WDL1 ~ ., data = wdl1Train,
                 method = "rf",
                 ntree=501,
                 metric= "ROC",
                 trControl = objControl)
confusionMatrix(WDL1.rf)
WDL1rf.pred <- predict(t, WDL1.rf, type = "prob") ## spatial predictions
plot(varImp(WDL1.rf,scale=F), main = "covariate importance Random Forest Woodland")

#gbm
WDL1.gbm <- train(WDL1 ~ ., data = wdl1Train,
                  method = "gbm",
                  metric = "ROC",
                  trControl = objControl)
confusionMatrix(WDL1.gbm)
WDL1gbm.pred <- predict(t, WDL1.gbm, type = "prob") ## spatial predictions
#neural net
WDL1.nn <- train(WDL1 ~., data=wdl1Train, 
                 method = "nnet", 
                 metric= "ROC",
                 #preProc = c("center", "scale"), 
                 trControl = objControl)
confusionMatrix((WDL1.nn))
WDL1.imp <- varImp(WDL1.nn, useModel = FALSE)
plot(WDL1.imp, top=20)
WDL1nn.pred <- predict(t, WDL1.nn, type="prob")

#ensemble regression glmnet (elastic net)
pred <- stack(1-WDL1rf.pred, 
              1-WDL1gbm.pred, 1-WDL1nn.pred)
names(pred) <- c("WDL1rf","WDL1gbm", "WDL1nn")
geospred <- extract(pred, geo.proj)

# presence/absence of Woodland (present = Y, absent = N)
WDL1ens <- cbind.data.frame(geos$WDL, geospred)
WDL1ens <- na.omit(WDL1ens)
WDL1ensTest <- WDL1ens[-wdl1Index,] ## replicate previous test set
names(WDL1ensTest)[1]= "WDL1"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of woodland (present = Y, absent = N)
WDL1.ens <- train(WDL1 ~. , data = WDL1ensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)
confusionMatrix(WDL1.ens) # print validation summaries on crossvalidation
WDL1ens.pred <- predict(WDL1.ens, WDL1ensTest,  type="prob") ## predict test-set
WDL1.test <- cbind(WDL1ensTest, WDL1ens.pred)
WDL1p <- subset(WDL1.test, WDL1=="Y", select=c(Y))
WDL1a <- subset(WDL1.test, WDL1=="N", select=c(Y))
WDL1.eval <- evaluate(p=WDL1p[,1], a=WDL1a[,1]) ## calculate ROC's on test set <dismo>
WDL1.eval
plot(WDL1.eval, 'ROC') ## plot ROC curve

#now on all data without statistics
#RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
WDL.rf <- train(WDL1 ~ ., data = wdl1dat,
                method = "rf",
                ntree=501,
                metric= "ROC",
                trControl = objControl)
confusionMatrix(WDL.rf)
WDLrf.pred <- predict(t, WDL.rf, type = "prob") ## spatial predictions
plot(varImp(WDL.rf,scale=F))

#gbm
WDL.gbm <- train(WDL1 ~ ., data = wdl1dat,
                 method = "gbm",
                 metric = "ROC",
                 trControl = objControl)
confusionMatrix(WDL.gbm)
WDLgbm.pred <- predict(t, WDL.gbm, type = "prob") ## spatial predictions
#neural net
WDL.nn <- train(WDL1 ~., data=wdl1dat, 
                method = "nnet", 
                metric= "ROC",
                #preProc = c("center", "scale"), 
                trControl = objControl)
confusionMatrix((WDL.nn))
WDLnn.pred <- predict(t, WDL.nn, type="prob")

#ensemble regression glmnet (elastic net)
pred <- stack(1-WDLrf.pred, 
              1-WDLgbm.pred, 1-WDLnn.pred)
names(pred) <- c("WDLrf","WDLgbm", "WDLnn")
geospred <- extract(pred, geo.proj)

# presence/absence of Woodland (present = Y, absent = N)
WDLens <- cbind.data.frame(geos$WDL, geospred)
WDLens <- na.omit(WDLens)
WDLensTest <- WDLens[-wdl1Index,] ## replicate previous test set
names(WDLensTest)[1]= "WDL1"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
#ens <- trainControl(method = "cv", number = 10)
ens<- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)

# presence/absence of Woodland (present = Y, absent = N)
WDL.ens <- train(WDL1 ~. , data = WDLensTest,
                 family = "binomial", 
                 method = "rf",
                 trControl = ens)
confusionMatrix(WDL.ens)
#spatial predictions
WDLens.pred <- predict(pred, WDL.ens, type="prob") 
dir.create("UG_results", showWarnings=F)
writeRaster(1-WDLens.pred, filename = "./UG_results/UG_woodlandpred.tif", overwrite=TRUE )

WDLens.pred <- predict(WDL.ens, WDLensTest,  type="prob") ## predict test-set
WDL.test <- cbind(WDLensTest, WDLens.pred)
WDLp <- subset(WDL.test, WDL1=="Y", select=c(Y))
WDLa <- subset(WDL.test, WDL1=="N", select=c(Y))
WDL.eval <- evaluate(p=WDLp[,1], a=WDLa[,1]) ## calculate ROC's on test set <dismo>
WDL.eval
WDL.thld <- threshold(WDL.eval, 'spec_sens')
WDLens.pred <- predict(pred, WDL.ens, type="prob") 
WDL.mask= (1-WDLens.pred)>WDL.thld
writeRaster(WDL.mask, filename = "./UG_results/UG_woodlandmask.tif", overwrite=TRUE )

WDLens.pred <- predict(pred, WDL.ens, type="prob") 

WDLENS=1-WDLens.pred
WDLENS=scale(WDLENS, center=TRUE,scale=TRUE)
t=addLayer(t,WDLENS)
names(t)[30]= "WDLENS"
#human settlements
#+ Data setup --------------------------------------------------------------
# Extract gridded variables at GeoSurvey locations
geogrid=extract(t,geo.proj)#extract 19 k
# Assemble dataframes

#presence absence of settlements
HSP1=geos$HSP
prop.table(table(HSP1))
hsp1dat=cbind.data.frame(HSP1, geogrid)
hsp1dat=na.omit(hsp1dat)
colnames(hsp1dat)[1]= "HSP1"
#First prediction of woodland
#+ Split data into train and test sets ------------------------------------
seed=12345
set.seed=seed

# woodland train/test split
hsp1Index <- createDataPartition(hsp1dat$HSP1, p = 2/3, list = FALSE, times = 1)
hsp1Train <- hsp1dat[ hsp1Index,]
hsp1Test  <- hsp1dat[-hsp1Index,]


mc <- makeCluster(detectCores())
registerDoParallel(mc)


#RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
HSP1.rf <- train(HSP1 ~ ., data = hsp1Train,
                 method = "rf",
                 ntree=501,
                 metric= "ROC",
                 trControl = objControl)
confusionMatrix(HSP1.rf)
HSP1rf.pred <- predict(t, HSP1.rf, type = "prob") ## spatial predictions
plot(varImp(HSP1.rf,scale=F), main = "Variable Importance Random Forest Settlements")

#gbm
HSP1.gbm <- train(HSP1 ~ ., data = hsp1Train,
                  method = "gbm",
                  metric = "ROC",
                  trControl = objControl)
confusionMatrix(HSP1.gbm)
HSP1gbm.pred <- predict(t, HSP1.gbm, type = "prob") ## spatial predictions
#neural net
HSP1.nn <- train(HSP1 ~., data=hsp1Train, 
                 method = "nnet", 
                 metric= "ROC",
                 #preProc = c("center", "scale"), 
                 trControl = objControl)
confusionMatrix((HSP1.nn))
HSP1.imp <- varImp(HSP1.nn, useModel = FALSE)
plot(HSP1.imp, top=20)
HSP1nn.pred <- predict(t, HSP1.nn, type="prob")

#ensemble regression glmnet (elastic net)
pred <- stack(1-HSP1rf.pred, 
              1-HSP1gbm.pred, 1-HSP1nn.pred)
names(pred) <- c("HSP1rf","HSP1gbm", "HSP1nn")
geospred <- extract(pred, geo.proj)

# presence/absence of settlements (present = Y, absent = N)
HSP1ens <- cbind.data.frame(geos$HSP, geospred)
HSP1ens <- na.omit(HSP1ens)
HSP1ensTest <- HSP1ens[-hsp1Index,] ## replicate previous test set
names(HSP1ensTest)[1]= "HSP1"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of hsp (present = Y, absent = N)
HSP1.ens <- train(HSP1 ~. , data = HSP1ensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)
confusionMatrix(HSP1.ens) # print validation summaries on crossvalidation
HSP1ens.pred <- predict(HSP1.ens, HSP1ensTest,  type="prob") ## predict test-set
HSP1.test <- cbind(HSP1ensTest, HSP1ens.pred)
HSP1p <- subset(HSP1.test, HSP1=="Y", select=c(Y))
HSP1a <- subset(HSP1.test, HSP1=="N", select=c(Y))
HSP1.eval <- evaluate(p=HSP1p[,1], a=HSP1a[,1]) ## calculate ROC's on test set <dismo>
HSP1.eval
plot(HSP1.eval, 'ROC')

#for all data RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
HSP.rf <- train(HSP1 ~ ., data = hsp1dat,
                method = "rf",
                ntree=501,
                metric= "ROC",
                trControl = objControl)
confusionMatrix(HSP.rf)
HSPrf.pred <- predict(t, HSP.rf, type = "prob") ## spatial predictions


#gbm
HSP.gbm <- train(HSP1 ~ ., data = hsp1dat,
                 method = "gbm",
                 metric = "ROC",
                 trControl = objControl)
confusionMatrix(HSP.gbm)
HSPgbm.pred <- predict(t, HSP.gbm, type = "prob") ## spatial predictions
#neural net
HSP.nn <- train(HSP1 ~., data=hsp1dat, 
                method = "nnet", 
                metric= "ROC",
                #preProc = c("center", "scale"), 
                trControl = objControl)
confusionMatrix((HSP.nn))

HSPnn.pred <- predict(t, HSP.nn, type="prob")

#ensemble regression glmnet (elastic net)
pred <- stack(1-HSPrf.pred, 
              1-HSPgbm.pred, 1-HSPnn.pred)
names(pred) <- c("HSPrf","HSPgbm", "HSPnn")
geospred <- extract(pred, geo.proj)

# presence/absence of settlements (present = Y, absent = N)
HSPens <- cbind.data.frame(geos$HSP, geospred)
HSPens <- na.omit(HSPens)
HSPensTest <- HSPens[-hsp1Index,] ## replicate previous test set
names(HSPensTest)[1]= "HSP"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of hsp (present = Y, absent = N)
HSP.ens <- train(HSP ~. , data = HSPensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
confusionMatrix(HSP.ens)
#spatial predictions
HSPens.pred <- predict(pred, HSP.ens, type="prob") 
dir.create("UG_results", showWarnings=F)
writeRaster(1-HSPens.pred, filename = "./UG_results/UG_hsppred.tif", overwrite=TRUE )

UG_hsppred=1-HSPens.pred
UG_hsppred=scale(UG_hsppred, center=TRUE,scale=TRUE)
t=addLayer(t,UG_hsppred)
names(t)[31]= "HSPENS" #check number

# Cropland prediction
# Assemble dataframes

#presence absence of cropland
CRP1=geos$CRP
prop.table(table(CRP1))
geogrid=extract(t,geo.proj)#extract 19 k
crp1dat=cbind.data.frame(CRP1, geogrid)
crp1dat=na.omit(crp1dat)
colnames(crp1dat)[1]= "CRP1"
#First prediction of woodland
#+ Split data into train and test sets ------------------------------------
seed=12345
set.seed=seed

# cropland train/test split
crp1Index <- createDataPartition(crp1dat$CRP1, p = 2/3, list = FALSE, times = 1)
crp1Train <- crp1dat[ crp1Index,]
crp1Test  <- crp1dat[-crp1Index,]


mc <- makeCluster(detectCores())
registerDoParallel(mc)


#RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
CRP1.rf <- train(CRP1 ~ ., data = crp1Train,
                 method = "rf",
                 ntree=501,
                 metric= "ROC",
                 trControl = objControl)
confusionMatrix(CRP1.rf)
plot(varImp(CRP1.rf,scale=F), main = "Variable importance Random Forest Cropland")
CRP1rf.pred <- predict(t, CRP1.rf, type = "prob") ## spatial predictions



#gbm
CRP1.gbm <- train(CRP1 ~ ., data = crp1Train,
                  method = "gbm",
                  metric = "ROC",
                  trControl = objControl)
confusionMatrix(CRP1.gbm)
CRP1gbm.pred <- predict(t, CRP1.gbm, type = "prob") ## spatial predictions
#neural net
CRP1.nn <- train(CRP1 ~., data=crp1Train, 
                 method = "nnet", 
                 metric= "ROC",
                 #preProc = c("center", "scale"), 
                 trControl = objControl)
confusionMatrix(CRP1.nn)

CRP1nn.pred <- predict(t,CRP1.nn, type="prob")

#for cropland
#ensemble regression glmnet (elastic net)
pred <- stack(1-CRP1rf.pred, 
              1-CRP1gbm.pred, 1-CRP1nn.pred)
names(pred) <- c("CRP1rf","CRP1gbm", "CRP1nn")
geospred <- extract(pred, geo.proj)

# presence/absence of Cropland (present = Y, absent = N)
CRP1ens <- cbind.data.frame(geos$CRP, geospred)
CRP1ens <- na.omit(CRP1ens)
CRP1ensTest <- CRP1ens[-crp1Index,] ## replicate previous test set
names(CRP1ensTest)[1]= "CRP1"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of hsp (present = Y, absent = N)
CRP1.ens <- train(CRP1 ~. , data = CRP1ensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)
confusionMatrix(CRP1.ens) # print validation summaries on crossvalidation
CRP1ens.pred <- predict(CRP1.ens, CRP1ensTest,  type="prob") ## predict test-set
CRP1.test <- cbind(CRP1ensTest, CRP1ens.pred)
CRP1p <- subset(CRP1.test, CRP1=="Y", select=c(Y))
CRP1a <- subset(CRP1.test, CRP1=="N", select=c(Y))
CRP1.eval <- evaluate(p=CRP1p[,1], a=CRP1a[,1]) ## calculate ROC's on test set <dismo>
CRP1.eval
plot(CRP1.eval, 'ROC')

#for all data RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
CRP.rf <- train(CRP1 ~ ., data = crp1dat,
                method = "rf",
                ntree=501,
                metric= "ROC",
                trControl = objControl)
confusionMatrix(CRP.rf)
CRPrf.pred <- predict(t, CRP.rf, type = "prob") ## spatial predictions


#gbm
CRP.gbm <- train(CRP1 ~ ., data = crp1dat,
                 method = "gbm",
                 metric = "ROC",
                 trControl = objControl)
confusionMatrix(CRP.gbm)
CRPgbm.pred <- predict(t, CRP.gbm, type = "prob") ## spatial predictions
#neural net
CRP.nn <- train(CRP1 ~., data=crp1dat, 
                method = "nnet", 
                metric= "ROC",
                #preProc = c("center", "scale"), 
                trControl = objControl)
confusionMatrix(CRP.nn)

CRPnn.pred <- predict(t, CRP.nn, type="prob")

#ensemble regression glmnet (elastic net)
pred <- stack(1-CRPrf.pred, 
              1-CRPgbm.pred, 1-CRPnn.pred)
names(pred) <- c("CRPrf","CRPgbm", "CRPnn")
geospred <- extract(pred, geo.proj)

# presence/absence of Woodland (present = Y, absent = N)
CRPens <- cbind.data.frame(geos$CRP, geospred)
CRPens <- na.omit(CRPens)
CRPensTest <- CRPens[-crp1Index,] ## replicate previous test set
names(CRPensTest)[1]= "CRP"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of hsp (present = Y, absent = N)
CRP.ens <- train(CRP ~. , data = CRPensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
#define threshold
CRPens.pred <- predict(CRP.ens, CRPensTest,  type="prob") ## predict test-set
CRP.test <- cbind(CRPensTest, CRPens.pred)
CRPp <- subset(CRP.test, CRP=="Y", select=c(Y))
CRPa <- subset(CRP.test, CRP=="N", select=c(Y))
CRP.eval <- evaluate(p=CRPp[,1], a=CRPa[,1]) ## calculate ROC's on test set <dismo>
CRP.eval
CRP.thld <- threshold(CRP.eval, 'spec_sens') 
#spatial predictions
CRPens.pred <- predict(pred, CRP.ens, type="prob") 
dir.create("UG_results", showWarnings=F)
writeRaster(1-CRPens.pred, filename = "./UG_results/UG_crppred.tif", overwrite= TRUE )
cropmask=(1-CRPens.pred)>CRP.thld
writeRaster(cropmask, filename="./UG_results/UG_cropmask_250m.tif", overwrite=TRUE )
UG_crop=1-CRPens.pred

UG_crop=scale(UG_crop, center=TRUE,scale=TRUE)
t=addLayer(t,UG_crop)
names(t)[32]= "CROPens" #check number

#presence absence of banana
BAN1=geos$Banana
prop.table(table(BAN1))
geogrid=extract(t,geo.proj)#extract 19 k
ban1dat=cbind.data.frame(BAN1, geogrid)
ban1dat=na.omit(ban1dat)
colnames(ban1dat)[1]= "BAN1"
#First prediction of banana
#+ Split data into train and test sets ------------------------------------
seed=12345
set.seed=seed

# banana train/test split
ban1Index <- createDataPartition(ban1dat$BAN1, p = 2/3, list = FALSE, times = 1)
ban1Train <- ban1dat[ ban1Index,]
ban1Test  <- ban1dat[-ban1Index,]

mc <- makeCluster(detectCores())
registerDoParallel(mc)

#RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
BAN1.rf <- train(BAN1 ~ ., data = ban1Train,
                 method = "rf",
                 ntree=501,
                 metric= "ROC",
                 trControl = objControl)
confusionMatrix(BAN1.rf)
plot(varImp(BAN1.rf,scale=F), main = "covariate importance Random Forest for Banana")
BAN1rf.pred <- predict(t, BAN1.rf, type = "prob") ## spatial predictions

#gbm
BAN1.gbm <- train(BAN1 ~ ., data = ban1Train,
                  method = "gbm",
                  metric = "ROC",
                  trControl = objControl)
confusionMatrix(BAN1.gbm)
BAN1gbm.pred <- predict(t, BAN1.gbm, type = "prob") ## spatial predictions
#neural net
BAN1.nn <- train(BAN1 ~., data=ban1Train, 
                 method = "nnet", 
                 metric= "ROC",
                 #preProc = c("center", "scale"), 
                 trControl = objControl)
confusionMatrix(BAN1.nn)

BAN1nn.pred <- predict(t,BAN1.nn, type="prob")

#for cropland
#ensemble regression glmnet (elastic net)
pred <- stack(1-BAN1rf.pred, 
              1-BAN1gbm.pred, 1-BAN1nn.pred)
names(pred) <- c("BAN1rf","BAN1gbm", "BAN1nn")
geospred <- extract(pred, geo.proj)

# presence/absence of Cropland (present = Y, absent = N)
BAN1ens <- cbind.data.frame(geos$Banana, geospred)
BAN1ens <- na.omit(BAN1ens)
BAN1ensTest <- BAN1ens[-ban1Index,] ## replicate previous test set
names(BAN1ensTest)[1]= "BAN1"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of hsp (present = Y, absent = N)
BAN1.ens <- train(BAN1 ~. , data = BAN1ensTest,
                  family = "binomial", 
                  method = "glmnet",
                  trControl = ens)
confusionMatrix(BAN1.ens) # print validation summaries on crossvalidation
BAN1ens.pred <- predict(BAN1.ens, BAN1ensTest,  type="prob") ## predict test-set
BAN1.test <- cbind(BAN1ensTest, BAN1ens.pred)
BAN1p <- subset(BAN1.test, BAN1=="Y", select=c(Y))
BAN1a <- subset(BAN1.test, BAN1=="N", select=c(Y))
BAN1.eval <- evaluate(p=BAN1p[,1], a=BAN1a[,1]) ## calculate ROC's on test set <dismo>
BAN1.eval
plot(BAN1.eval, 'ROC')

#for all data RF--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
BAN.rf <- train(BAN1 ~ ., data = ban1dat,
                method = "rf",
                ntree=501,
                metric= "ROC",
                trControl = objControl)
confusionMatrix(BAN.rf)
BANrf.pred <- predict(t, BAN.rf, type = "prob") ## spatial predictions


#gbm
BAN.gbm <- train(BAN1 ~ ., data = ban1dat,
                 method = "gbm",
                 metric = "ROC",
                 trControl = objControl)
confusionMatrix(BAN.gbm)
BANgbm.pred <- predict(t, BAN.gbm, type = "prob") ## spatial predictions
#neural net
BAN.nn <- train(BAN1 ~., data=ban1dat, 
                method = "nnet", 
                metric= "ROC",
                #preProc = c("center", "scale"), 
                trControl = objControl)
confusionMatrix(BAN.nn)

BANnn.pred <- predict(t, BAN.nn, type="prob")

#ensemble regression glmnet (elastic net)
pred <- stack(1-BANrf.pred, 
              1-BANgbm.pred, 1-BANnn.pred)
names(pred) <- c("BANrf","BANgbm", "BANnn")
geospred <- extract(pred, geo.proj)

# presence/absence of Banana (present = Y, absent = N)
BANens <- cbind.data.frame(geos$Banana, geospred)
BANens <- na.omit(BANens)
BANensTest <- BANens[-ban1Index,] ## replicate previous test set
names(BANensTest)[1]= "BAN"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method = "cv", number = 10)

# presence/absence of hsp (present = Y, absent = N)
BAN.ens <- train(BAN ~. , data = BANensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
#define threshold
BANens.pred <- predict(BAN.ens, BANensTest,  type="prob") ## predict test-set
BAN.test <- cbind(BANensTest, BANens.pred)
BANp <- subset(BAN.test, BAN=="Y", select=c(Y))
BANa <- subset(BAN.test, BAN=="N", select=c(Y))
BAN.eval <- evaluate(p=BANp[,1], a=BANa[,1]) ## calculate ROC's on test set <dismo>
BAN.eval
BAN.thld <- threshold(BAN.eval, 'spec_sens') 
#spatial predictions
BANens.pred <- predict(pred, BAN.ens, type="prob") 
dir.create("UG_results", showWarnings=F)
writeRaster(1-BANens.pred, filename = "./UG_results/UG_BANpred.tif", overwrite= TRUE )
BANmask=(1-BANens.pred)>BAN.thld
writeRaster(BANmask2, filename="./UG_results/UG_BANmask2_250m.tif", overwrite=TRUE )
UG_BAN=1-BANens.pred
plot(UG_BAN, main="Banana ensemble Predictions")