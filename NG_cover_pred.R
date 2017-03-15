#' Ensemble predictions of Nigeria GeoSurvey cropland, woody vegetation cover,
#' First woodland predictions and then settlement to refine cropland
#' Alex Verlinden 2016 after M. Walsh, April 2014
# observations collected by crowdsourcing using "Geosurvey"  in 2017 and 5000 in 2016 
# Required packages
# install.packages(c("devtools", "doParallel", "downloader","raster","rgdal","caret","randomForest","gbm","nnet","glmnet","dismo")), dependencies=TRUE)
require(devtools)
require(doParallel)
require(downloader)
require(raster)
require(rgdal)
require(caret)
require(randomForest)
require(gbm)
require(nnet)
require(glmnet)
require(dismo)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("NG_data", showWarnings=F)
dat_dir <- "NG_data"

#download Geosurvey data  13000 +
download.file("https://www.dropbox.com/s/jrbw4yv05j6vab0/NG_cover2017all.zip?raw=1", "./NG_data/NG_cover2017all.zip", mode="wb")
unzip("./NG_data/NG_cover2017all.zip", exdir= "./NG_data", overwrite=T)
geos1 <- read.table(paste(dat_dir, "/NG_cover2017all.csv", sep=""), header=T, sep=",")
geos1 <- na.omit(geos1)
# Project GeoSurvey coords to grid CRS for 15000+
geos1.proj <- as.data.frame(project(cbind(geos1$Longitude, geos1$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(geos1.proj) <- c("x","y")
coordinates(geos1.proj) <- ~x+y
projection(geos1.proj) <- projection(grid)

# download NG Gtifs (!!!~1.26 GB) and stack in raster
#download.file("https://www.dropbox.com/s/w6w6axgtgob5x9e/NG_GRIDS.zip?raw=1", "./NG_data/NG_GRIDS.zip", mode="wb")
unzip("./NG_data/NG_GRIDS.zip", exdir="./NG_data", overwrite=T)
glist <- list.files(path="/Volumes/Transcend/Nigeria/GRIDS", pattern="tif", full.names=T)
grid <- stack(glist)
#grid= dropLayer(grid, 8) # check for unwanted grids e.g previous cropland preds
t=scale(grid,center=TRUE, scale=TRUE)
names(t)=c("Al", "BIO12ALT", "BIOMFI", "BLD",
           "CEC", "PREC_2015", "PREC_AVG", "B" , "Ca", "K", "Mg", "Na", 
          "fPARavg", "fPARsd", "fPARvar", "EVI", "NDVI", "B1", "B2", "B3", "B7",
           "BNALT", "BSALT", "BVALT", "LSTD", "LSTN", "NTOT", "ORC", "PH", "RDSDIST",
          "SENT1VH", "SENT1VV", "SND", "ELEV", "TWI", "WDPA")
#+ Data setup --------------------------------------------------------------
# Extract gridded variables at GeoSurvey locations
geogrid=extract(t, geos1.proj) #extract 15k

# Assemble dataframes

#presence absence of Woodland >60%
WDL1=geos1$WDL
prop.table(table(WDL1))
wdl1dat=cbind.data.frame(WDL1, geogrid)
wdl1dat=na.omit(wdl1dat)

#First prediction of woodland
#+ Split data into train and test sets ------------------------------------
seed=12345
set.seed=seed

# woodland train/test split
wdl1Index <- createDataPartition(wdl1dat$WDL1, p = 2/3, list = FALSE, times = 1)
wdl1Train <- wdl1dat[ wdl1Index,]
wdl1Test  <- wdl1dat[-wdl1Index,]


objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)
WDL1.rf <- train(WDL1 ~ ., data = wdl1Train,
                 method = "rf",
                 ntree=501,
                 metric= "ROC",
                 trControl = objControl)
confusionMatrix(WDL1.rf)
plot(varImp(WDL1.rf,scale=F), main= "Variable Importance Random Forest Woodland")
#spatial prediction for woodland
WDL1rf.pred=predict(t, WDL1.rf, type = "prob")

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
#WDL1.imp <- varImp(WDL1.nn, useModel = FALSE)
#plot(WDL1.imp, top=20)
WDL1nn.pred <- predict(t, WDL1.nn, type="prob")

#ensemble regression glmnet (elastic net)
pred <- stack(1-WDL1rf.pred, 
              1-WDL1gbm.pred, 1-WDL1nn.pred)
names(pred) <- c("WDL1rf","WDL1gbm", "WDL1nn")
geospred <- extract(pred, geos1.proj)

# presence/absence of Woodland (present = Y, absent = N)
WDL1ens <- cbind.data.frame(geos1$WDL, geospred)
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
geospred <- extract(pred, geos1.proj)

WDLens <- cbind.data.frame(geos1$WDL, geospred)
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
                 method = "glmnet",
                 metric= "ROC",
                 trControl = ens)
confusionMatrix(WDL.ens)
WDLens.pred <- predict(WDL.ens, WDLensTest,  type="prob") ## predict test-set
WDL.test <- cbind(WDLensTest, WDLens.pred)
WDLp <- subset(WDL.test, WDL1=="Y", select=c(Y))
WDLa <- subset(WDL.test, WDL1=="N", select=c(Y))
WDL.eval <- evaluate(p=WDLp[,1], a=WDLa[,1]) ## calculate ROC's on test set <dismo>
WDL.eval

#spatial predictions
WDLens.pred <- predict(pred, WDL.ens, type="prob") 

#create results
dir.create("./NG_results")
writeRaster(1-WDLens.pred, filename = "./NG_results/WDLens.tif")
WDLens=1-WDLens.pred
WDLens=scale(WDLens, center = TRUE, scale =TRUE)
t=addLayer(t,WDLens)
names(t)[37]="WDLENS" #check numbers

# on 15k
#human settlements
#+ Data setup --------------------------------------------------------------
# Extract gridded variables at GeoSurvey locations
#geoall.proj=geos1.proj+geo.proj
geogrid=extract(t,geos1.proj)#extract 15 k
# Assemble dataframes

#presence absence of settlements
HSP1=geos1$HSP
prop.table(table(HSP1))
hsp1dat=cbind.data.frame(HSP1, geogrid)
hsp1dat=na.omit(hsp1dat)
colnames(hsp1dat)[1]= "HSP1"
#First prediction of settlements
#+ Split data into train and test sets ------------------------------------
seed=1345
set.seed=seed

# hsp train/test split
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
geospred <- extract(pred, geos1.proj)

# presence/absence of settlements (present = Y, absent = N)
HSP1ens <- cbind.data.frame(HSP1, geospred)
HSP1ens <- na.omit(HSP1ens)
HSP1ensTest <- HSP1ens[-hsp1Index,] ## replicate previous test set
names(HSP1ensTest)[1]= "HSP1"

# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens<- trainControl(method='cv', number=10, returnResamp='none', 
                   allowParallel = TRUE, classProbs = TRUE,
                   summaryFunction= twoClassSummary)
# presence/absence of hsp (present = Y, absent = N)
HSP1.ens <- train(HSP1 ~. , data = HSP1ensTest,
                  family = "binomial", 
                  method = "glmnet",
                  metric= "ROC",
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

geospred <- extract(pred, geos1.proj)

# presence/absence of settlements (present = Y, absent = N)
HSPens <- cbind.data.frame(geos1$HSP, geospred)
HSPens <- na.omit(HSPens)
HSPensTest <- HSPens[-hsp1Index,] ## replicate previous test set
names(HSPensTest)[1]= "HSP"

mc <- makeCluster(detectCores())
registerDoParallel(mc)
# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method='cv', number=10, returnResamp='none', 
                    allowParallel = TRUE, 
                    classProbs = TRUE,
                    summaryFunction= twoClassSummary)
# presence/absence of hsp (present = Y, absent = N)
HSP.ens <- train(HSP ~. , data = HSPensTest,
                 family = "binomial", 
                 metric= "ROC",
                 method = "rf",
                 trControl = ens)
confusionMatrix(HSP.ens)
HSPens.pred <- predict(HSP.ens, HSPensTest,  type="prob") ## predict test-set
HSP.test <- cbind(HSPensTest, HSPens.pred)
HSPp <- subset(HSP.test, HSP=="Y", select=c(Y))
HSPa <- subset(HSP.test, HSP=="N", select=c(Y))
HSP.eval <- evaluate(p=HSPp[,1], a=HSPa[,1]) ## calculate ROC's on test set <dismo>
HSP.eval

#spatial predictions
HSPens.pred <- predict(pred, HSP.ens, type="prob") 
writeRaster(1-HSPens.pred, filename = "./NG_results/HSPens.tif", overwrite=TRUE)
HSPens=1-HSPens.pred
HSPens=scale(HSPens, center = TRUE, scale =TRUE)
t=addLayer(t, HSPens)
names(t)[39]= "HSPENS"

#predict cropland on 15k
geos1grid=extract(t, geos1.proj)
crp1dat= cbind.data.frame(geos1$CRP, geos1grid)
colnames(crp1dat)[1]= "CRP"
crp1dat=na.omit(crp1dat)

#divide in train and test data
seed=12345
set.seed=seed

# crp train/test split
crp1Index <- createDataPartition(crp1dat$CRP, p = 2/3, list = FALSE, times = 1)
crp1Train <- crp1dat[ crp1Index,]
crp1Test  <- crp1dat[-crp1Index,]

mc <- makeCluster(detectCores())
registerDoParallel(mc)


#glmnet--------------------------------
objControl <- trainControl(method='cv', number=10, returnResamp='none', 
                           allowParallel = TRUE, classProbs = TRUE,
                           summaryFunction= twoClassSummary)

CRP1.glm=train(CRP ~ ., data=crp1Train, family= "binomial",method="glmnet",
               metric="ROC", 
               trControl=objControl)

confusionMatrix(CRP1.glm)
CRP1glm.pred <- predict(t, CRP1.glm, type="prob") ## spatial predictions
plot(varImp(CRP1.glm,scale=F))

#rf
CRP1.rf <- train(CRP ~ ., data = crp1Train,
                 method = "rf",
                 ntree=501,
                 metric= "ROC",
                 trControl = objControl)
confusionMatrix(CRP1.rf)
crp1rf.pred <- predict(t, CRP1.rf, type = "prob") ## spatial predictions
plot(varImp(CRP1.rf,scale=F),main="Variable Importance Random Forest Cropland")

#gbm
CRP1.gbm <- train(CRP ~ ., data = crp1Train,
                  method = "gbm",
                  metric = "ROC",
                  trControl = objControl)
confusionMatrix(CRP1.gbm)
crp1gbm.pred <- predict(t, CRP1.gbm, type = "prob") ## spatial predictions
#neural net
CRP1.nn <- train(CRP ~., data=crp1Train, 
                 method = "nnet", 
                 metric= "ROC",
                 #preProc = c("center", "scale"), 
                 trControl = objControl)
print(CRP1.nn)
CRP1.imp <- varImp(CRP1.nn, useModel = FALSE)
plot(CRP1.imp, top=27)
crp1nn.pred <- predict(t, CRP1.nn, type="prob")


#ensemble regression glmnet (elastic net)
pred <- stack(1-CRP1glm.pred, 1-crp1rf.pred, 
              1-crp1gbm.pred, 1-crp1nn.pred)
names(pred) <- c("CRPglm", "CRPrf","CRPgbm", "CRPnn")

geospred <- extract(pred, geos1.proj)

# presence/absence of settlements (present = Y, absent = N)
CRPens <- cbind.data.frame(geos1$CRP, geospred)
CRPens <- na.omit(CRPens)
CRPensTest <- CRPens[-crp1Index,] ## replicate previous test set
names(CRPensTest)[1]= "CRP"

mc <- makeCluster(detectCores())
registerDoParallel(mc)
# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method='cv', number=10, returnResamp='none', 
                    allowParallel = TRUE, 
                    classProbs = TRUE,
                    summaryFunction= twoClassSummary)
# presence/absence of hsp (present = Y, absent = N)
CRP.ens <- train(CRP ~. , data = CRPensTest,
                 family = "binomial", 
                 metric= "ROC",
                 method = "glmnet",
                 trControl = ens)
confusionMatrix(CRP.ens)
CRPens.pred <- predict(CRP.ens, CRPensTest,  type="prob") ## predict test-set
CRP.test <- cbind(CRPensTest, CRPens.pred)
CRPp <- subset(CRP.test, CRP=="Y", select=c(Y))
CRPa <- subset(CRP.test, CRP=="N", select=c(Y))
CRP.eval <- evaluate(p=CRPp[,1], a=CRPa[,1]) ## calculate ROC's on test set <dismo>
CRP.eval
plot(CRP.eval, "ROC")

#for all 15k obs
#rf
CRP1.rf <- train(CRP ~ ., data = crp1dat,
                 method = "rf",
                 ntree=501,
                 metric= "ROC",
                 trControl = objControl)
confusionMatrix(CRP1.rf)
crp1rf.pred <- predict(t, CRP1.rf, type = "prob") ## spatial predictions
plot(varImp(CRP1.rf,scale=F),main="Variable Importance Random Forest Cropland")

#gbm
CRP1.gbm <- train(CRP ~ ., data = crp1dat,
                  method = "gbm",
                  metric = "ROC",
                  trControl = objControl)
confusionMatrix(CRP1.gbm)
crp1gbm.pred <- predict(t, CRP1.gbm, type = "prob") ## spatial predictions
#neural net
CRP1.nn <- train(CRP ~., data=crp1dat, 
                 method = "nnet", 
                 metric= "ROC",
                 #preProc = c("center", "scale"), 
                 trControl = objControl)
crp1nn.pred <- predict(t, CRP1.nn, type="prob")

#ensemble
pred <- stack(1-crp1rf.pred, 
              1-crp1gbm.pred, 1-crp1nn.pred)
names(pred) <- c("CRPrf","CRPgbm", "CRPnn")

geospred <- extract(pred, geos1.proj)

# presence/absence of cropland (present = Y, absent = N)
CRPens <- cbind.data.frame(geos1$CRP, geospred)
CRPens <- na.omit(CRPens)
CRPensTest <- CRPens[-crp1Index,] ## replicate previous test set
names(CRPensTest)[1]= "CRP"

mc <- makeCluster(detectCores())
registerDoParallel(mc)
# Regularized ensemble weighting on the test set <glmnet>
# 10-fold CV
ens <- trainControl(method='cv', number=10, returnResamp='none', 
                    allowParallel = TRUE, 
                    classProbs = TRUE,
                    summaryFunction= twoClassSummary)
# presence/absence of cropland (present = Y, absent = N)
CRP.ens <- train(CRP ~. , data = CRPensTest,
                 family = "binomial", 
                 metric= "ROC",
                 method = "glmnet",
                 trControl = ens)
confusionMatrix(CRP.ens)
CRPens.pred <- predict(CRP.ens, CRPensTest,  type="prob") ## predict test-set
CRP.test <- cbind(CRPensTest, CRPens.pred)
CRPp <- subset(CRP.test, CRP=="Y", select=c(Y))
CRPa <- subset(CRP.test, CRP=="N", select=c(Y))
CRP.eval <- evaluate(p=CRPp[,1], a=CRPa[,1]) ## calculate ROC's on test set <dismo>
CRP.eval
plot(CRP.eval, "ROC")
#spatial predictions
CRPens.pred=predict(pred, CRP.ens, type= "prob")
writeRaster(1-CRPens.pred, filename = "./NG_results/cropens.tif")