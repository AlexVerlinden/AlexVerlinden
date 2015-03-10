#' Ensemble predictions of a portion of Namibia GeoSurvey bush encroachment,
#' woody vegetation cover observations. 
#' Alex Verlinden March 2015, modified and based on M. Walsh, November 2014

# Required packages 
# some might no more be needed
# install.packages(c("downloader","raster","rgdal","caret","MASS","randomForest","gbm","nnet","glmnet","dismo")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(caret)
require(MASS)
require(randomForest)
require(gbm)
require(nnet)
require(glmnet)
require(dismo)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory
dir.create("NAM_data", showWarnings=F)
dat_dir <- "./NAM_data"

# download GeoSurvey Namibia data
download("https://www.dropbox.com/s/7v1nskkdafhj6ni/namibia_bush.csv?dl=0", "./NAM_data/namibia_bush.csv", mode="wb")
bush <- read.table(paste(dat_dir, "/namibia_bush.csv", sep=""), header=T, sep=",")
bush <- na.omit(bush)

# download Namibia Gtifs (~44.5 Mb) and stack in raster
download("https://www.dropbox.com/s/9w0gb4tq8c9ivij/NAM.zip?dl=0", "./NAM_data/NAM.zip", mode="wb")
unzip("./NAM_data/NAM.zip", exdir="./NAM_data", overwrite=T)
glist <- list.files(path="./NAM_data", pattern="tif", full.names=T)
grid <- stack(glist)

#+ Data setup --------------------------------------------------------------
# Project GeoSurvey coords to grid CRS
bush.proj <- as.data.frame(project(cbind(bush$Longitude, bush$Latitude), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(bush.proj) <- c("x","y")
bush <- cbind(bush, bush.proj)
coordinates(bush) <- ~x+y
projection(bush) <- projection(grid)

# Extract gridded variables at GeoSurvey locations
bushgrid <- extract(grid, bush)
#extent (xmin, xmax, ymin, ymax)
ext= extent (-340000, -260000, -2760000,-2710000)
# Assemble dataframes
# presence/absence of high bush cover (X.6, present = Y, absent = N)
bush60 <- bush$X.6
bush6dat <- cbind.data.frame(bush60, bushgrid)
bush6dat <- na.omit(bush6dat)



# set train/test set randomization seed
seed <- 1385321
set.seed(seed)

#+ Split data into train and test sets ------------------------------------
# bushland train/test split
bush6Index <- createDataPartition(bush6dat$bush60, p = 0.75, list = FALSE, times = 1)
bush6Train <- bush6dat[ bush6Index,]
bush6Test  <- bush6dat[-bush6Index,]



#+ Stepwise main effects GLM's <MASS> --------------------------------------
# 10-fold CV
step <- trainControl(method = "cv", number = 10)

# presence/absence of bushland (bush60dat, present = Y, absent = N)
bush6.glm <- train(bush60 ~ ., data = bush6Train,
                 family = binomial, 
                 method = "glmStepAIC",
                 trControl = step)
bush6glm.test <- predict(bush6.glm, bush6Test) ## predict test-set
confusionMatrix(bush6glm.test, bush6Test$bush60, "Y") ## print validation summaries
bush6glm.pred <- predict(grid, bush6.glm, type = "prob", ext =ext) ## spatial predictions
plot (1- bush6glm.pred)

#+ Random forests <randomForest> -------------------------------------------
# out-of-bag predictions
oob <- trainControl(method = "oob")

# presence/absence of bushland (>60%, present = Y, absent = N)
bush6.rf <- train(bush60 ~ ., data = bush6Train,
                method = "rf",
                trControl = oob)
bushrf.test <- predict(bush6.rf, bush6Test) ## predict test-set
confusionMatrix(bushrf.test, bush6Test$bush60, "Y") ## print validation summaries
bushrf.pred <- predict(grid, bush6.rf, ext=ext, type = "prob") ## spatial predictions


#+ Gradient boosting <gbm> ------------------------------------------
# CV for training gbm's
gbm <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# presence/absence of >60% bush cover (bush60, present = Y, absent = N)
bush6.gbm <- train(bush60 ~ ., data = bush6Train,
                 method = "gbm",
                 trControl = gbm)
bush6gbm.test <- predict(bush6.gbm, bush6Test) ## predict test-set
confusionMatrix(bush6gbm.test, bush6Test$bush60, "Y") ## print validation summaries
bush6gbm.pred=predict(grid, bush6.gbm, ext=ext, type = "prob")
# presence/absence of Woody Vegetation Cover >60% (WCP, present = Y, absent = N)

#+ Neural nets <nnet> ------------------------------------------------------
# CV for training nnet's
nn <- trainControl(method = "cv", number = 10)

# presence/absence of >60% bush(bush60, present = Y, absent = N)
bush60.nn <- train(bush60 ~ ., data = bush6Train,
                method = "nnet",
                trControl = nn)
bush60nn.test <- predict(bush60.nn, bush6Test) ## predict test-set
confusionMatrix(bush60nn.test, bush6Test$bush60, "Y") ## print validation summaries
bush60nn.pred <- predict(grid, bush60.nn, ext=ext, type = "prob") ## spatial predictions


#+ Plot predictions by GeoSurvey variables ---------------------------------
# bush encroachment >60 % prediction plots
bush60.preds <- stack(1-bush6glm.pred, 1-bushrf.pred, 1-bush6gbm.pred, 1-bush60nn.pred)
names(bush60.preds) <- c("glmStepAIC","randomForest","gradient boosting","neural net")
plot(bush60.preds, axes = F)

bush60pred=extract(bush60.preds, bush)


# presence/absence of bush>60 % (bush>60%, present = Y, absent = N)
bushens <- cbind.data.frame(bush60, bush60pred)
bushens <- na.omit(bushens)
bushensTest <- bushens[-bush6Index,] ## replicate previous test set


# Regularized ensemble weighting on the test set <glm>
# 5-fold CV
ens <- trainControl(method = "cv", number = 5)

# presence/absence of bushland (bush 60, present = Y, absent = N)
bush.ens <- train(bush60 ~ glmStepAIC + randomForest + gradient.boosting + neural.net, data = bushensTest,
                 family = "binomial", 
                 method = "glmnet",
                 trControl = ens)
bushens.pred <- predict(bush.ens, bushensTest, type="prob")
bush.test <- cbind(bushensTest, bushens.pred)
bush <- subset(bush.test, bush60=="Y", select=c(Y))
busha <- subset(bush.test, bush60=="N", select=c(Y))
bush.eval <- evaluate(p=bush[,1], a=busha[,1]) ## calculate ROC's on test set <dismo>
bush.eval
plot(bush.eval, 'ROC') ## plot ROC curve
bush.thld <- threshold(bush.eval, 'spec_sens') ## TPR+TNR threshold for classification
bushens.pred <- predict(bush60.preds, bush.ens, type="prob") ## spatial prediction
bushmask <- 1-bushens.pred > bush.thld
plot(bushmask, axes = F, legend = F)


#+ Write spatial predictions -----------------------------------------------
# Create a "Results" folder in current working directory
dir.create("NAM_results", showWarnings=F)

# Export Gtif's to "./NAM_results"

#write tiff
rf=writeRaster(bushmask,filename="./NAM_Results/bush_60%.tif", format= "GTiff", overwrite = TRUE)

