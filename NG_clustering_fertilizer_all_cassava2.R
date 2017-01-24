# clustering of NIGERIA fertilizer data to select 15/30 Agent locations
# install.packages(c("downloader","raster","rgdal", "caret", e1071")), dependencies=TRUE)
# currently incomplete needs cleaning up some libraries not (yet) needed
#kmeans  clustering is used to group covariates in "agro ecologies"
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)
require(e1071)
require (MASS)
require(cluster) 
dir.create("NG_cass", showWarnings=F)
dat_dir <- "./NG_cass"


# download data from 2015 locations of extension agents
download.file("https://www.dropbox.com/s/d0x9lxkbdtpfy70/NG_pts_fert.csv?dl=0", "./NG_cass/NG_pts_fert.csv", mode="wb")
fert1=read.table(paste(dat_dir, "/NG_pts_fert.csv", sep=""), header=T, sep=",")
#reduce columns to coords
fert1=fert1[,3:5]
#only ibjectid
fert2=fert1[,3]
#download grids for NG fert  30 MB fetilizer tools area
download.file("https://www.dropbox.com/s/8kt9xstqzhnsaht/NG_CASS_area.zip?dl=0","./NG_cass/NG_cass_area.zip")
unzip("./NG_cass/NG_cass_area.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid, center=TRUE, scale= TRUE)
#+ Data setup --------------------------------------------------------------
# Project test data to grid CRS
NOT_NG.proj <- as.data.frame(project(cbind(fert1$X.1, fert1$Y.1), "+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(NOT_NG.proj) <- c("x","y")
coordinates(NOT_NG.proj) <- ~x+y  #convert to Spatial DataFrame
projection(NOT_NG.proj) <- projection(grid)

# Extract gridded variables for NG fert to include observations with random set
# scaled variables
NOTexs <- data.frame(coordinates(NOT_NG.proj), extract(t, NOT_NG.proj))
NOTexs <- na.omit(NOTexs)
NOTexs=NOTexs[,3:12]
#download predicted cropland for NG fertilizer trials
#note that when you repeat the script, this cropland file will be included in the grid!
download.file("https://www.dropbox.com/s/4hiivua0l8jyyvw/cropland_cass_area.zip?dl=0", "./NG_cass/cropland_cass_area.zip", mode="wb")
unzip("./NG_cass/cropland_cass_area.zip", exdir=dat_dir, overwrite=T)
ng_rand=raster("./NG_fert/cropland_cass_area.tif")
#create random points for predicted cropland as background
# for testing we take random samples as background data from NG fertilizer area cropland mask
set.seed(1234)
n=randomPoints(ng_rand, 1000)
#extract covariates for random points in predicted cropland
nextr=extract(t,n)
nextr=na.omit(nextr)
nextr=as.data.frame(nextr)
NOT=rbind(NOTexs,nextr)

#kmeans clustering
#select number of clusters for kmeans
wss <- (nrow(NOT)-1)*sum(apply(NOT,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(NOT, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

# K-Means Cluster Analysis
fit <- kmeans(NOT, 15) # 15 cluster solution
# get cluster means 
aggregate(NOT,by=list(fit$cluster),FUN=mean)
# append cluster assignment
NOTc <- data.frame(NOT, fit$cluster)

#plot kmeans
clusplot(NOTs, fit$cluster, color=TRUE, shade=TRUE, 
         labels=2, cex= 0.5,lines=0)


# selection done manually after deliniating clusters in a spreadsheet and marking agents samples
test=NOTex[c(6,8,12,13,15,16,19,21,28,34,37,44,46,53,56),] # just an example
write.csv(test,"testsites.csv", row.names=T)
testcoord=test[,1:2]
# add id column
testcoord$id=c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
coordinates(testcoord)=~x+y
# buffer on raster width = 5000m
b=rasterize(testcoord, ng_rand, "id") #fun="count"
par(mar=c(1,1,1,1))
plot(b)
b1=buffer(b, width=5000, copyfields= TRUE)
#select 150 samples (fail to do so systematically 10 per point so far)
x=sampleStratified(b1, size=150, xy=TRUE)
#save coordinates for 150 samples around 15 locations
y=x[,2:3]
write.csv(y,"test_plot.csv")
xextr=extract(t, y)
xextr=na.omit(xextr)
xextr=as.data.frame(xextr)
xextr=data.frame("Y",xextr) # make new column with Y= "present"
colnames(xextr)[1]="pres"
nextr=as.data.frame(nextr)
rextr=data.frame("N",nextr) # make clumn with N for random samples
colnames(rextr)[1]="pres"
extrap=rbind(xextr,rextr)
colnames(extrap)[1]="pres"
extrap=na.omit(extrap)
extrap$pres=as.factor(extrap$pres)
extrapIndex <- createDataPartition(extrap$pres, p = 2/3, list = FALSE, times = 1)
extrapTrain <- extrap[ extrapIndex,]
extrapTest  <- extrap[-extrapIndex,]
extrapTest= na.omit(extrapTest)
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for cassava training set
cass.glm=train(pres ~ ., data=extrapTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(cass.glm, extrapTest[,1:11], type="prob")
#confusionMatrix on cross validation
confusionMatrix(cass.glm)
#variable importance
plot(varImp(cass.glm,scale=F))

casstest=cbind(extrapTest, predictions)
cassp=subset(casstest, pres=="Y", select=c(Y) )
cassa=subset(casstest, pres=="N", select=c(Y))
cass.eval=evaluate(p=cassp[,1],a=cassa[,1])
cass.eval
plot(cass.eval, 'ROC') ## plot ROC curve
cass.thld <- threshold(cass.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cassglm.pred <- predict(grid,cass.glm, type="prob") 
cassmask=cassglm.pred>cass.thld
cassmask2=cassmask*ng_rand
plot(cassmask2, legend=F)
points(test, col="red")
points(y, cex=0.2)


#for scaled covariates
xextrs=extract(t, y)
xextrs=na.omit(xextrs)
xextrs=data.frame("Y",xextrs) # make new column with Y= "present"
colnames(xextrs)[1]="pres"
#for scaled n random
nextrs=extract(t,n)
nextrs=na.omit(nextrs)
nextrs=as.data.frame(nextrs)
rextrs=data.frame("N",nextrs) # make column with N for random samples
colnames(rextrs)[1]="pres"
extraps=rbind(xextrs,rextrs)
colnames(extraps)[1]="pres"
extraps=na.omit(extraps)
extraps$pres=as.factor(extraps$pres)
extrapsIndex <- createDataPartition(extraps$pres, p = 2/3, list = FALSE, times = 1)
extrapsTrain <- extraps[ extrapsIndex,]
extrapsTest  <- extraps[-extrapsIndex,]
extrapsTest= na.omit(extrapsTest)
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for cassava training set
casss.glm=train(pres ~ ., data=extrapsTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictionss <- predict(casss.glm, extrapsTest[,1:11], type="prob")
#confusionMatrix on cross validation
confusionMatrix(casss.glm)
#variable importance
plot(varImp(casss.glm,scale=F))
dev.off()
cassstest=cbind(extrapsTest, predictionss)
casssp=subset(cassstest, pres=="Y", select=c(Y) )
casssa=subset(cassstest, pres=="N", select=c(Y))
casss.eval=evaluate(p=casssp[,1],a=casssa[,1])
casss.eval
plot(casss.eval, 'ROC') ## plot ROC curve
casss.thld <- threshold(casss.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions with scaled variables
casssglm.pred <- predict(t,casss.glm, type="prob") 
casssmask=casssglm.pred>casss.thld
casssmask2=casssmask*ng_rand
plot(casssmask2, legend=F)
points(test, col="red")
points(y, cex=0.2)
#cluster in image
f=t
f[is.na(f)]=0
cc=f*ng_rand
km=kmeans(f[],16, iter.max = 100, nstart =3)
kmrs=raster(f)
kmrs[]=km$cluster
plot(kmrs)
kmrsc=kmrs*ng_rand
plot(kmrsc)
points(test, col="red")
points(y, cex=0.2)
rf=writeRaster(kmrsc, filename="./NG_fert/NG_class25", format= "GTiff", overwrite=TRUE)
NOTsites <- data.frame(coordinates(NOT_NG.proj), extract(kmrsc, NOT_NG.proj))
colnames(NOTsites)[3]="CLUS"
hist(NOTsites$CLUS, breaks=25)
par(mfrow=c(1,2))
hist(kmrsc)
hist(NOTsites$CLUS, breaks=25)
