# Script for crop trial representativity
# basis for cropland mask is the 1 Million point survey for Africa of 2014-2015 conducted by AfSIS for Africa
# grids are from Africasoils.net
# field data are collected by TanSIS in Tanzania 2015-2016 
# representativeness is based on selected covariates by ACAI
#Alex Verlinden November 2016 
#+ Required packages
# install.packages(c("downloader","raster","rgdal", "caret")), dependencies=TRUE)
require(downloader)
require(raster)
require(rgdal)
require(dismo)
require(caret)

#+ Data downloads ----------------------------------------------------------
# Create a "Data" folder in your current working directory

dir.create("TZ_cass", showWarnings=F)
dat_dir <- "./TZ_cass"

#download ACAI trial site data
download.file("https://www.dropbox.com/s/zldu22dnlrwquh5/Site%20description%20EZ%2007092016-PP.csv?dl=0", "./TZ_cass/Site%20description%20EZ%2007092016-PP.csv", mode="wb")
acai=read.csv(paste(dat_dir, "/Site%20description%20EZ%2007092016-PP.csv", sep="" ), header=T, sep=",")
#select Fertilizer trials
acaiFR=subset(acai, acai$Use.case=="FR")
#download grids for TZ  ~86 MB
download.file("https://www.dropbox.com/s/utjtskk7k7rmliw/TZ_grids250.zip?dl=0","./TZ_cass/TZ_grids250.zip",  mode="wb")
unzip("./TZ_cass/TZ_grids250.zip", exdir=dat_dir, overwrite=T)
glist <- list.files(path=dat_dir, pattern="tif", full.names=T)
grid <- stack(glist)
t=scale(grid, center=TRUE, scale=TRUE)

#project acai data to grid TZ
acai.proj=as.data.frame(project(cbind(acaiFR$Long.corr, acaiFR$Lat.corr),"+proj=laea +ellps=WGS84 +lon_0=20 +lat_0=5 +units=m +no_defs"))
colnames(acai.proj)=c("x", "y")
coordinates(acai.proj)= ~x+y
projection(acai.proj)=projection(grid)

# Extract gridded variables for all TZ ACAI FR data
acaiF=data.frame(coordinates(acai.proj), extract(t, acai.proj))
acaiF= acaiF[,3:11]# exclude coordinates

#download cropmask
dir.create("./TZ_cropland")
download.file("https://www.dropbox.com/s/glmgpwp4h6wm83r/TZ_cropland.zip?dl=0","./TZ_cropland/TZ_cropland.zip",  mode="wb")
unzip("./TZ_cropland/TZ_cropland.zip", exdir= "./TZ_cropland", overwrite=T)

crp=raster("./TZ_cropland/TZ_cropland.tif")
crp[crp==0]=NA

#download cassava mask of 2016
download.file("https://www.dropbox.com/s/pr67j9u11v4738g/TZ_cassava_250m.zip?dl=0", "./TZ_cropland/TZ_cassava_250m.zip", mode="wb")
unzip("./TZ_cropland/TZ_cassava_250m.zip", exdir= "./TZ_cropland", overwrite=T)
cass=raster("./TZ_cropland/TZ_cassava_250m.tif")
cass[cass==0]=NA
#select random points from cassava background
set.seed(15234)
n=randomPoints(cass, 8000, acai.proj, excludep = TRUE) # there are a lot of NAS
n=n[1:300,]
#extract covariates for random points in predicted cropland, scaled
nextrs=extract(t,n)
nextrs=na.omit(nextrs)
nextrs=as.data.frame(nextrs)

xextr=data.frame("Y",acaiFR) # make new column with Y= "present" where trials will be undertaken
colnames(xextr)[1]="presabs"
rextr=data.frame("N",nextrs) # make column with N for the x random samples for the area where no trials are undertaken
colnames(rextr)[1]="presabs"
extrap=rbind(xextr,rextr)
extrap=na.omit(extrap)
extrap$presabs=as.factor(extrap$presabs)
extrapIndex <- createDataPartition(extrap$presabs, p = 2/3, list = FALSE, times = 1)
extrapTrain <- extrap[ extrapIndex,]
extrapTest  <- extrap[-extrapIndex,]
extrapTest= na.omit(extrapTest)
objControl <- trainControl(method='cv', number=3, classProbs = T,returnResamp='none')
#glmnet using binomial distribution for cassava training set = elastic net regression for penalizing
cass.glm=train(presabs ~ ., data=extrapTrain, family= "binomial",method="glmnet",metric="Accuracy", trControl=objControl)

predictions <- predict(cass.glm, extrapTest[,1:10], type="prob")
#confusionMatrix on cross validation
confusionMatrix(cass.glm)
#variable importance with glmnet you see pos and neg influences  on positive side means that higher levels are underrepresented
plot(varImp(cass.glm,scale=F))

# Checking the AUC, fr this the AUC should be close to 0.5
#when you run this with a different stratified sample you will see big differences in AUC, normal for only 150
casstest=cbind(extrapTest, predictions)
cassp=subset(casstest, presabs=="Y", select=c(Y) )
cassa=subset(casstest, presabs=="N", select=c(Y))
cass.eval=evaluate(p=cassp[,1],a=cassa[,1])
cass.eval
plot(cass.eval, 'ROC') ## plot ROC curve
cass.thld <- threshold(cass.eval, 'spec_sens') ## TPR+TNR threshold for classification
#spatial predictions
cassglm.pred <- predict(t,cass.glm, type="prob") 
cassmask=cassglm.pred>cass.thld
cassmask2=cassmask*cass
plot(cassmask2, legend=F)
points(acai.proj, col="red", cex=0.2)
writeRaster(cassmask2, filename= "casspred.tif", overwrite=TRUE)

#kmeans clustering

#for standardized covariates 
cc= t*cass #as t = already a brick for cluster select only cassava
cc[is.na(cc)]=0
set.seed(222)
km=kmeans(cc[],10, iter.max = 100, nstart =3) # only 10 clusters + 0
kmrs=raster(cc) #to turn result into raster image first create raster
kmrs[]=km$cluster #turn ids of layer into cluster ids
plot(kmrs)
kmrsc=kmrs*cass
plot(kmrsc, main = "10 Cassava agro-ecological zones")
points(acai.proj, col="red", cex=0.2)
#plot
casskm= extract(kmrsc, acai.proj)
hist(casskm, main="Frequency of NOT trials in Clusters")
hist(kmrsc, nclass=10, main="Frequency of Clusters in Cassava areas from AEZ map")
writeRaster(kmrsc, filename="TZ_AEZ_km.tif")

