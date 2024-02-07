# David Chaucayanqui
# Quantitative methods in finance
# Replicate paper:
# Does Correlation betwee Stock Returns really increase during Turbulent periods?
# Francois Chesnay, Eric Jondeau
# 2000, Journal of Economic Perspectives
# using PWT data (?)

closeAllConnections()
rm(list=ls())
setwd("/Users/User/Documents/R/projects/QMF")
library(lmtest)
library(heavy)
library(e1071) 
library(tseries)
library(zoo)
library(reshape)
library(ggplot2)
library(urca)
library(lmtest)
library(forecast)
library(TSA)
library(xts)
library(reshape2)
library(xtable)
library(mFilter)
library(caret)
library(gdata)
library(aod)
library(tsDyn)
library(corrplot)
library(openair)
library(psych)
library(MSGARCH)
library(MSwM)

# import the data
df <- read.csv('Data.csv')

# check the classes of the columns
sapply(df,class)
any(is.na(df))
# all values are numeric, no NAs

# convert dates to date format
df$Date <- as.Date(df$Date,format="%m/%d/%Y")
# set the dates as index
row.names(df) <- df$Date
colnames(df)= c("Date", "SP", "DAX", "FTSE")



###plots to observe the dynamics
ggplot( data = df, aes( Date, SP )) + geom_line()
dev.copy(jpeg,filename="sp.jpg")
dev.off()
ggplot( data = df, aes( Date, DAX )) + geom_line()
dev.copy(jpeg,filename="dax.jpg")
dev.off()
ggplot( data = df, aes( Date, FTSE )) + geom_line()
dev.copy(jpeg,filename="ftse.jpg")
dev.off()
#the dynamics is similar


#creating returns as %
attach(df)
retsp <- diff(log(SP), lag=1)*100
retdax <- diff(log(DAX), lag=1)*100
retftse <- diff(log(FTSE), lag=1)*100
df<-df[-1,]
ggplot( data = df, aes( Date, retsp )) + geom_line()
dev.copy(jpeg,filename="retsp.jpg")
dev.off()
ggplot( data = df, aes( Date, retdax )) + geom_line()
dev.copy(jpeg,filename="retdax.jpg")
dev.off()
ggplot( data = df, aes( Date, retftse )) + geom_line()
dev.copy(jpeg,filename="retftse.jpg")
dev.off()
#we can assume that the volatilities shift in these 3 markets at the same periods
#here we create a matrix with all the returns
indexes <- cbind(retsp,retdax,retftse)


###observing data
#mean and st deviation
mean(retsp)
mean(retdax)
mean(retftse)
sd(retsp)
sd(retdax)
sd(retftse)
skewness(retsp)
skewness(retdax)
skewness(retftse)
#normality
library(EnvStats)
kurtosis(retsp,excess = TRUE)
kurtosis(retdax,excess = TRUE)
kurtosis(retftse,excess = TRUE)
#fat tails
library("ggpubr")
ggdensity(retsp, 
          main = "Density plot of returns of S&P",
          xlab = "Return")
ggdensity(retdax, 
          main = "Density plot of returns of DAX",
          xlab = "Return")
ggdensity(retftse, 
          main = "Density plot of returns of FTSE",
          xlab = "Return")
#indeed fat tails on the graphs
#normality test
library(stats)
shapiro.test(retsp)
shapiro.test(retdax)
shapiro.test(retftse)
#low pvalues, reject H0 about normality

#ADF
adf.test(retsp)
adf.test(retdax)     
adf.test(retftse)     
#pvalues are lower than 0,01
#reject H0 about unit root 

#White test for heteroskedasticity
library(het.test)
ret<-data.frame(cbind(retsp,retdax,retftse))
model <- VAR(ret, p = 1, type="none")
whites.htest(model)
#test stat 146,8 with p-value lower than 0,01
#reject HO about homoskedastisity

#serial correlation
#Durbin-Watson test
library(orcutt)
model1<-lm(retsp~retdax+retftse)
model2<-lm(retdax~retsp+retftse)
model3<-lm(retftse~retsp+retdax)
dwtest(model1)
dwtest(model2)
dwtest(model3)
#high p-values, no serial correlation

rm(model, model1, model2, model3)

#Creating subsamples (3 different subsamples)
firstsub<-indexes[1:208,]
secondsub<-indexes[210:416,]
thirdsub<-indexes[418:625,]

#this is an alternate way to subsetting
#first<-with(indexes, indexes[(Date <= "1991-30-12")])
#second<-with(indexes, indexes[(Date <= "1995-25-12" & Date>="1992-06-01")])
#third<-with(indexes, indexes[(Date <= "1999-27-12" & Date>="1996-01-01")])

#VAriance and covariance of the different subsamples and the sample and the different indexes
cor1<-cor(indexes)
cor1
cor2<-cor(firstsub)
cor2
cor3<-cor(secondsub)
cor3
cor4<-cor(thirdsub)
cor4

varsp<-sd(indexes[,1])**2
vardax<-sd(indexes[,2])**2
varftse<-sd(indexes[,3])**2


sd(firstsub[,1])**2
sd(firstsub[,2])**2
sd(firstsub[,3])**2

sd(secondsub[,1])**2
sd(secondsub[,2])**2
sd(secondsub[,3])**2

sd(thirdsub[,1])**2
sd(thirdsub[,2])**2
sd(thirdsub[,3])**2

#this are the correlation matrices and the jennrich test amond the subsamples and among the different indexes
jennrich1 <- cortest.jennrich(R1=cor2,R2=cor3,n1=209, n2=208)
jennrich2 <- cortest.jennrich(R1=cor3,R2=cor4,n1=208, n2=209)
jennrich1
jennrich2
corspdax2 <- cor(firstsub[,1:2])
corspdax3 <- cor(secondsub[,1:2])
corspdax4 <- cor(thirdsub[,1:2])

jennrichspdax1<-cortest.jennrich(R1=corspdax2,R2=corspdax3,n1=209, n2=208)
jennrichspdax2<-cortest.jennrich(R1=corspdax3,R2=corspdax4,n1=208, n2=209)
jennrichspdax1
jennrichspdax2

corspftse2<-cor(firstsub[,c(1,3)])
corspftse3<-cor(secondsub[,c(1,3)])
corspftse4<-cor(thirdsub[,c(1,3)])

jennrichspftse1<-cortest.jennrich(R1=corspftse2,R2=corspftse3,n1=209, n2=208)
jennrichspftse2<-cortest.jennrich(R1=corspftse3,R2=corspftse4,n1=208, n2=209)
jennrichspftse1
jennrichspftse2

cordaxftse2<-cor(firstsub[,2:3])
cordaxftse3<-cor(secondsub[,2:3])
cordaxftse4<-cor(thirdsub[,2:3])

jennrichdaxftse1<-cortest.jennrich(R1=cordaxftse2,R2=cordaxftse3,n1=209, n2=208)
jennrichdaxftse2<-cortest.jennrich(R1=cordaxftse3,R2=cordaxftse4,n1=209, n2=208)
jennrichdaxftse1
jennrichdaxftse2

####CONSTANT VARIANCE FOR ONE and two regimes
#GAUSS dependent correlation
#Linear model for retsp gauss
modspg<-lm(retsp~1)

#Linear model for retdax gauss
moddaxg<-lm(retdax~1)

#Linear model for retftse gauss
modftseg<-lm(retftse~1)

#ONE and two regimes
modsp12=msmFit(modspg, k=2, sw=c(T,T), p=0)
summary(modsp12)
moddax12=msmFit(moddaxg, k=2, sw=c(T,T), p=0)
summary(moddax12)
modftse12=msmFit(modftseg, k=2, sw=c(T,T), p=0)
summary(modftse12)


####GARCH FOR ONE AND TWO REGIMES
#specification for markov switching model one regime, GARCH, gauss
specgg <- CreateSpec(variance.spec = list(model = c("sGARCH")),
                   distribution.spec = list(distribution = c("norm")),
                   switch.spec = list(do.mix = FALSE, K = 1))

#specification for markov switching model one regime, GARCH, student
specgs <- CreateSpec(variance.spec = list(model = c("sGARCH")),
                     distribution.spec = list(distribution = c("std")),
                     switch.spec = list(do.mix = FALSE, K = 1))

#specification for markov switching model one regime, GJR, gauss
specgrg <- CreateSpec(variance.spec = list(model = c("gjrGARCH")),
                     distribution.spec = list(distribution = c("norm")),
                     switch.spec = list(do.mix = FALSE, K = 1))

#specification for markov switching model one regime, GJR, student
specgrs <- CreateSpec(variance.spec = list(model = c("gjrGARCH")),
                     distribution.spec = list(distribution = c("std")),
                     switch.spec = list(do.mix = FALSE, K = 1))

#specification for markov switching model two regime, GARCH, gauss
spec2gg <- CreateSpec(variance.spec = list(model = c("sGARCH")),
                     distribution.spec = list(distribution = c("norm")),
                     switch.spec = list(do.mix = FALSE, K = 2))

#specification for markov switching model two regime, GARCH, student
spec2gs <- CreateSpec(variance.spec = list(model = c("sGARCH")),
                     distribution.spec = list(distribution = c("std")),
                     switch.spec = list(do.mix = FALSE, K = 2))

#specification for markov switching model two regime, GJR, gauss
spec2grg <- CreateSpec(variance.spec = list(model = c("gjrGARCH")),
                     distribution.spec = list(distribution = c("norm")),
                     switch.spec = list(do.mix = FALSE, K = 2))

#specification for markov switching model two regime, GJR, student
spec2grs <- CreateSpec(variance.spec = list(model = c("gjrGARCH")),
                     distribution.spec = list(distribution = c("std")),
                     switch.spec = list(do.mix = FALSE, K = 2))

####GARCH GJR one regime with the complete sample
#fit the model on the data with ML estimation: one regime, GARCH, gauss
set.seed(123)
fitggind <- FitML(specgg, data=indexes)
print(fitggind)

#fit the model on the data with ML estimation: one regime, GARCH, student
set.seed(123)
fitgsind <- FitML(specgs, data=indexes)
print(fitgsind)

#fit the model on the data with ML estimation: one regime, GJR, gauss
set.seed(123)
fitgrgind <- FitML(specgrg, data=indexes)
print(fitgrgind)

#fit the model on the data with ML estimation: one regime, GJR, student
set.seed(123)
fitgrsind <- FitML(specgrs, data=indexes)
print(fitgrsind)

####GARCH GJR two regime with the complete sample
#fit the model on the data with ML estimation: two regime, GARCH, gauss
set.seed(123)
fit2ggind <- FitML(spec2gg, data=indexes)
print(fit2ggind)

#fit the model on the data with ML estimation: two regime, GARCH, student
set.seed(123)
fit2gsind <- FitML(spec2gs, data=indexes)
print(fit2gsind)

#fit the model on the data with ML estimation: two regime, GJR, gauss
set.seed(123)
fit2grgind <- FitML(spec2grg, data=indexes)
print(fit2grgind)
fit2grgind$
#fit the model on the data with ML estimation: two regime, GJR, student
set.seed(123)
fit2grsind <- FitML(spec2grs, data=indexes)
print(fit2grsind)

######LR TESTS
#one regime GARCh: GAUSSIAN vs Student
LRg1gvs<-lrtest(fitggind,fitgsind)
print(LRg1gvs)

#one regime GJR: GAUSSIAN vs Student
LRgr1gvs<-lrtest(fitgrgind,fitgrsind)
print(LRgr1gvs)

#one regime GAUSSIAN: GARCH vs GJR
LRg1gvgr<-lrtest(fitggind,fitgrgind)
print(LRg1gvgr)

#one regime student-t: GARCh vs GJR
LRs1gvgr<-lrtest(fitgsind,fitgrsind)
print(LRs1gvgr)

####REgime-dependent correlations LR test
#two regime GARCh: GAUSSIAN vs Student
LRg2gvs<-lrtest(fit2ggind,fit2gsind)
print(LRg2gvs)

#two regime GJR: GAUSSIAN vs Student
LRgr2gvs<-lrtest(fit2grgind,fit2grsind)
print(LRgr2gvs)

#two regime GAUSSIAN: GARCH vs GJR
LRg2gvgr<-lrtest(fit2ggind,fit2grgind)
print(LRg2gvgr)

#two regime student-t: GARCh vs GJR
LRs2gvgr<-lrtest(fit2gsind,fit2grsind)
print(LRs2gvgr)

lrtest()
####GARCH GJR one regime adn two with the subsamples
#fit the model on the data with ML estimation: one regime, GARCH, gauss
set.seed(123)
fitggsp <- FitML(specgg, data=retsp)
print(fitggsp)
fitggdax <- FitML(specgg, data=retdax)
print(fitggdax)
fitggftse <- FitML(specgg, data=retftse)
print(fitggftse)

#fit the model on the data with ML estimation: one regime, GARCH, student
set.seed(123)
fitgssp <- FitML(specgs, data=retsp)
print(fitgssp)
fitgsdax <- FitML(specgs, data=retdax)
print(fitgsdax)
fitgsftse <- FitML(specgs, data=retftse)
print(fitgsftse)

#fit the model on the data with ML estimation: one regime, GJR, gauss
set.seed(123)
fitgrgsp <- FitML(specgrg, data=retsp)
print(fitgrgsp)
fitgrgdax <- FitML(specgrg, data=retdax)
print(fitgrgdax)
fitgrgftse <- FitML(specgrg, data=retftse)
print(fitgrgftse)

#fit the model on the data with ML estimation: one regime, GJR, student
set.seed(123)
fitgrssp <- FitML(specgrs, data=retsp)
print(fitgrssp)
fitgrsdax <- FitML(specgrs, data=retdax)
print(fitgrsdax)
fitgrsftse <- FitML(specgrs, data=retftse)
print(fitgrsftse)

#fit the model on the data with ML estimation: two regime, GARCH, gauss
set.seed(123)
fit2ggsp <- FitML(spec2gg, data=retsp)
print(fit2ggsp)
fit2ggdax <- FitML(spec2gg, data=retdax)
print(fit2ggdax)
fit2ggftse <- FitML(spec2gg, data=retftse)
print(fit2ggftse)
SR.fit <- ExtractStateFit(fit2ggftse)
print(SR.fit)

#fit the model on the data with ML estimation: two regime, GARCH, student
set.seed(123)
fit2gssp <- FitML(spec2gs, data=retsp)
print(fit2gssp)
fit2gsdax <- FitML(spec2gs, data=retdax)
print(fit2gsdax)
fit2gsftse <- FitML(spec2gs, data=retftse,ctr=(list(do.init=TRUE)))
print(fit2gsftse)

#fit the model on the data with ML estimation: two regime, GJR, gauss
set.seed(123)
fit2grgsp <- FitML(spec2grg, data=retsp)
print(fit2grgsp)
fit2grgdax <- FitML(spec2grg, data=retdax)
print(fit2grgdax)
fit2grgftse <- FitML(spec2grg, data=retftse)
print(fit2grgftse)

#fit the model on the data with ML estimation: two regime, GJR, student
set.seed(123)
fit2grssp <- FitML(spec2grs, data=retsp)
print(fit2grssp)
fit2grsdax <- FitML(spec2grs, data=retdax)
print(fit2grsdax)
fit2grsftse <- FitML(spec2grs, data=retftse,ctr=(list(do.init=TRUE)))
print(fit2grsftse)

####SOME PLOTS
##PLOTS FOR VARIANCES:
#VAriance for return of sp
ggplot( data = df, aes( Date, varsp )) + geom_line()
dev.copy(jpeg,filename="varretsp.jpg")
dev.off()
#VAriance for return of dax
ggplot( data = df, aes( Date, vardax )) + geom_line()
dev.copy(jpeg,filename="varretdax.jpg")
dev.off()
#VAriance for return of ftse
ggplot( data = df, aes( Date, varftse )) + geom_line()
dev.copy(jpeg,filename="varretsftse.jpg")
dev.off()

#PLOTS CORRELATIONS
#correlation between S&P and DAX
corspdaxind <- cor(retsp,retdax)
ggplot( data = df, aes( Date, corspdaxind )) + geom_line()
dev.copy(jpeg,filename="corretspdax.jpg")
dev.off()
#correlation between S&P and FTSE
corspftseind <- cor(retsp,retftse)
ggplot( data = df, aes( Date, corspftseind )) + geom_line()
dev.copy(jpeg,filename="corretspftse.jpg")
dev.off()
#correlation between FTSE and DAX
corftsedaxind <- cor(retftse,retdax)
ggplot( data = df, aes( Date, corftsedaxind )) + geom_line()
dev.copy(jpeg,filename="corretftsedax.jpg")
dev.off()

fit2ggind <- FitML(spec2gg, data=indexes)
print(fit2ggind)
stateind<-State(fit2ggind)
plot(stateind, type.prob = "smoothed")

ggplot( data = df, aes( Date, stateind )) + geom_line()
dev.copy(jpeg,filename="corretftsedax.jpg")
dev.off()
     