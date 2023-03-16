library(rptR)
library(lme4)
library(glmmTMB)
library(mgcv)

data <- read.csv("data_step_first_day.csv", header=TRUE, sep=",")
data[,4] <- data[, 4] + 0.00001
head(data)
rep1 <- rpt(step ~ (1 | level_0), grname = "level_0", data = data, datatype = "Poisson", nboot = 0, npermut = 0)
print(rep1)
ba_model = bam(
  #Reaction ~  Days + s(Subject, bs='re') + s(Days, Subject, bs='re'), 
  step ~ (1 | level_0),
  data = data
)
print(ba_model)
#rep2 <- glmer(step ~ (1 | level_0), family=Gamma(link="log"), data=data)
#print(rep2)
