library(DBI)
con <- dbConnect(
  odbc::odbc(),
  dsn = 'databricks',
  HTTPPath = 'sql/protocolv1/o/847064027862604/0622-162121-dts9kxvy',
  PWD = rstudioapi::askForPassword('Please enter Databricks PAT')
)
library(data.table)
library(tidyverse)
library(rstpm2)
library(ggplot2)

#----------------------------------------------------#
# Read File
#----------------------------------------------------#
dfm <-  readRDS('~/collab/CCU043/data/ccu043_01_survival_cohort_m.rds')
summary(dfm$followup_time_t1)
summary(dfm$bmi_value)
dfm %>%
  group_by(exposed) %>% 
  summarise(n = n())
dfm %>%
  group_by(exposed, outcome_t1) %>% 
  summarise(n = n())

tim_seq <- seq(0.1, 4.4, by = 0.1)  
age_seq <- seq(40, 70,   by = 10)
exp_seq <- seq(0, 1,     by = 1)

#----------------------------------------------------#
# Age + BMI as continuous confounder  
#----------------------------------------------------#
m_bmi    <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4) 
                  + nsx(bmi_value, df = 4)*exposed,
                  data = dfm, k = 4)
m_bmitvc <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4)
                  + nsx(bmi_value, df = 4)*exposed,
                  data = dfm, k = 4, tvc = list(exposed = 1))

anova(m_bmi, m_bmitvc)
print(BIC(m_bmi, m_bmitvc))       

bmi_seq <- seq(20, 35,   by = 5)
pre_bmi <- expand.grid(followup_time_t1 = tim_seq, age = age_seq,
                       bmi_value = bmi_seq, exposed = exp_seq)
pre_bmi$bmiv    <- as.factor(pre_bmi$bmi_value)
pre_bmi$expv    <- as.factor(pre_bmi$exposed)

m_hf_bmi         <- predict(m_bmi, newdata = pre_bmi, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_bmi$rbmi    <- m_hf_bmi$Estimate * 1000  
m_hf_bmi$rbmi_lb <- m_hf_bmi$lower    * 1000
m_hf_bmi$rbmi_ub <- m_hf_bmi$upper    * 1000
fwrite(m_hf_bmi,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_bmi_male.csv')
m_hf_bmi <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_bmi_male.csv',
                     header = TRUE, sep = ",")
m_hf_bmi$bmiv    <- as.factor(m_hf_bmi$bmi_value)
m_hf_bmi$expv    <- as.factor(m_hf_bmi$exposed)
ggplot(m_hf_bmi, 
       aes(x = followup_time_t1, y = rbmi)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = bmiv, linetype = expv)) +
  scale_y_log10() 

m_hf_bmi         <- predict(m_bmitvc, newdata = pre_bmi, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_bmi$rbmi    <- m_hf_bmi$Estimate * 1000  
m_hf_bmi$rbmi_lb <- m_hf_bmi$lower    * 1000
m_hf_bmi$rbmi_ub <- m_hf_bmi$upper    * 1000
fwrite(m_hf_bmi,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_bmitvc_male.csv')
m_hf_bmi <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_bmitvc_male.csv',
                     header = TRUE, sep = ",")
m_hf_bmi$bmiv    <- as.factor(m_hf_bmi$bmi_value)
m_hf_bmi$expv    <- as.factor(m_hf_bmi$exposed)
ggplot(m_hf_bmi, 
       aes(x = followup_time_t1, y = rbmi)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = bmiv, linetype = expv)) +
  scale_y_log10() 

#----------------------------------------------------#
# Age + IMD as categorical confounder  
#----------------------------------------------------#
dfm$imdv <- as.factor(dfm$imd)
m_imd    <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4) 
                  + imdv*exposed, data = dfm, k = 4)
m_imdtvc <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4)
                  + imdv*exposed, data = dfm, k = 4, tvc = list(exposed = 1))
anova(m_imd, m_imdtvc)
print(BIC(m_imd, m_imdtvc))       

imd_seq <- seq(1, 5, by = 1)
pre_imd <- expand.grid(followup_time_t1 = tim_seq, age = age_seq,
                       imd = imd_seq, exposed = exp_seq)
pre_imd$imdv <- as.factor(pre_imd$imd)
pre_imd$expv <- as.factor(pre_imd$exposed)

m_hf_imd <- predict(m_imd, newdata = pre_imd, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_imd$rimd    <- m_hf_imd$Estimate * 1000  #To transform the rate to 1000 pyrs
m_hf_imd$rimd_lb <- m_hf_imd$lower    * 1000
m_hf_imd$rimd_ub <- m_hf_imd$upper    * 1000
fwrite(m_hf_imd,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_imd_male.csv')
m_hf_imd <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_imd_male.csv',
                     header = TRUE, sep = ",")
m_hf_imd$imdv    <- as.factor(m_hf_imd$imd)
m_hf_imd$expv    <- as.factor(m_hf_imd$exposed)
ggplot(m_hf_imd, 
       aes(x = followup_time_t1, y = rimd)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = imdv, linetype = expv)) +
  scale_y_log10() 

m_hf_imd <- predict(m_imdtvc, newdata = pre_imd, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_imd$rimd    <- m_hf_imd$Estimate * 1000  #To transform the rate to 1000 pyrs
m_hf_imd$rimd_lb <- m_hf_imd$lower    * 1000
m_hf_imd$rimd_ub <- m_hf_imd$upper    * 1000
fwrite(m_hf_imd,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_imdtvc_male.csv')
m_hf_imd <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_imdtvc_male.csv',
                     header = TRUE, sep = ",")
m_hf_imd$imdv    <- as.factor(m_hf_imd$imd)
m_hf_imd$expv    <- as.factor(m_hf_imd$exposed)
ggplot(m_hf_imd, 
       aes(x = followup_time_t1, y = rimd)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = imdv, linetype = expv)) +
  scale_y_log10() 

#----------------------------------------------------#
# Age + Ethnicity as categorical confounder  
#----------------------------------------------------#
dfm$ethv <- as.factor(dfm$eth)
levels(dfm$ethv)
m_eth    <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4) 
                  + ethv*exposed, data = dfm, k = 4)
m_ethtvc <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4)
                  + ethv*exposed, data = dfm, k = 4, tvc = list(exposed = 1))
anova(m_eth, m_ethtvc)
print(BIC(m_eth, m_ethtvc))       

eth_seq <- c("0.White","1.Asian or Asian British","2.Black or Black British",
               "3.Mixed/Other", "4.Unknown")
pre_eth <- expand.grid(followup_time_t1 = tim_seq, age = age_seq,
                       eth = eth_seq, exposed = exp_seq)
pre_eth$ethv <- as.factor(pre_eth$eth)
pre_eth$expv <- as.factor(pre_eth$exposed)
levels(pre_eth$ethv)

m_hf_eth <- predict(m_eth, newdata = pre_eth, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_eth$reth    <- m_hf_eth$Estimate * 1000  
m_hf_eth$reth_lb <- m_hf_eth$lower    * 1000
m_hf_eth$reth_ub <- m_hf_eth$upper    * 1000
fwrite(m_hf_eth,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_eth_male.csv')
m_hf_eth <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_eth_male.csv',
                     header = TRUE, sep = ",")
m_hf_eth$ethv    <- as.factor(m_hf_eth$eth)
m_hf_eth$expv    <- as.factor(m_hf_eth$exposed)
ggplot(m_hf_eth, 
       aes(x = followup_time_t1, y = reth)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = ethv, linetype = expv)) +
  scale_y_log10() 


m_hf_eth <- predict(m_ethtvc, newdata = pre_eth, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_eth$reth    <- m_hf_eth$Estimate * 1000  
m_hf_eth$reth_lb <- m_hf_eth$lower    * 1000
m_hf_eth$reth_ub <- m_hf_eth$upper    * 1000
fwrite(m_hf_eth,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_ethtvc_male.csv')
m_hf_eth <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_ethtvc_male.csv',
                     header = TRUE, sep = ",")
m_hf_eth$ethv    <- as.factor(m_hf_eth$eth)
m_hf_eth$expv    <- as.factor(m_hf_eth$exposed)
ggplot(m_hf_eth, 
       aes(x = followup_time_t1, y = reth)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = ethv, linetype = expv)) +
  scale_y_log10() 

#----------------------------------------------------#
# Age + Region as categorical confounder  
#----------------------------------------------------#
dfm$regv <- as.factor(dfm$region)
levels(dfm$regv)

m_reg    <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4) 
                  + regv*exposed, data = dfm, k = 4)
m_regtvc <- stpm2(Surv(followup_time_t1, outcome_t1) ~ nsx(age, df = 4) 
                  + regv*exposed, data = dfm, k = 4, tvc = list(exposed = 1))
anova(m_reg, m_regtvc)
print(BIC(m_reg, m_regtvc))       

reg_seq <- c("East Midlands","East of England","London","North East",
             "North West","South East","South West","West Midlands",
             "Yorkshire and The Humber")
pre_reg <- expand.grid(followup_time_t1 = tim_seq, age = age_seq,
                       reg = reg_seq, exposed = exp_seq)
pre_reg$regv <- as.factor(pre_reg$reg)
pre_reg$expv <- as.factor(pre_reg$exposed)

m_hf_reg <- predict(m_reg, newdata = pre_reg, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_reg$rreg    <- m_hf_reg$Estimate * 1000  #To transform the rate to 1000 pyrs
m_hf_reg$rreg_lb <- m_hf_reg$lower    * 1000
m_hf_reg$rreg_ub <- m_hf_reg$upper    * 1000
fwrite(m_hf_reg,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_reg_male.csv')
m_hf_reg <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_reg_male.csv',
                     header = TRUE, sep = ",")
m_hf_reg$regv    <- as.factor(m_hf_reg$reg)
m_hf_reg$expv    <- as.factor(m_hf_reg$exposed)
ggplot(m_hf_reg, 
       aes(x = followup_time_t1, y = rreg)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = regv, linetype = expv)) +
  scale_y_log10() 

m_hf_reg <- predict(m_regtvc, newdata = pre_reg, type = "hazard", se.fit=TRUE, full=TRUE)
m_hf_reg$rreg    <- m_hf_reg$Estimate * 1000  #To transform the rate to 1000 pyrs
m_hf_reg$rreg_lb <- m_hf_reg$lower    * 1000
m_hf_reg$rreg_ub <- m_hf_reg$upper    * 1000
fwrite(m_hf_reg,'~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_regtvc_male.csv')
m_hf_reg <- read.csv('~/collab/CCU043/output/CCU043_01_t1_predicted_hazard_regtvc_male.csv',
                     header = TRUE, sep = ",")
m_hf_reg$regv    <- as.factor(m_hf_reg$reg)
m_hf_reg$expv    <- as.factor(m_hf_reg$exposed)
ggplot(m_hf_reg, 
       aes(x = followup_time_t1, y = rreg)) +
  facet_grid(~age) +
  xlab("Time (years)") +
  ylab("Predicted Hazard Rate (per 1000 pyrs)") +
  geom_line(aes(color = regv, linetype = expv)) +
  scale_y_log10() 
