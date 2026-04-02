#install.packages("tseries")
#install.packages("forecast")
#install.packages("vars")
#install.packages("earth")

library(tseries)
library(forecast)
library(vars)
library(earth)

############################################################
################ unified TS by electricity #################
############################################################

electricity <- read.csv("electricity.csv", header=TRUE)
electricity <- electricity[1:260, -c(1)]   # remove 1st column Date

Demand <- ts(electricity$TotalDemand, start=c(2001,1), freq=12)
IPI <- ts(electricity$IPI, start=c(2001,1), freq=12)
CPI <- ts(electricity$CPI, start=c(2001,1), freq=12)
temperature <- ts(electricity$AverageTemp, start=c(2001,1), freq=12)
export <- ts(electricity$ExportOrderAmount, start=c(2001,1), freq=12)
season <- ts(electricity$SeasonIndex, start=c(2001,1), freq=12)

xreg_demand <- cbind(IPI, CPI, temperature, export)
xdemand.df <- data.frame(Demand, IPI, CPI, temperature, export, season)

############################################################
###################### comparison table ####################
############################################################
table <- array(0, c(3,7))
rownames(table) <- c("RMSE","MAE","MAPE")
colnames(table) <- c("H-W additive","SARIMA","MLR","MARS","VAR","ARIMAX","Lagged ARIMAX")

############################################################
#################### stationarity check ####################
############################################################
adf.test(Demand)
pp.test(Demand)
kpss.test(Demand)
ndiffs(Demand)

dif1_demand <- diff(Demand)
adf.test(dif1_demand)
pp.test(dif1_demand)
kpss.test(dif1_demand)
ndiffs(dif1_demand)

############################################################
#################### 1. Holt-Winters #######################
############################################################
fity11 <- HoltWinters(Demand, seasonal="additive")
hw_pred <- ts(fity11$fitted[,1], start=start(fity11$fitted[,1]), freq=12)

Demand_hw <- window(Demand, start=start(hw_pred), end=end(hw_pred))

table[1,1] <- sqrt(mean((Demand_hw - hw_pred)^2))
table[2,1] <- mean(abs(Demand_hw - hw_pred))
table[3,1] <- mean(abs((Demand_hw - hw_pred) / Demand_hw))

############################################################
######################## 2. SARIMA #########################
############################################################
auto.arima(Demand, max.order=10, trace=TRUE, ic="aic",
           approximation=FALSE, stepwise=FALSE)

demand_TS <- Arima(Demand,
                   order=c(5,0,2),
                   seasonal=list(order=c(0,1,1), period=12),
                   include.drift=TRUE)
print(demand_TS)

sarima_resid <- residuals(demand_TS)
Demand_sarima <- window(Demand, start=start(sarima_resid), end=end(sarima_resid))

table[1,2] <- sqrt(mean((sarima_resid)^2))
table[2,2] <- mean(abs(sarima_resid))
table[3,2] <- mean(abs(sarima_resid / Demand_sarima))

fore_demand <- forecast(demand_TS, level=95, h=12)
fore_demand

sarima_pred <- ts(fitted(demand_TS),
                  start=start(fitted(demand_TS)),
                  freq=frequency(Demand))

############################################################
########################## 3. MLR ##########################
############################################################
mlr.demand <- lm(TotalDemand ~ ., data = electricity)
summary(mlr.demand)

mlr_pred <- ts(mlr.demand$fitted.values, start=c(2001,1), freq=12)

table[1,3] <- sqrt(mean((electricity$TotalDemand - mlr.demand$fitted.values)^2))
table[2,3] <- mean(abs(electricity$TotalDemand - mlr.demand$fitted.values))
table[3,3] <- mean(abs((electricity$TotalDemand - mlr.demand$fitted.values) / electricity$TotalDemand))

############################################################
########################## 4. MARS #########################
############################################################
mars.demand <- earth(TotalDemand ~ ., degree=3, trace=2, data=electricity)
summary(mars.demand)
evimp(mars.demand, trim=FALSE)

mars_pred <- ts(mars.demand$fitted.values, start=c(2001,1), freq=12)

table[1,4] <- sqrt(mean((electricity$TotalDemand - mars.demand$fitted.values)^2))
table[2,4] <- mean(abs(electricity$TotalDemand - mars.demand$fitted.values))
table[3,4] <- mean(abs((electricity$TotalDemand - mars.demand$fitted.values) / electricity$TotalDemand))

############################################################
########################### 5. VAR #########################
############################################################
VARselect(xdemand.df, lag.max=3, type="both")$criteria[1,]
VARselect(xdemand.df, lag.max=3, type="none")$criteria[1,]
VARselect(xdemand.df, lag.max=3, type="const")$criteria[1,]
VARselect(xdemand.df, lag.max=3, type="trend")$criteria[1,]

demand.lag <- 3   # lag 3 + both

var.demand <- VAR(xdemand.df, p=demand.lag, type="both")
summary(var.demand)

var_predall <- data.frame(fitted(var.demand))
var_pred <- ts(var_predall$Demand, start=c(2001,4), freq=12)

stability.demand <- stability(var.demand, type="OLS-CUSUM", h=0.15,
                              dynamic=FALSE, rescale=TRUE)
plot(stability.demand)

demand.pre <- predict(var.demand, n.ahead=12, ci=0.95)
plot(demand.pre, xlab="month", ylab="electricity (KWH)")

resid.demand <- as.data.frame(resid(var.demand))$Demand
table[1,5] <- sqrt(mean((resid.demand)^2))
table[2,5] <- mean(abs(resid.demand))
table[3,5] <- mean(abs(resid.demand / Demand[(demand.lag+1):length(Demand)]))

############################################################
######################## 6. ARIMAX #########################
############################################################
arimax.demand <- auto.arima(Demand,
                            xreg = xreg_demand,
                            max.order = 10,
                            trace = TRUE,
                            ic = "aic",
                            approximation = FALSE,
                            stepwise = FALSE)

summary(arimax.demand)

Box.test(arimax.demand$residuals, lag = 12, type = "Ljung-Box")
tsdiag(arimax.demand)

arimax_resid <- residuals(arimax.demand)
Demand_arimax <- window(Demand,
                        start = start(arimax_resid),
                        end = end(arimax_resid))

table[1,6] <- sqrt(mean((arimax_resid)^2))
table[2,6] <- mean(abs(arimax_resid))
table[3,6] <- mean(abs(arimax_resid / Demand_arimax))

arimax_pred <- ts(fitted(arimax.demand),
                  start = start(fitted(arimax.demand)),
                  freq = frequency(Demand))

############################################################
###################### 7. Lagged ARIMAX ####################
############################################################
best_aic <- Inf
best_ipi <- 0
best_cpi <- 0
best_temp <- 0
best_export <- 0

for (i in 0:4)
{
  for (j in 0:4)
  {
    for (k in 0:4)
    {
      for (m in 0:4)
      {
        max_lag <- max(i,j,k,m)
        start_idx <- 1 + max_lag
        
        xreg_lag <- cbind(
          IPI[(start_idx-i):(length(IPI)-i)],
          CPI[(start_idx-j):(length(CPI)-j)],
          temperature[(start_idx-k):(length(temperature)-k)],
          export[(start_idx-m):(length(export)-m)]
        )
        
        fit <- auto.arima(Demand[start_idx:length(Demand)],
                          xreg = xreg_lag,
                          max.order = 10,
                          ic = "aic",
                          trace = FALSE,
                          approximation = FALSE,
                          stepwise = FALSE)
        
        if (fit$aic < best_aic)
        {
          best_aic <- fit$aic
          best_ipi <- i
          best_cpi <- j
          best_temp <- k
          best_export <- m
        }
      }
    }
  }
}

cat("IPI lag:", best_ipi, "\n")
cat("CPI lag:", best_cpi, "\n")
cat("Temp lag:", best_temp, "\n")
cat("Export lag:", best_export, "\n")
cat("Best AIC:", best_aic, "\n")

lag_max <- max(best_ipi, best_cpi, best_temp, best_export)
start_idx <- 1 + lag_max

xreg_best <- cbind(
  IPI[(start_idx-best_ipi):(length(IPI)-best_ipi)],
  CPI[(start_idx-best_cpi):(length(CPI)-best_cpi)],
  temperature[(start_idx-best_temp):(length(temperature)-best_temp)],
  export[(start_idx-best_export):(length(export)-best_export)]
)

lag_arimax <- auto.arima(Demand[start_idx:length(Demand)],
                         xreg = xreg_best,
                         max.order = 10,
                         trace = TRUE,
                         ic = "aic",
                         approximation = FALSE,
                         stepwise = FALSE)

summary(lag_arimax)

Box.test(lag_arimax$residuals, lag = 12, type = "Ljung-Box")
tsdiag(lag_arimax)

resid_lag <- residuals(lag_arimax)
Demand_lag <- Demand[start_idx:length(Demand)]

table[1,7] <- sqrt(mean((resid_lag)^2))
table[2,7] <- mean(abs(resid_lag))
table[3,7] <- mean(abs(resid_lag / Demand_lag))

lag_arimax_pred <- ts(fitted(lag_arimax),
                      start = start(fitted(lag_arimax)),
                      freq = frequency(Demand))

############################################################
######################## result table ######################
############################################################
table

############################################################
######################## fitted plot #######################
############################################################
plot(Demand, type="l", lty=1, col="black", ylab="Electricity", xlab="Month")
lines(hw_pred, type="l", lty=1, col="purple")
lines(sarima_pred, type="l", lty=1, col="red")
lines(mlr_pred, type="l", lty=1, col="green")
lines(mars_pred, type="l", lty=1, col="blue")
lines(var_pred, type="l", lty=1, col="orange")
lines(arimax_pred, type="l", lty=1, col="brown")
lines(lag_arimax_pred, type="l", lty=1, col="pink")
title("Forecasts for Electricity Consumption")
legend("bottomright",
       legend=c("Actual","H-W additive","SARIMA","MLR","MARS","VAR","ARIMAX","Lagged ARIMAX"),
       fill=c("black","purple","red","green","blue","orange","brown","pink"),
       cex=0.8,
       bty="n",
       y.intersp=0.9,
       x.intersp=0.8)