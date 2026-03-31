# ==============================================================
# ETF를 활용한 글로벌 대체투자 자산들의 동적 연결성과
# 포트폴리오 최적화에 관한 연구 (윤병조, 2024)
#
# R ConnectednessApproach 패키지 구현
# ==============================================================

# install.packages("ConnectednessApproach")
# install.packages("quantmod")
library(ConnectednessApproach)
library(quantmod)
library(zoo)

# ============================================================
# 1. 데이터 다운로드
# ============================================================
cat("=" , rep("=", 69), "\n", sep="")
cat("[Step 1] Downloading ETF data...\n")

tickers <- c("USO", "VNQ", "IGF", "ICLN", "SPY")
asset_names <- c("Crude Oil", "Real Estate", "Infrastructure", "Clean Energy", "Equity")
start_date <- "2011-01-02"
end_date <- "2023-09-30"

prices <- NULL
for (tk in tickers) {
  getSymbols(tk, src = "yahoo", from = start_date, to = end_date, auto.assign = TRUE)
  adj_close <- Ad(get(tk))
  if (is.null(prices)) {
    prices <- adj_close
  } else {
    prices <- merge(prices, adj_close)
  }
}
colnames(prices) <- tickers
prices <- na.omit(prices)

cat(sprintf("  Period: %s ~ %s\n", index(prices)[1], tail(index(prices), 1)))
cat(sprintf("  Observations: %d\n", nrow(prices)))

# ============================================================
# 2. 로그 수익률 → 22일 롤링 변동성
# ============================================================
cat("\n[Step 2] Computing log returns & volatility...\n")

returns <- diff(log(prices)) * 100  # 퍼센트 단위 로그 수익률
returns <- na.omit(returns)

# 22일 롤링 표준편차 (변동성)
vol_window <- 22
volatility <- rollapply(returns, width = vol_window, FUN = sd, by.column = TRUE, align = "right")
volatility <- na.omit(volatility)

cat(sprintf("  Volatility window: %d days\n", vol_window))
cat(sprintf("  Volatility observations: %d\n", nrow(volatility)))

# ============================================================
# 3. TVP-VAR 동적 연결성 분석
# ============================================================
cat("\n", rep("=", 70), "\n", sep="")
cat("[Step 3] TVP-VAR Dynamic Connectedness Analysis\n")
cat(rep("=", 70), "\n", sep="")

# ConnectednessApproach 패키지로 TVP-VAR 연결성 분석
# 논문과 동일한 설정: nlag=1, nfore=10, kappa1=0.99, kappa2=0.96, BayesPrior
dca <- ConnectednessApproach(
  as.zoo(volatility),
  model = "TVP-VAR",
  connectedness = "Time",
  nlag = 1,
  nfore = 10,
  VAR_config = list(
    TVPVAR = list(
      kappa1 = 0.99,
      kappa2 = 0.96,
      prior = "BayesPrior",
      gamma = 0.01
    )
  )
)

# 평균 연결성 테이블 (TABLE은 문자형 → 숫자 변환, 텍스트 행 제외)
cat("\n--- Average Connectedness Table ---\n")
n <- length(tickers)
table_core <- dca$TABLE[1:n, 1:n]
table_num <- matrix(as.numeric(table_core), nrow = n, dimnames = dimnames(table_core))
print(round(table_num, 2))

# FROM/TO/NET 출력
cat("\nFROM: ", dca$TABLE[1:n, n+1], "\n")
if (nrow(dca$TABLE) > n) {
  cat("TO:   ", dca$TABLE[n+1, 1:n], "\n")
  cat("NET:  ", dca$TABLE[n+2, 1:n], "\n")
}

# 평균 TCI
avg_tci <- mean(dca$TCI)
cat(sprintf("\nAverage TCI: %.2f%%\n", avg_tci))
cat(sprintf("논문 보고치: 36.8%%\n"))

# ============================================================
# 4. TVP-VAR 별도 추정 (공분산 Q 추출용)
# ============================================================
cat("\n", rep("=", 70), "\n", sep="")
cat("[Step 4] TVP-VAR Estimation for Hedge Ratios\n")
cat(rep("=", 70), "\n", sep="")

# TVP-VAR 모형 직접 추정하여 Q (시변 공분산) 추출
bp <- BayesPrior(as.zoo(volatility), nlag = 1)
tvpvar_model <- TVPVAR(as.zoo(volatility),
                       configuration = list(
                         l = c(0.99, 0.96),
                         nlag = 1,
                         prior = bp
                       ))

# 헤지비율 (Kroner & Sultan, 1993)
hr <- HedgeRatio(as.zoo(volatility), tvpvar_model$Q)
cat("\n--- Average Hedge Ratios ---\n")
hr_table <- matrix(as.numeric(hr$TABLE[1:n, 1:n]), nrow = n,
                   dimnames = list(tickers, tickers))
print(round(hr_table, 4))

# ============================================================
# 5. 포트폴리오 최적화
# ============================================================
cat("\n", rep("=", 70), "\n", sep="")
cat("[Step 5] Portfolio Optimization\n")
cat(rep("=", 70), "\n", sep="")

# MVP 함수 (직접 구현): min w'Σw s.t. w'1=1, w>=0
calc_mvp <- function(Sigma) {
  n <- nrow(Sigma)
  Sigma_inv <- tryCatch(solve(Sigma + diag(n) * 1e-8), error = function(e) diag(n))
  w <- Sigma_inv %*% rep(1, n)
  w <- as.numeric(w / sum(w))
  w <- pmax(w, 0)
  w / sum(w)
}

# MCP 함수 (직접 구현): 상관행렬의 MVP → 변동성 역수로 스케일링
calc_mcp <- function(Sigma) {
  n <- nrow(Sigma)
  sigma_vec <- sqrt(pmax(diag(Sigma), 1e-10))
  D_inv <- diag(1 / sigma_vec)
  R <- D_inv %*% Sigma %*% D_inv
  w_R <- calc_mvp(R)
  w <- D_inv %*% w_R
  w <- as.numeric(w / sum(w))
  w <- pmax(w, 0)
  w / sum(w)
}

# 동적 MVP/MCP 계산
Q <- tvpvar_model$Q
T_q <- dim(Q)[3]
w_mvp <- matrix(0, T_q, n)
w_mcp <- matrix(0, T_q, n)
colnames(w_mvp) <- colnames(w_mcp) <- tickers

for (t in 1:T_q) {
  Sigma_t <- Q[,,t]
  w_mvp[t,] <- calc_mvp(Sigma_t)
  w_mcp[t,] <- calc_mcp(Sigma_t)
}

cat("\n--- Minimum Variance Portfolio (MVP) ---\n")
cat("Average weights:\n")
print(round(colMeans(w_mvp), 4))

cat("\n--- Minimum Correlation Portfolio (MCP) ---\n")
cat("Average weights:\n")
print(round(colMeans(w_mcp), 4))

# 최소연결성 포트폴리오 (MCoP)
# returns와 PCI 길이 맞추기 (PCI는 변동성 기반이라 더 짧음)
T_pci <- dim(dca$PCI)[3]
returns_aligned_mcop <- tail(as.zoo(returns) / 100, T_pci)
mcop <- MinimumConnectednessPortfolio(
  returns_aligned_mcop,
  dca$PCI,
  statistics = "Fisher"
)
cat("\n--- Minimum Connectedness Portfolio (MCoP) ---\n")
cat("Average weights:\n")
print(round(colMeans(mcop$Weights), 4))

# ============================================================
# 6. 포트폴리오 성과 비교
# ============================================================
cat("\n", rep("=", 70), "\n", sep="")
cat("[Step 6] Portfolio Performance Comparison\n")
cat(rep("=", 70), "\n", sep="")

portfolio_stats <- function(r, name) {
  data.frame(
    Portfolio = name,
    Mean = mean(r) * 100,
    StdDev = sd(r) * 100,
    Sharpe = mean(r) / sd(r),
    Min = min(r) * 100,
    Max = max(r) * 100
  )
}

# Equal Weight
T_port <- min(nrow(w_mvp), nrow(returns))
ret_aligned <- tail(as.matrix(returns) / 100, T_port)

r_equal <- rowSums(ret_aligned * (1/n))
r_mvp <- rowSums(ret_aligned * tail(w_mvp, T_port))
r_mcp <- rowSums(ret_aligned * tail(w_mcp, T_port))
r_mcop <- rowSums(ret_aligned * tail(mcop$Weights, T_port))

perf <- rbind(
  portfolio_stats(r_equal, "Equal Weight (1/N)"),
  portfolio_stats(r_mvp, "MVP"),
  portfolio_stats(r_mcp, "MCP"),
  portfolio_stats(r_mcop, "MCoP")
)
print(perf)

# ============================================================
# 7. 시각화
# ============================================================
cat("\n", rep("=", 70), "\n", sep="")
cat("[Step 7] Generating Figures\n")
cat(rep("=", 70), "\n", sep="")

dir.create("figures_r", showWarnings = FALSE)

# TCI 시계열
png("figures_r/tci_dynamic.png", width = 1400, height = 500, res = 150)
plot(dca$TCI, type = "l", col = "navy", lwd = 1.2,
     main = "Dynamic Total Connectedness Index (TCI)",
     ylab = "TCI (%)", xlab = "")
abline(h = mean(dca$TCI), col = "red", lty = 2)
legend("topright", legend = sprintf("Mean: %.1f%%", mean(dca$TCI)),
       col = "red", lty = 2, cex = 0.8)
dev.off()
cat("Saved: figures_r/tci_dynamic.png\n")

# NET 연결성
png("figures_r/net_connectedness.png", width = 1400, height = 800, res = 150)
par(mfrow = c(n, 1), mar = c(2, 4, 2, 1))
for (i in 1:n) {
  net_i <- dca$NET[, i]
  plot(net_i, type = "h", col = ifelse(net_i >= 0, "steelblue", "gray60"),
       main = tickers[i], ylab = "NET (%)", xlab = "")
  abline(h = 0, col = "black", lwd = 0.5)
}
dev.off()
cat("Saved: figures_r/net_connectedness.png\n")

# GFEVD 히트맵
png("figures_r/gfevd_heatmap.png", width = 800, height = 600, res = 150)
avg_gfevd <- dca$TABLE[1:n, 1:n]
heatmap(as.matrix(avg_gfevd), Rowv = NA, Colv = NA,
        col = heat.colors(256), scale = "none",
        main = "Average Connectedness Table (GFEVD, %)",
        margins = c(6, 6))
dev.off()
cat("Saved: figures_r/gfevd_heatmap.png\n")

# 누적 수익률
png("figures_r/cumulative_returns.png", width = 1400, height = 600, res = 150)
cum_equal <- cumsum(r_equal) * 100
cum_mvp <- cumsum(r_mvp) * 100
cum_mcp <- cumsum(r_mcp) * 100
cum_mcop <- cumsum(r_mcop) * 100
ylim <- range(c(cum_equal, cum_mvp, cum_mcp, cum_mcop))
plot(cum_equal, type = "l", col = "gray50", lty = 2, lwd = 1.2,
     ylim = ylim, main = "Cumulative Portfolio Returns",
     ylab = "Cumulative Return (%)", xlab = "")
lines(cum_mvp, col = "blue", lwd = 1.2)
lines(cum_mcp, col = "orange", lwd = 1.2)
lines(cum_mcop, col = "green4", lwd = 1.2)
abline(h = 0, col = "black", lwd = 0.3)
legend("topleft", legend = c("Equal (1/N)", "MVP", "MCP", "MCoP"),
       col = c("gray50", "blue", "orange", "green4"),
       lty = c(2, 1, 1, 1), lwd = 1.2, cex = 0.8)
dev.off()
cat("Saved: figures_r/cumulative_returns.png\n")

# ============================================================
# 결과 저장
# ============================================================
dir.create("results_r", showWarnings = FALSE)
write.csv(dca$TABLE, "results_r/connectedness_table.csv")
write.csv(hr$TABLE, "results_r/hedge_ratios.csv")
write.csv(perf, "results_r/portfolio_performance.csv", row.names = FALSE)

cat("\n", rep("=", 70), "\n", sep="")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 70), "\n", sep="")
cat(sprintf("\n  Average TCI: %.2f%% (논문: 36.8%%)\n", avg_tci))

net_means <- colMeans(dca$NET)
cat(sprintf("  순전출자: %s (NET=%.2f%%)\n", tickers[which.max(net_means)], max(net_means)))
cat(sprintf("  순유입자: %s (NET=%.2f%%)\n", tickers[which.min(net_means)], min(net_means)))
cat("\n  Output: results_r/, figures_r/\n")
