"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        tradeable_assets = self.price.columns[self.price.columns != self.exclude]
        prices = self.price[tradeable_assets]
        
        # 2. 計算動能 (Momentum)
        # 我們看過去 126 天 (約半年) 的累積漲幅
        # 用來判斷哪些板塊現在最強
        momentum = prices.pct_change(126)
        
        # 3. 排名 (Ranking)
        # rank(ascending=False) 代表漲最多的排第 1
        ranks = momentum.rank(axis=1, ascending=False)
        
        # 4. 篩選 Top 4 強勢股
        # 只有排名前 4 的板塊會被選中 (設為 1)，其他為 0
        top_n = 4
        signals = (ranks <= top_n).astype(float)
        
        # 5. 平均分配權重 (Equal Weight)
        # 被選中的 4 檔各分 25%
        # 為了防止分母為 0 (例如最開始幾天沒有動能數據)，我們用 sum + 1e-8
        weights = signals.div(signals.sum(axis=1) + 1e-8, axis=0)
        
        # 6. 填補缺值 (剛開始 126 天沒有動能數據)
        # 在沒有訊號時，預設平均持有所有資產 (Equal Weight All)
        num_assets = len(tradeable_assets)
        weights = weights.fillna(1.0 / num_assets)
        
        # 如果某天完全沒有訊號 (極少見)，也退回平均分配
        mask = weights.sum(axis=1) == 0
        weights.loc[mask] = 1.0 / num_assets
        
        # 7. 初始化權重表
        self.portfolio_weights = pd.DataFrame(
            0.0, index=self.price.index, columns=self.price.columns
        )
        
        # 8. 避免前視偏差 (Look-ahead Bias) 與 填入
        # 今天的訊號決定明天的權重，所以要 shift(1)
        self.portfolio_weights[tradeable_assets] = weights.shift(1)
        
        # 9. 處理第一天的 NaN (shift 產生的) 和排除資產
        self.portfolio_weights.fillna(0, inplace=True)
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights[self.exclude] = 0.0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)