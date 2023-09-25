import os
from enum import IntEnum
from typing import Dict, Tuple

import joblib
import jquantsapi
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split


class Col(IntEnum):
    Y = 0
    CLOSE = 1
    PROFIT = 2
    LONG = 3
    SHORT = 4
    TOTAL = 5
    RETURN = 6
    Z = 7


class Ml(object):
    def __init__(
        self,
        *,
        outputs_dir_path: str = ".",
        data_dir_path: str = ".",
        model_dir_path: str = ".",
        mail_address: str = "",
        password: str = "",
        ticker: str = "",
        size: int = 1,
        window: int = 2,
    ) -> None:
        self.ticker = ticker
        self.outputs_dir_path = outputs_dir_path
        self.data_dir_path = data_dir_path
        self.model_dir_path = model_dir_path
        self.size = size
        self.window = window
        self.cli = jquantsapi.Client(mail_address=mail_address, password=password)
        os.makedirs(self.outputs_dir_path, exist_ok=True)
        os.makedirs(self.data_dir_path, exist_ok=True)
        os.makedirs(self.model_dir_path, exist_ok=True)
        self.X = pd.DataFrame()

    def features(self) -> None:
        pass

    def _get_prices_daily_quotes(self) -> pd.DataFrame:
        df = (
            self.cli.get_prices_daily_quotes(self.ticker)[
                [
                    "Date",
                    "AdjustmentOpen",
                    "AdjustmentHigh",
                    "AdjustmentLow",
                    "AdjustmentClose",
                    "AdjustmentVolume",
                ]
            ]
            .rename(
                columns={
                    "AdjustmentOpen": "Open",
                    "AdjustmentHigh": "High",
                    "AdjustmentLow": "Low",
                    "AdjustmentClose": "Close",
                    "AdjustmentVolume": "Volume",
                }
            )
            .set_index("Date")
        )
        df.to_csv(
            "{}/{}.d.csv".format(self.data_dir_path, self.ticker),
        )
        return df

    def backtest(self) -> Dict:
        self.df = self._get_prices_daily_quotes()
        self.features()
        profit = ((self.df.Close.shift(-1) - self.df.Close) * float(self.size)).round(2)
        y = np.where(
            profit > 0,
            1,
            -1,
        )
        model = lgb.LGBMClassifier()
        model.fit(self.X, y)
        joblib.dump(
            model,
            "{}/{}.joblib".format(self.model_dir_path, self.ticker),
            compress=True,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            y,
            random_state=0,
            shuffle=False,
        )
        model.fit(
            X_train,
            y_train,
        )
        y_pred = model.predict(X_test)
        X_test["Y"] = y_pred
        df = X_test[["Y"]].join(self.df.Close)
        df["Profit"] = profit
        df["Long"] = df["Short"] = df["Total"] = np.nan
        df["Return"] = (
            (self.df.Close.shift(-1) - self.df.Close) / self.df.Close * 100
        ).round(3)
        df["Z"] = y_test
        long = short = 0.0
        for i in range(len(df) - 1):
            if df.iat[i, Col.Y] > 0:
                long += df.iat[i, Col.PROFIT]
            else:
                short += df.iat[i, Col.PROFIT] * -1.0
            df.iat[i, Col.LONG] = long
            df.iat[i, Col.SHORT] = short
            df.iat[i, Col.TOTAL] = long + short
        fig = plt.figure(figsize=((6.4 * 2.5), (4.8 * 2.5)))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        ax1.plot(df["Long"])
        ax1.plot(df["Short"])
        ax1.plot(df["Total"])
        ax1.legend(["long profit", "short profit", "total profit"])
        ax2.plot(df["Close"])
        ax2.legend(["close"])
        feature_imp = pd.DataFrame(
            sorted(zip(model.feature_importances_, self.X.columns.tolist())),
            columns=["value", "features"],
        )
        sns.barplot(
            x="value",
            y="features",
            data=feature_imp.sort_values(by="value", ascending=False),
            ax=ax3,
        )
        cv = KFold(n_splits=2, shuffle=True, random_state=0)
        scores = cross_val_score(
            model,
            self.X,
            y,
            scoring="r2",
            cv=cv,
        )
        ax3.set_title(
            "score mean, std {} {}".format(
                np.mean(scores).round(4), np.std(scores).round(4)
            )
        )
        fig.suptitle(self.ticker)
        fig.tight_layout()
        plt.savefig("{}/{}.png".format(self.outputs_dir_path, self.ticker))
        plt.clf()
        plt.close()
        df.to_csv("{}/{}.csv".format(self.outputs_dir_path, self.ticker))
        _df = df[:-1]
        rd = {"long": {}, "short": {}, "total": {}}
        # long
        long_trades = _df.query("Y>0").count()["Y"]
        long_win_num = _df.query("Profit>0&Y>0").count()["Y"]
        long_loss_num = long_trades - long_win_num
        long_win = _df.query("Profit>0&Y>0").sum()["Profit"]
        long_loss = _df.query("Profit<0&Y>0").sum()["Profit"]
        long_average_win = long_win / long_win_num
        long_average_loss = abs(long_loss) / long_loss_num
        long_sharpe_ratio = (
            _df.query("Y>0")["Return"].mean() / _df.query("Y>0")["Return"].std()
        )
        long_mdd = (np.maximum.accumulate(_df["Long"]) - _df["Long"]).max()
        rd["long"]["profit"] = "{:.3f}".format(_df["Long"][-1])
        rd["long"]["trades"] = "{:.3f}".format(long_trades)
        rd["long"]["win rate"] = "{:.3f}".format(long_win_num / long_trades)
        rd["long"]["profit factor"] = "{:.3f}".format(long_win / abs(long_loss))
        rd["long"]["riskreward ratio"] = "{:.3f}".format(
            long_average_win / long_average_loss
        )
        rd["long"]["sharpe ratio"] = "{:.3f}".format(long_sharpe_ratio)
        rd["long"]["maximum drawdown"] = "{:.3f}".format(long_mdd)
        # short
        short_trades = _df.query("Y<0").count()["Y"]
        short_win_num = _df.query("Profit<0&Y<0").count()["Y"]
        short_loss_num = short_trades - short_win_num
        short_win = _df.query("Profit<0&Y<0").sum()["Profit"]
        short_loss = _df.query("Profit>0&Y<0").sum()["Profit"]
        short_average_win = abs(short_win) / short_win_num
        short_average_loss = abs(short_loss) / short_loss_num
        short_sharpe_ratio = (
            _df.query("Y<0")["Return"].mean() / _df.query("Y<0")["Return"].std()
        )
        short_mdd = (np.maximum.accumulate(_df["Short"]) - _df["Short"]).max()
        rd["short"]["profit"] = "{:.3f}".format(_df["Short"][-1])
        rd["short"]["trades"] = "{:.3f}".format(short_trades)
        rd["short"]["win rate"] = "{:.3f}".format(short_win_num / short_trades)
        rd["short"]["profit factor"] = "{:.3f}".format(abs(short_win) / abs(short_loss))
        rd["short"]["riskreward ratio"] = "{:.3f}".format(
            short_average_win / short_average_loss
        )
        rd["short"]["sharpe ratio"] = "{:.3f}".format(short_sharpe_ratio)
        rd["short"]["maximum drawdown"] = "{:.3f}".format(short_mdd)
        # total
        total_trades = _df.count()["Y"]
        total_win_num = _df.query("Profit<0&Y<0|Profit>0&Y>0").count()["Y"]
        total_loss_num = total_trades - total_win_num
        total_win = abs(_df.query("Profit<0&Y<0").sum()["Profit"]) + abs(
            _df.query("Profit>0&Y>0").sum()["Profit"]
        )
        total_loss = abs(_df.query("Profit>0&Y<0").sum()["Profit"]) + abs(
            _df.query("Profit<0&Y>0").sum()["Profit"]
        )
        total_average_win = total_win / total_win_num
        total_average_loss = total_loss / total_loss_num
        total_sharpe_ratio = _df["Return"].mean() / _df["Return"].std()
        total_mdd = (np.maximum.accumulate(_df["Total"]) - _df["Total"]).max()
        rd["total"]["profit"] = "{:.3f}".format(_df["Total"][-1])
        rd["total"]["trades"] = "{:.3f}".format(total_trades)
        rd["total"]["win rate"] = "{:.3f}".format(total_win_num / total_trades)
        rd["total"]["profit factor"] = "{:.3f}".format(total_win / total_loss)
        rd["total"]["riskreward ratio"] = "{:.3f}".format(
            total_average_win / total_average_loss
        )
        rd["total"]["sharpe ratio"] = "{:.3f}".format(total_sharpe_ratio)
        rd["total"]["maximum drawdown"] = "{:.3f}".format(total_mdd)
        return rd

    def sma(self, *, period: int) -> pd.DataFrame:
        return self.df.Close.rolling(period).mean()

    def ema(self, *, period: int) -> pd.DataFrame:
        return self.df.Close.ewm(span=period).mean()

    def bbands(
        self, *, period: int = 20, band: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        std = self.df.Close.rolling(period).std()
        mean = self.df.Close.rolling(period).mean()
        return mean + (std * band), mean, mean - (std * band)

    def macd(
        self,
        *,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        macd = (
            self.df.Close.ewm(span=fast_period).mean()
            - self.df.Close.ewm(span=slow_period).mean()
        )
        signal = macd.ewm(span=signal_period).mean()
        return macd, signal

    def stoch(
        self, *, k_period: int = 5, d_period: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        k = (
            (self.df.Close - self.df.Low.rolling(k_period).min())
            / (
                self.df.High.rolling(k_period).max()
                - self.df.Low.rolling(k_period).min()
            )
            * 100
        )
        d = k.rolling(d_period).mean()
        return k, d

    def rsi(self, *, period: int = 14) -> pd.DataFrame:
        return 100 - 100 / (
            1
            - self.df.Close.diff().clip(lower=0).rolling(period).mean()
            / self.df.Close.diff().clip(upper=0).rolling(period).mean()
        )

    def atr(self, *, period: int = 14) -> pd.DataFrame:
        a = (self.df.High - self.df.Low).abs()
        b = (self.df.High - self.df.Close.shift()).abs()
        c = (self.df.Low - self.df.Close.shift()).abs()

        df = pd.concat([a, b, c], axis=1).max(axis=1)
        return df.ewm(span=period).mean()

    def mom(self, *, period: int = 10) -> pd.DataFrame:
        return self.df.Close.diff(period)

    def predict(self) -> Dict:
        self.df = self._get_prices_daily_quotes()
        self.features()
        model = joblib.load(
            "{}/{}.joblib".format(self.model_dir_path, self.ticker),
        )
        y_pred = model.predict(self.X)
        self.X["Y"] = y_pred
        df = self.X[["Y"]].join(self.df.Close)
        return {
            "Date": df.index[-1].strftime("%Y-%m-%d"),
            "Price": (df.iloc[-1]["Close"]),
            "Sign": "long" if int(df.iloc[-1]["Y"]) > 0 else "short",
        }
