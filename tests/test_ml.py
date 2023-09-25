import pandas as pd
import pytest

from jquants_ml import Ml


@pytest.fixture(scope="module", autouse=True)
def scope_module():
    class MyMl(Ml):
        def features(self):
            self.X["close"] = self.df.Close
            self.X["ror"] = self.df.Close.pct_change(1)
            self.X["sma"] = self.sma(period=5)
            self.X["ema"] = self.ema(period=5)
            self.X["upper"], self.X["mid"], self.X["lower"] = self.bbands(
                period=20, band=2
            )
            self.X["macd"], self.X["signal"] = self.macd(
                fast_period=12, slow_period=26, signal_period=9
            )
            self.X["k"], self.X["d"] = self.stoch(k_period=5, d_period=3)
            self.X["rsi"] = self.rsi(period=14)
            self.X["atr"] = self.atr(period=14)
            self.X["mom"] = self.mom(period=10)

    yield MyMl(
        mail_address="dummy@dummy",
        password="dummy",
        outputs_dir_path="tests",
        model_dir_path="tests",
        data_dir_path="tests",
        ticker="7203",  # TOYOTA
        size=100,  # 100 shares
    )


@pytest.fixture(scope="function", autouse=True)
def ml(scope_module, mocker):
    mocker.patch(
        "jquants_ml.Ml._get_prices_daily_quotes",
        return_value=pd.read_csv("tests/7203.d.csv", index_col=0, parse_dates=True),
    )
    yield scope_module


# @pytest.mark.skip
def test_backtest(ml):
    ml.backtest()


# @pytest.mark.skip
def test_predict(ml):
    ml.predict()
