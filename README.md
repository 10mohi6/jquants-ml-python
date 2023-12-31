# jquants-ml

[![PyPI](https://img.shields.io/pypi/v/jquants-ml)](https://pypi.org/project/jquants-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/10mohi6/jquants-ml-python/graph/badge.svg?token=X8QKKFK6AL)](https://codecov.io/gh/10mohi6/jquants-ml-python)
[![Python package](https://github.com/10mohi6/jquants-ml-python/actions/workflows/python-package.yml/badge.svg)](https://github.com/10mohi6/jquants-ml-python/actions/workflows/python-package.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jquants-ml)](https://pypi.org/project/jquants-ml/)
[![Downloads](https://pepy.tech/badge/jquants-ml)](https://pepy.tech/project/jquants-ml)

jquants-ml is a python library for machine learning with japanese stock trade using J-Quants on Python 3.8 and above.

## Installation

    $ pip install jquants-ml

## Usage

### backtest

```python
from jquants_ml import Ml
import pprint

class MyMl(Ml):
    def features(self):
        self.X["close"] = self.df.Close
        self.X["ror"] = self.df.Close.pct_change(1)
        self.X["sma"] = self.sma(period=5)

ml = MyMl(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    ticker="7203",  # TOYOTA
    size=100,  # 100 shares
)
pprint.pprint(ml.backtest())
```

![7203.png](https://raw.githubusercontent.com/10mohi6/jquants-ml-python/main/tests/7203.p.png)

```python
{'long': {'maximum drawdown': '15250.000',
          'profit': '12700.000',
          'profit factor': '1.183',
          'riskreward ratio': '1.213',
          'sharpe ratio': '0.063',
          'trades': '81.000',
          'win rate': '0.494'},
 'short': {'maximum drawdown': '50100.000',
           'profit': '-43800.000',
           'profit factor': '0.413',
           'riskreward ratio': '0.478',
           'sharpe ratio': '0.298',
           'trades': '41.000',
           'win rate': '0.463'},
 'total': {'maximum drawdown': '47200.000',
           'profit': '-31100.000',
           'profit factor': '0.784',
           'riskreward ratio': '0.837',
           'sharpe ratio': '0.149',
           'trades': '122.000',
           'win rate': '0.484'}}
```

### predict

```python
from jquants_ml import Ml
import pprint

class MyMl(Ml):
    def features(self):
        self.X["close"] = self.df.Close
        self.X["ror"] = self.df.Close.pct_change(1)
        self.X["sma"] = self.sma(period=5)

ml = MyMl(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    ticker="7203",  # TOYOTA
    size=100,  # 100 shares
)
pprint.pprint(ml.predict())
```

```python
{'Date': '2023-08-01', 'Price': 2445.5, 'Sign': 'short'}
```

### advanced

```python
from jquants_ml import Ml

class MyMl(Ml):
    # Awesome Oscillator
    def ao(self, *, fast_period: int = 5, slow_period: int = 34):
        return ((self.df.H + self.df.L) / 2).rolling(fast_period).mean() - (
            (self.df.H + self.df.L) / 2
        ).rolling(slow_period).mean()

    def features(self):
        self.X["ao"] = self.ao(fast_period=5, slow_period=34)
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

ml = MyMl(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    ticker="7203",  # TOYOTA
    size=100,  # 100 shares
    outputs_dir_path="outputs",
    model_dir_path="model",
    data_dir_path="data",
)
pprint.pprint(ml.backtest())
pprint.pprint(ml.predict())
```

## Supported indicators

- Simple Moving Average 'sma'
- Exponential Moving Average 'ema'
- Moving Average Convergence Divergence 'macd'
- Relative Strenght Index 'rsi'
- Bollinger Bands 'bbands'
- Market Momentum 'mom'
- Stochastic Oscillator 'stoch'
- Average True Range 'atr'

## Getting started

For help getting started with J-Quants, view our online [documentation](https://jpx-jquants.com/).
