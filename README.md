# jquants-ml

[![PyPI](https://img.shields.io/pypi/v/jquants-ml)](https://pypi.org/project/jquants-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/10mohi6/jquants-ml-python/branch/main/graph/badge.svg?token=DukbkJ6Pnx)](https://codecov.io/gh/10mohi6/jquants-ml-python)
[![Build Status](https://app.travis-ci.com/10mohi6/jquants-ml-python.svg?branch=main)](https://app.travis-ci.com/10mohi6/jquants-ml-python)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jquants-ml)](https://pypi.org/project/jquants-ml/)
[![Downloads](https://pepy.tech/badge/jquants-ml)](https://pepy.tech/project/jquants-ml)

jquants-ml is a python library for machine learning with japanese stock trade using J-Quants on Python 3.8 and above.

## Installation

    $ pip install jquants-ml

## Usage

### basic backtest

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

![7203.png](https://raw.githubusercontent.com/10mohi6/jquants-ml-python/main/tests/7203.png)

```python
{'long': {'maximum drawdown': '21700.000',
          'profit': '12100.000',
          'profit factor': '1.066',
          'riskreward ratio': '1.099',
          'sharpe ratio': '0.031',
          'trades': '193.000',
          'win rate': '0.492'},
 'short': {'maximum drawdown': '86800.000',
           'profit': '-53950.000',
           'profit factor': '0.688',
           'riskreward ratio': '0.809',
           'sharpe ratio': '0.123',
           'trades': '111.000',
           'win rate': '0.459'},
 'total': {'maximum drawdown': '83400.000',
           'profit': '-41850.000',
           'profit factor': '0.883',
           'riskreward ratio': '0.955',
           'sharpe ratio': '0.068',
           'trades': '304.000',
           'win rate': '0.480'}}
```

### basic predict

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
{'Date': '2023-09-22', 'Price': 2788.5, 'Sign': 'short'}
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

MyMl(
    mail_address="<your J-Quants mail address>",
    password="<your J-Quants password>",
    ticker="7203",  # TOYOTA
    size=100,  # 100 shares
    outputs_dir_path="outputs",
    model_dir_path="model",
    data_dir_path="data",
).backtest()
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
