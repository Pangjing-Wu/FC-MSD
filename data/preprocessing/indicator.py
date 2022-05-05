import pandas as pd
import talib as ta


__all__ = ['CalculateIndicators']


class CalculateIndicators(object):

    def __init__(self) -> None:
        self.indicator = ['ma7', 'ma14', 'ma21', 'ma21', 'pct_chg', 'bbands_upper', 'bbands_mid',
                          'bbands_lower', 'adx', 'cci', 'mfi', 'rsi7', 'rsi14', 'rsi21', 'aroondown',
                          'aroonup', 'macd', 'macdsignal', 'macdhist', 'ad', 'nart', 'dcperiod',
                          'dcphase', 'inhpase', 'quadrature', 'sine', 'leadsine', 'trendmode']

    def get_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # overlap studies
        df['ma7']  = ta.MA(data.close, timeperiod=7)
        df['ma14'] = ta.MA(data.close, timeperiod=14)
        df['ma21'] = ta.MA(data.close, timeperiod=21)
        df['pct_chg'] = ta.ROC(data.close, timeperiod=1)
        df['bbands_upper'], df['bbands_mid'], df['bbands_lower'] = ta.BBANDS(data.close, timeperiod=5, nbdevup=2, nbdevdn=2)

        # momentum indicators
        df['adx'] = ta.ADX(data.high, data.low, data.close, timeperiod=14)
        df['cci'] = ta.CCI(data.high, data.low, data.close, timeperiod=14)
        df['mfi']  = ta.MFI(data.high, data.low, data.close, data.volume, timeperiod=14)
        df['rsi7']  = ta.RSI(data.close, timeperiod=7)
        df['rsi14'] = ta.RSI(data.close, timeperiod=14)
        df['rsi21'] = ta.RSI(data.close, timeperiod=21)
        df['aroondown'], df['aroonup'] = ta.AROON(data.high, data.low, timeperiod=14)
        df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(data.close, fastperiod=12, slowperiod=26, signalperiod=9)

        # volume indicators
        df['ad'] = ta.AD(data.high, data.low, data.close, data.volume)

        # volatility indicators
        df['nart'] = ta.NATR(data.high, data.low, data.close, timeperiod=14)

        # cycle indicators
        df['dcperiod'] = ta.HT_DCPERIOD(data.close)
        df['dcphase'] = ta.HT_DCPHASE(data.close)
        df['inhpase'], df['quadrature'] = ta.HT_PHASOR(data.close)
        df['sine'], df['leadsine'] = ta.HT_SINE(data.close)
        df['trendmode'] = ta.HT_TRENDMODE(data.close)
        return df