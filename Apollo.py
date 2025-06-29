import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import yfinance as yf
import warnings

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import (GlobalInlineStyleSheet, InlineStyleSheet,
    Select, NumericInput, Button, Div, DataTable, 
    TableColumn, ColumnDataSource
)
from bokeh.models.widgets import DatePicker

import yfinance as yf
from datetime import datetime, timedelta


base_variables = """ :host { /* CSS Custom Properties for easy theming */ --primary-color: #8b5cf6; --secondary-color: #06b6d4; --background-color: #1f2937; --surface-color: #343838; --text-color: #f9fafb; --accent-color: #f59e0b; --danger-color: #ef4444; --success-color: #10b981; --border-color: #4b5563; --hover-color: #6366f1; background: none !important; } """

gstyle = GlobalInlineStyleSheet(css=""" html, body, .bk, .bk-root {background-color: #343838; margin: 0; padding: 0; height: 100%; color: white; font-family: 'Consolas', 'Courier New', monospace; } .bk { color: white; } .bk-input, .bk-btn, .bk-select, .bk-slider-title, .bk-headers, .bk-label, .bk-title, .bk-legend, .bk-axis-label { color: white !important; } .bk-input::placeholder { color: #aaaaaa !important; } """)
button_style = InlineStyleSheet(css=base_variables + """ :host button { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important; color: white !important; border: none !important; border-radius: 6px !important; padding: 10px 20px !important; font-size: 14px !important; font-weight: 600 !important; cursor: pointer !important; transition: all 0.2s ease !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; } :host button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important; background: linear-gradient(135deg, var(--hover-color), var(--primary-color)) !important; } :host button:active { transform: translateY(0) !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; } :host button:disabled { background: #6b7280 !important; cursor: not-allowed !important; transform: none !important; box-shadow: none !important; } """)
style2 = InlineStyleSheet(css=""" .bk-input { background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1px solid #3c3c3c; border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } /* Input Hover */ .bk-input:hover { background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1.5px solid #ff3232;        /* Red border */ box-shadow: 0 0 9px 2px #ff3232cc;  /* Red glow */ border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } /* Input Focus */ .bk-input:focus { background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1.5px solid #ff3232; box-shadow: 0 0 11px 3px #ff3232dd; border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } .bk-input:active { outline: none; background-color: #1e1e1e; color: #d4d4d4; font-weight: 500; border: 1.5px solid #ff3232; box-shadow: 0 0 14px 3px #ff3232; border-radius: 5px; padding: 6px 10px; font-family: 'Consolas', 'Courier New', monospace; transition: all 0.2s ease; } .bk-input:-webkit-autofill { background-color: #1e1e1e !important; color: #d4d4d4 !important; -webkit-box-shadow: 0 0 0px 1000px #1e1e1e inset !important; -webkit-text-fill-color: #d4d4d4 !important; } """)
dark_table_style = InlineStyleSheet(css=""" /* Container styling */ :host { background: #2e2e30 !important; border-radius: 14px !important; padding: 16px !important; box-shadow: 0 4px 18px #0006 !important; margin: 10px !important; } /* Headers */ :host div[class*="header"], :host div[class*="slick-header"], :host th, :host [class*="header"] { background: #2e2e30 !important; color: #ffd000 !important; } /* cells */ :host div[class*="cell"], :host div[class*="slick-cell"], :host td { background: #565755 !important; color: #f4d67b !important; } /* Alternating rows */ :host div[class*="row"]:nth-child(even) div[class*="cell"], :host div[class*="slick-row"]:nth-child(even) div[class*="slick-cell"], :host tr:nth-child(even) td { background: #2a2a2c !important; color: #f4d67b !important; } /* Hover effects */ :host div[class*="row"]:hover div[class*="cell"], :host div[class*="slick-row"]:hover div[class*="slick-cell"], :host tr:hover td { background: #3eafff !important; color: #0c0c0c !important; border-color: #ff0000 !important; border-style: solid !important; border-width: 1px !important; } """)
warnings.filterwarnings('ignore')
fancy_div_style = InlineStyleSheet(css=""" :host { position: relative; background: #444444; color: #fff; border-radius: 16px; padding: 18px 28px; text-align: center; overflow: hidden; box-shadow: 0 6px 18px red; margin: 28px auto; } """)
# === FRIENDLY NAME TO YAHOO SYMBOL MAPPING ===
# 1. MASTER FRIENDLY NAME -> YAHOO SYMBOL MAP
FRIENDLY_TO_YAHOO = {
    # US Tech, MegaCap
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL",
    "meta": "META", "nvidia": "NVDA", "tesla": "TSLA", "intel": "INTC", "amd": "AMD",
    "netflix": "NFLX", "adobe": "ADBE", "oracle": "ORCL", "ibm": "IBM", "salesforce": "CRM",
    "qualcomm": "QCOM", "broadcom": "AVGO", "cisco": "CSCO", "snowflake": "SNOW",
    # US Consumer/Bluechip
    "coca-cola": "KO", "mcdonalds": "MCD", "nike": "NKE", "pepsi": "PEP", "boeing": "BA",
    "paypal": "PYPL", "uber": "UBER", "visa": "V", "mastercard": "MA", "alibaba": "BABA",
    "starbucks": "SBUX", "walmart": "WMT", "costco": "COST", "procter & gamble": "PG",
    "disney": "DIS", "target": "TGT", "pfizer": "PFE", "johnson & johnson": "JNJ",
    # US Energy, Autos, Industrials
    "exxon": "XOM", "chevron": "CVX", "ford": "F", "ge": "GE", "caterpillar": "CAT",
    "general motors": "GM", "lockheed martin": "LMT", "northrop grumman": "NOC",
    "raytheon": "RTX", "shell": "SHEL", "bp": "BP", "phillips 66": "PSX",
    # ETFs & Indexes (US, world, sectors)
    "arkk": "ARKK", "spy": "SPY", "qqq": "QQQ", "iwm": "IWM", "dia": "DIA",
    "voo": "VOO", "vti": "VTI", "vt": "VT", "efa": "EFA", "eem": "EEM",
    "xlk": "XLK", "xlf": "XLF", "xlv": "XLV", "xle": "XLE", "xly": "XLY",
    "xli": "XLI", "xlc": "XLC", "xlb": "XLB", "xlp": "XLP", "xlre": "XLRE", "xlp": "XLP",
    # Commodities
    "gld": "GLD", "slv": "SLV", "gold": "GC=F", "silver": "SI=F", "brent oil": "BZ=F",
    "wti oil": "CL=F", "natural gas": "NG=F", "copper": "HG=F", "platinum": "PL=F",
    "palladium": "PA=F", "corn": "ZC=F", "soybeans": "ZS=F", "wheat": "ZW=F",
    # Crypto
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "dogecoin": "DOGE-USD", "solana": "SOL-USD",
    "cardano": "ADA-USD", "ripple": "XRP-USD", "polkadot": "DOT-USD", "litecoin": "LTC-USD",
    "chainlink": "LINK-USD", "polygon": "MATIC-USD", "binance coin": "BNB-USD", "tron": "TRX-USD",
    # Forex (major & some minors)
    "aud/usd": "AUDUSD=X", "eur/usd": "EURUSD=X", "gbp/usd": "GBPUSD=X", "usd/jpy": "USDJPY=X",
    "nzd/usd": "NZDUSD=X", "usd/cad": "USDCAD=X", "usd/chf": "USDCHF=X",
    "eur/gbp": "EURGBP=X", "eur/aud": "EURAUD=X", "gbp/jpy": "GBPJPY=X", "chf/jpy": "CHFJPY=X",
    "aud/jpy": "AUDJPY=X", "eur/chf": "EURCHF=X", "gbp/cad": "GBPCAD=X", "aud/cad": "AUDCAD=X",
    "nzd/chf": "NZDCHF=X", "nzd/cad": "NZDCAD=X", "cad/jpy": "CADJPY=X", "eur/cad": "EURCAD=X",
    "cad/chf": "CADCHF=X", "aud/chf": "AUDCHF=X", "aud/nzd": "AUDNZD=X", "eur/jpy": "EURJPY=X",
    "gbp/aud": "GBPAUD=X", "gbp/chf": "GBPCHF=X", "gbp/nzd": "GBPNZD=X", "usd/hkd": "HKD=X",
    "usd/sgd": "SGD=X", "usd/try": "TRY=X", "usd/zar": "ZAR=X", "usd/mxn": "MXN=X",
    "usd/inr": "INR=X", "usd/krw": "KRW=X",
    # World Indices
    "s&p500": "^GSPC", "nasdaq": "^IXIC", "dow jones": "^DJI",
    "ftse100": "^FTSE", "dax": "^GDAXI", "cac40": "^FCHI", "nikkei": "^N225",
    "hang seng": "^HSI", "sensex": "^BSESN", "kospi": "^KS11", "tsx": "^GSPTSE",
    "asx200": "^AXJO", "smi": "^SSMI", "ibex": "^IBEX", "aex": "^AEX",
    # International MegaCaps/ADRs
    "samsung": "005930.KS", "sony": "SONY", "toyota": "TM", "honda": "HMC",
    "volkswagen": "VWAGY", "bayer": "BAYRY", "siemens": "SIEGY", "novartis": "NVS",
    "nestle": "NSRGY", "unilever": "UL", "shell": "SHEL", "astra zeneca": "AZN",
    "baidu": "BIDU", "tencent": "TCEHY", "shopify": "SHOP", "infosys": "INFY",
    "biontech": "BNTX", "ping an": "PNGAY",
    # Others
    "amgen": "AMGN", "moderna": "MRNA", "roche": "RHHBY", "glaxosmithkline": "GSK",
    "sanofi": "SNY", "prudential": "PRU", "manulife": "MFC",
}

# Auto-generated dropdown for your app:
TICKER_OPTIONS = sorted([
    (k.replace("_", " ").title(), k) for k in FRIENDLY_TO_YAHOO.keys()
])




def resolve_symbol(user_input):
    if user_input is None:
        return None
    cleaned = user_input.strip().lower()
    cleaned = cleaned.replace("/", "") if "/" in cleaned else cleaned
    yahoo = FRIENDLY_TO_YAHOO.get(cleaned, cleaned)
    # If not found and looks like FX, append =X
    if yahoo.isalpha() and len(yahoo) == 6 and not yahoo.endswith("=X"):
        yahoo += "=X"
    return yahoo



class ForexDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
    def get_historical_data(self, lookback_years=2):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * lookback_years)
        try:
            df = yf.download(
                self.symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                actions=False,
                auto_adjust=True
            )
            if df is None or df.empty:
                print(f"No Yahoo data for {self.symbol}")
                return None

            df = df.reset_index()
            # This fixes the MultiIndex/tuple bug:
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

            required_cols = ['date', 'close', 'open', 'high', 'low', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ {self.symbol}: missing column {col}")
                    return None

            df = df.dropna(subset=['close'])
            return df[required_cols]
        except Exception as e:
            print(f"Yahoo Finance fetch error: {e}")
            return None


class ForexSARIMAPredictor:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    def prepare_data(self, data):
        df = data.copy()
        if 'date' not in df.columns:
            df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df
    def train(self, historical_data):
        if historical_data is None or historical_data.empty:
            raise Exception("No historical data for training.")
        data = self.prepare_data(historical_data)
        self.model = SARIMAX(
            data['close'],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False)
    def predict_next_n_days(self, historical_data, steps=5):
        if self.fitted_model is None:
            raise Exception("Model needs to be trained first")
        forecast_obj = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)
        current_price = historical_data['close'].iloc[-1]
        signals = []
        confidences = []
        for pred, (lower, upper) in zip(forecast, conf_int.values):
            signal = 'BUY' if pred > current_price else 'SELL'
            interval_width = upper - lower
            confidence = (1 - (interval_width / abs(pred))) * 100 if pred != 0 else 0
            confidence = max(0, min(100, confidence))
            signals.append(signal)
            confidences.append(confidence)
            current_price = pred
        return {
            'predicted_prices': forecast.values,
            'signals': signals,
            'confidences': confidences,
            'confidence_intervals': conf_int.values
        }
    def evaluate_model(self, historical_data):
        if historical_data is None or len(historical_data) < 30:
            return None
        df = self.prepare_data(historical_data)
        test_days = min(90, len(df))
        y_true = df['close'][-test_days:]
        predictions = self.fitted_model.get_prediction(start=-test_days).predicted_mean
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = np.mean(np.abs(y_true - predictions))
        mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
        return {'rmse': rmse, 'mae': mae, 'mape': mape}
curdoc().theme = "dark_minimal"
title_div = Div(text="<h1>Apollo: 📈 Forecasts for Stocks, Crypto, & Forex</h1>", width=800, height=60)

ticker_select = Select(title="Select Ticker:", 
                      options=[("", "Select a ticker...")] + TICKER_OPTIONS,
                      value="", width=200, stylesheets = [style2])

start_date_picker = DatePicker(title="Start Date:", 
                              value=date.today() - timedelta(days=30),
                              min_date=date(1995, 1, 1),
                              max_date=date.today(),
                              width=150, stylesheets = [style2])

end_date_picker = DatePicker(title="End Date:", 
                            value=date.today(),
                            min_date=date(1995, 1, 1),
                            max_date=date.today(),
                            width=150, stylesheets = [style2])

forecast_days_input = NumericInput(title="Forecast Days:", value=5, low=1, high=30, width=120, stylesheets = [style2])

forecast_button = Button(label="Forecast & Indicators", button_type="primary", width=200, stylesheets = [button_style])
find_assets_button = Button(label="🔍 Find Reliable Assets", button_type="success", width=200, stylesheets = [button_style])

status_div = Div(text="", width=800, height=30)
results_div = Div(text="", width=800, height=100, styles = {'font-size': '16px',})
assets_div = Div(text="", width=800, height=250, styles = {'font-size': '16px',})

plot = figure(title="Select a ticker and click 'Forecast & Indicators' to begin", 
              x_axis_type='datetime', width=800, height=400, 
              tools="hover,pan,wheel_zoom,box_zoom,reset,save", active_scroll='wheel_zoom',
              border_fill_color="#444444", background_fill_color="#444444",)
empty_source = ColumnDataSource(data=dict(x=[], y=[]))
plot.line('x', 'y', source=empty_source, alpha=0)

forecast_columns = [
    TableColumn(field="day", title="Day"),
    TableColumn(field="price", title="Predicted Price"),
    TableColumn(field="signal", title="Signal"),
    TableColumn(field="confidence", title="Confidence (%)")
]
forecast_source = ColumnDataSource(data=dict(day=[], price=[], signal=[], confidence=[]))
forecast_table = DataTable(source=forecast_source, columns=forecast_columns, width=600, height=200, stylesheets=[dark_table_style],)

def update_forecast():
    ticker = ticker_select.value
    start_date_val = start_date_picker.value
    end_date_val = end_date_picker.value
    days = int(forecast_days_input.value) if forecast_days_input.value else 5
    if not ticker:
        status_div.text = "⚠️ Please select a ticker"
        return
    try:
        _symbol = resolve_symbol(ticker)
        status_div.text = f"🔄 Fetching data for {ticker} ({_symbol})..."
        curdoc().add_next_tick_callback(lambda: None)
        fetcher = ForexDataFetcher(_symbol)
        data = fetcher.get_historical_data(lookback_years=2)
        if data is None or data.empty or 'date' not in data.columns:
            status_div.text = f"⚠️ No valid data for '{ticker}'"
            results_div.text = ""
            forecast_source.data = dict(day=[], price=[], signal=[], confidence=[])
            plot.renderers = []
            plot.title.text = "No Data"
            return
        if not np.issubdtype(data['date'].dtype, np.datetime64):
            try:
                data['date'] = pd.to_datetime(data['date'])
            except Exception:
                status_div.text = "⚠️ Could not parse 'date' column as datetime."
                return
        start_date_ts = pd.to_datetime(start_date_val)
        end_date_ts = pd.to_datetime(end_date_val)
        plot_data = data[(data['date'] >= start_date_ts) & (data['date'] <= end_date_ts)].copy()
        if plot_data.empty:
            status_div.text = "⚠️ No data in the selected date range."
            results_div.text = ""
            forecast_source.data = dict(day=[], price=[], signal=[], confidence=[])
            plot.renderers = []
            plot.title.text = "No Data"
            return
        predictor = ForexSARIMAPredictor()
        predictor.train(data)
        forecast_result = predictor.predict_next_n_days(data, steps=days)
        forecast_prices = forecast_result['predicted_prices'][:days]
        signals = forecast_result['signals'][:days]
        confidences = forecast_result['confidences'][:days]
        conf_int = forecast_result['confidence_intervals'][:days]
        last_date = data['date'].iloc[-1]
        forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)
        plot.renderers = []
        empty_source = ColumnDataSource(data=dict(x=[], y=[]))
        plot.line('x', 'y', source=empty_source, alpha=0)
        hist_source = ColumnDataSource(data=dict(
            x=plot_data['date'], y=plot_data['close']
        ))
        # plot.line('x', 'y', source=hist_source, legend_label='Historical Close', line_width=2)
        forecast_source_line = ColumnDataSource(data=dict(
            x=forecast_dates, y=forecast_prices
        ))
        plot.line('x', 'y', source=forecast_source_line, legend_label='SARIMA Forecast',color = 'orange', line_width=2,)
        plot.circle('x', 'y', source=forecast_source_line, size=12, color = 'orange')
        # band_x = np.concatenate([forecast_dates, forecast_dates[::-1]])
        # band_y = np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]])
        # plot.patches(xs=[band_x], ys=[band_y], color='green', alpha=0.2, legend_label='95% Confidence Interval')
        plot.title.text = f"SARIMA {ticker.upper()} {days}-Day Forecast"
        plot.legend.location = "top_left"
        x_idx = np.arange(1, days + 1)
        slope, intercept, r_value, p_value, std_err = linregress(x_idx, forecast_prices)
        trend_line = intercept + slope * x_idx
        trend_source = ColumnDataSource(data=dict(x=forecast_dates, y=trend_line))
        plot.line('x', 'y', source=trend_source, color = 'deepskyblue', line_width=6, legend_label="Trend Line")
        trend_msg = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        metrics = predictor.evaluate_model(data)
        eval_text = ""
        if metrics is not None:
            eval_text = (
                f"Model RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | "
                f"MAPE: {metrics['mape']:.2f}%<br>"
            )

        # NEW: Suggestion message with color logic
        if p_value < 0.05:
            if slope > 0:
                suggestion = '<span style="color:limegreen;font-weight:bold; font-size: 1.6em">Suggest to BUY (Significant Uptrend)</span>'
            else:
                suggestion = '<span style="color:#ff3333;font-weight:bold; font-size: 1.6em">Suggest to SELL (Significant Downtrend)</span>'
        else:
            suggestion = '<span style="color:silver">No statistically significant trend</span>'

        eval_text += (
            f"Slope: {slope:.4f} | p-value: {p_value:.4f} ({trend_msg})<br>"
            f"{suggestion}"
        )

        table_data = {
            'day': [str(i+1) for i in range(days)],
            'price': [f"{forecast_prices[i]:.4f}" for i in range(days)],
            'signal': signals,
            'confidence': [f"{confidences[i]:.1f}" for i in range(days)]
        }
        forecast_source.data = table_data
        status_div.text = f"✅ SARIMA forecast successful for {ticker.upper()}"
        results_div.text = f"<h3>Forecast Table & Trading Signals</h3><br><b>{eval_text}</b>"


    except Exception as e:
        print(f"[ERROR] SARIMA Forecast: {e}")
        status_div.text = "⚠️ SARIMA forecast failed. Please try again."
        results_div.text = ""
        forecast_source.data = dict(day=[], price=[], signal=[], confidence=[])
        plot.renderers = []
        plot.title.text = "No Data"

def find_reliable_assets():
    curdoc().add_next_tick_callback(lambda: None)

    significant_signals = []
    scan_days = 5
    for friendly_name, user_symbol in TICKER_OPTIONS:
        try:
            yahoo_symbol = resolve_symbol(user_symbol)
            fetcher = ForexDataFetcher(yahoo_symbol)
            df = fetcher.get_historical_data(lookback_years=2)
            if df is None or len(df) < 50:
                continue
            predictor = ForexSARIMAPredictor()
            predictor.train(df)
            forecast = predictor.predict_next_n_days(df, steps=scan_days)
            preds = forecast['predicted_prices'][:scan_days]
            x_idx = np.arange(1, scan_days+1)
            slope, intercept, r_value, p_value, std_err = linregress(x_idx, preds)
            if p_value < 0.05:
                trend = "🔼 Uptrend" if slope > 0 else "🔽 Downtrend"
                first_signal = forecast['signals'][0]
                significant_signals.append({
                    "name": friendly_name,
                    "code": yahoo_symbol,
                    "trend": trend,
                    "slope": slope,
                    "p_value": p_value,
                    "signal": first_signal,
                    "final_forecast": preds[-1]
                })
        except Exception as e:
            continue
    if not significant_signals:
        assets_div.text = "<b>⚠️ No significant SARIMA up/down trends found in sample set.</b>"
        return
    # Sort by absolute value of slope (strength of move)
    significant_signals.sort(key=lambda x: abs(x["slope"]), reverse=True)
    result_html = "<h3>📈 SARIMA Significant Forecast Trends (Up/Down, p<0.05):</h3><ul>"
    for sig in significant_signals[:10]:
        result_html += (
            f"<li><b>{sig['name']}</b> ({sig['code'].upper()}) "
            f"→ {sig['trend']} | Slope: <b>{sig['slope']:.3f}</b> | p-value: <b>{sig['p_value']:.4f}</b> "
            f"| 1st Signal: <b>{sig['signal']}</b> | Final Forecast: {sig['final_forecast']:.3f}</li>"
        )
    result_html += "</ul>"
    assets_div.text = result_html

def start_find_reliable_assets():
    # This is called when the button is clicked
    assets_div.text = "<b>🔄 Please wait: Scanning for reliable significant trends (this may take 2-3 minutes)...</b>"
    curdoc().add_next_tick_callback(find_reliable_assets)

forecast_button.on_click(lambda: update_forecast())
find_assets_button.on_click(start_find_reliable_assets)
about_div = Div( text=""" <div style="text-align:center; color:#00ffe0; font-size:1.07em; font-family:Consolas, monospace;"> Developed with <span style="color:#ff4c4c;">&#10084;&#65039;</span> by <a href="https://github.com/mixstam1821" target="_blank" style="color:#ffb031; font-weight:bold; text-decoration:none;"> mixstam1821 </a> </div> """, width=420, height=38, styles={"margin-top": "10px"} )

controls_row = row(
    ticker_select, start_date_picker, end_date_picker, forecast_days_input, column(Div(text=" ", height = 2),forecast_button)
)
layout = column(
    title_div,
    controls_row,
    status_div,
    row(column(plot, stylesheets=[fancy_div_style],styles = {'margin-left': '20px'}),
    column(results_div,Div(text=" ", width=600, height=70),
    forecast_table,styles = {'margin-left': '40px'}),),
    find_assets_button,
    row(assets_div,about_div),
    stylesheets = [gstyle]
)

curdoc().add_root(layout)
curdoc().title = "Apollo: 📈 Forecasts for Stocks, Crypto, & Forex"
