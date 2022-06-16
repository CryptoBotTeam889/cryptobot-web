import pandas as pd
import numpy as np
from datetime import datetime
import mplfinance as mpf
import streamlit as st
from datetime import datetime
import requests

INTERVAL = "1h"
SYMBOL = "ETHUSDT"
BASE_API_URL = "https://cryptobot-889-amz3m5pjwa-ew.a.run.app"
# THRESHOLD =.517037
THRESHOLD = 0.517037
INITIAL_INVEST = 10000
DAYS_CANDLES = 30
# candles=get_candles(SYMBOL, INTERVAL, datetime.now()-timedelta(days=DAYS_CANDLES), datetime.now())

CSS = """
h1 {
    color: red;
}
.stApp {
    background-image: url(https://img.freepik.com/vector-gratis/fondo-tecnologia-digital-abstracto-particulas-rojas_1017-23148.jpg?t=st=1654985006~exp=1654985606~hmac=f6232ddec6474cf502b77986e126d907acdcc9d0820bedebdf10909410b43a88&w=996);
    background-size: cover;
}
"""
st.write(f"<style>{CSS}</style>", unsafe_allow_html=True)


def get_candles(symbol, interval, start_time: datetime, end_time: datetime):
    response = requests.get(
        BASE_API_URL + "/candles",
        params=dict(
            symbol=symbol,
            interval=interval,
            start_time=round(start_time.timestamp()),
            end_time=round(end_time.timestamp()),
        ),
    ).json()
    return response


@st.cache
def candles_hist():
    candles=get_candles(SYMBOL, INTERVAL, datetime(2022 , 4 , 15), datetime.now())
    df = pd.DataFrame(candles)
    df.open_time = df.open_time.apply(lambda x: datetime.utcfromtimestamp(x/1000))
    return df
candles_hist=candles_hist()

def predict(
    symbol, interval, start_time: datetime, end_time: datetime
):
    response = requests.get(
        BASE_API_URL + "/predict-range",
        params=dict(
            symbol=symbol,
            interval=interval,
            init=round(start_time.timestamp()),
            end=round(end_time.timestamp()),
        ),
    ).json()
    return response

st.markdown("""
            ##             CRYPTOBOT
            """
            )

def get_select_box_data():

        return pd.DataFrame({
          'Pair List': ["ETHEREUM (ETHUSDT)",
                        "BITCOIN (BTCUSDT)",
                        "TETHER (THETAUSDT)"]
        })

box_data = get_select_box_data()

st.selectbox('Select a Currency to operate (in USD)', box_data['Pair List'])


st.markdown("""
            ### HISTORICAL ETH PRICE CANDLESTICKS (IN USD)
            (WITH MOVING AVERAGES AND BUY/SELL VOLUME)
            """
            )


df_plot = candles_hist[["open_time","open","high","low","close","volume"]].set_index("open_time")
df_plot_hist = df_plot[:720]

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_historical(df):

    fig = mpf.plot(df,style='yahoo', type="candlestick" ,volume=True,mav=(20,50,100))
    return fig

st.pyplot(plot_historical(df_plot_hist))

st.markdown("""
            ### INITIAL AND END HISTORICAL PERIOD VALUES
            """
            )

col1, col2 = st.columns(2)
col1.metric("ETC PERIOD OPEN", df_plot_hist["open"][0])
col2.metric("ETC PERIOD CLOSE", df_plot_hist["close"][-1],"{:.2%}".format((df_plot_hist["close"][-1]-df_plot_hist["open"][0])/df_plot_hist["open"][0]))

def symbol_overzero(symbol,price):
    signal   = []
    previous = 1
    for date,value in symbol.iteritems():
        if value > 0 and previous <= 0:
            signal.append(price[date]*0.98)  # distancia entre la vela y el triangulo
        else:
            signal.append(np.nan)
        previous = value
    return signal

def symbol_abovezero(symbol,price):
    signal   = []
    previous = 1
    for date,value in symbol.iteritems():
        if value == 0 and previous != 0:
            signal.append(price[date]*1.02)
        else:
            signal.append(np.nan)
        previous = value
    return signal

st.markdown("""
            ### HOW MUCH DO YOU WANT TO INVEST WITH CRYPTOBOT?
            """
            )

number = st.number_input('Invest Simulator')
INITIAL_INVEST = number

@st.cache
def prediction():
    prediction=predict(SYMBOL, INTERVAL, datetime(2022 , 5 , 15), datetime.now())
    prediction=pd.DataFrame(prediction)
    return prediction
prediction=prediction()

@st.cache
def candles_pred():
    candles=get_candles(SYMBOL, INTERVAL, datetime(2022 , 5 , 15), datetime.now())
    df = pd.DataFrame(candles)
    df.open_time = df.open_time.apply(lambda x: datetime.utcfromtimestamp(x/1000))
    return df
candles_hist=candles_pred()

def create_df_stock(candles, prediction):
    df_stock = candles[["open_time","open","close"]]
    df_stock["symbol"] = prediction["predictions"].apply(lambda x: 0 if x <=THRESHOLD else 1)
    df_stock["USD"] = 0
    df_stock["ETH"] = 0
    df_stock.dropna(inplace=True)
    df_stock.reset_index(inplace=True)
    for index,row in df_stock.iterrows():
        if index == 0:
            df_stock.loc[index,'USD'] = INITIAL_INVEST
        else:
            df_stock.loc[index,'USD'] = df_stock.loc[index-1,'USD'] - (df_stock.loc[index,'symbol']*df_stock.loc[index-1,'USD'] - (1-df_stock.loc[index,'symbol'])*(df_stock.loc[index-1,'ETH']*df_stock.loc[index-1,'close']))
            df_stock.loc[index,'ETH'] = df_stock.loc[index-1,'ETH'] - ((1-df_stock.loc[index,'symbol'])*df_stock.loc[index-1,'ETH'] - df_stock.loc[index,'symbol']*(df_stock.loc[index-1,'USD']/df_stock.loc[index-1,'close']))
    df_stock["TOTAL_VALUE_IN_USD"] = df_stock["USD"] + df_stock["ETH"]*df_stock["close"]
    df_stock[["USD","TOTAL_VALUE_IN_USD"]] = df_stock[["USD","TOTAL_VALUE_IN_USD"]].applymap(lambda x: round(x,1))
    df_stock = df_stock.set_index("open_time")
    return df_stock

df_stock = create_df_stock(candles_hist,prediction)

investment_result = df_stock["TOTAL_VALUE_IN_USD"][len(df_stock["TOTAL_VALUE_IN_USD"])-1] - number

# low_signal = symbol_abovezero(df_plot_pred['symbol'], df_plot_pred['close'])
# high_signal = symbol_overzero(df_plot_pred['symbol'], df_plot_pred['close'])

# apds = [
#        mpf.make_addplot(high_signal,type='scatter',markersize=50,marker='^', color='green'),
#        mpf.make_addplot(low_signal,type='scatter',markersize=50,marker='v', color='red')
#       ]

# mpf.plot(df_stock,addplot=apds,style='yahoo', type="candlestick" ,volume=True,mav=(20,50,100))

df_stock_pred=df_stock[len(df_stock)-720:]

st.markdown("""
            ### PREDICTED ETH STOCK EVOLUTION
            """)

st.line_chart(df_stock_pred["ETH"], height = 100)

st.markdown("""
            ### PREDICTED ETH PRICE CANDLESTICKS (IN USD)
            """
            )

df_plot_pred = df_plot[720:]

st.pyplot(mpf.plot(df_plot_pred, style='yahoo', type="candlestick",mav=(20,50,100)))

st.markdown("""
            ### PREDICTED INVEST EVOLUTION (IN USD)
            """
            )

df_stock_pred=df_stock[len(df_stock)-720:]
st.line_chart(df_stock_pred["TOTAL_VALUE_IN_USD"])


st.markdown("""
            ### INITIAL AND END PREDICTED PERIOD VALUES
            """
            )

col3, col4 = st.columns(2)
col3.metric("ETC PERIOD OPEN", df_stock_pred["open"][0])
col4.metric("ETC PERIOD CLOSE", df_stock_pred["close"][-1],"{:.2%}".format((df_stock_pred["close"][-1]-df_stock_pred["open"][0])/df_stock_pred["open"][0]))

col1, col2 = st.columns(2)
col1.metric("USING CRYPTOBOT, YOUR RESULT WOULD HAVE BEEN:", round(investment_result,3),"{:.2%}".format((investment_result/number)))

if st.button('Deposit funds in  wallet'):
    # print is visible in the server output, not in the page
    st.write("Scan QR to connect with wallet")
    st.image("images/QR.png")
