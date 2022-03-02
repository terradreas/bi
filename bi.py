import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

st.set_page_config(layout="wide")
st.title('Exceedingly exciting extremely experimental Terra dashboard')

URL1 = 'https://api.flipsidecrypto.com/api/v2/queries/08c78cad-d2a6-4fa8-89cd-50884e55a389/data/latest'

#  @st.cache
def load_data():
    data = pd.read_json(URL1)
    data = data.sort_values("WEEK")
    #  data["REPAYS_AMOUNT"] = -data["REPAYS_AMOUNT"]
    #  lowercase = lambda x: str(x).lower()
    #  data.rename(lowercase, axis='columns', inplace=True)
    return data

#  data_load_state = st.text('Loading data...')


buttonBasic = st.sidebar.button("Basics")#, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)
buttonCDP = st.sidebar.button("CDP")

if buttonBasic:
    st.write("Hi")


if buttonCDP:

    col1, col2 = st.columns([2,5])

    table = load_data()

    #  st.write(table)
    bor = table["BORROW_AMOUNT"].to_numpy()
    paid = table["REPAYS_AMOUNT"].to_numpy()
    x = table["WEEK"].apply(lambda x: pd.Timestamp(x).date())

    #  col2.write(table)
    fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(10,10))
    fig.suptitle('Borrow change')
   
    cumu = np.cumsum(bor-paid)*100
    cumu /= 5.5e9

    #  x2 = [datetime.datetime.strptime(str(d),"%Y-%m-%d").date() for d in x]
    ax0.plot(x, cumu)
    ax0.plot(x, np.zeros(len(x)), "k--")
    ax0.set_title("Cumulative change")
    ax0.set_ylabel("percent")
    ax1.bar(x, -paid)
    ax1.bar(x, bor)
    ax1.legend(["Repaid", "Borrowed"],prop={'size': 7}) 
    ax1.set_ylabel("UST")
    ax1.set_title("Deltas")
    ax2.bar(x, bor-paid)
    ax2.set_ylabel("UST")
    ax2.set_title("Net Delta")

    ax0.set_xticklabels(x, rotation=45, ha='right')
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax2.set_xticklabels(x, rotation=45, ha='right')

    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax0.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    #  locator = mdates.DayLocator()
    #  ax0.xaxis.set_major_locator(locator)

    fig.subplots_adjust(hspace=1)
    st.pyplot(fig)
#  chart_data = pd.DataFrame(
#       np.random.randn(50, 3),
#       columns=["a", "b", "c"])
#
#  st.bar_chart(chart_data)
