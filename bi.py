import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
#  import altair as alt

import math
import re

from decimal import Decimal

### following millify function by  Alexander Zaitsev (azaitsev@gmail.com)
# __copyright__ = "Copyright 2018, azaitsev@gmail.com"
# __license__ = "MIT"
# __version__ = "0.1.1"


def remove_exponent(d):
    """Remove exponent."""
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def millify(n, precision=0, drop_nulls=True, prefixes=[]):
    """Humanize number."""
    millnames = ['', 'k', 'M', 'B', 'T', 'P', 'E', 'Z', 'Y']
    if prefixes:
        millnames = ['']
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    result = '{:.{precision}f}'.format(n / 10**(3 * millidx), precision=precision)
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    return '{0}{dx}'.format(result, dx=millnames[millidx])
####################################



st.set_page_config(layout="wide")

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
if 'section' not in st.session_state:
    st.session_state['section'] = 'basic'

def click_basic(): 
    st.session_state["section"] = "basic"
def click_CDP(): 
    st.session_state["section"] = "cdp"

buttonBasic = st.sidebar.button("Basics", on_click = click_basic)#, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)
buttonCDP = st.sidebar.button("CDP", on_click = click_CDP)


if st.session_state["section"] == "basic":
    st.title('Misc')

if st.session_state["section"] == "cdp":
    st.title('Borrowing')
    
    table = load_data()

    #  st.write(table)
    bor = table["BORROW_AMOUNT"].to_numpy()
    paid = table["REPAYS_AMOUNT"].to_numpy()
    borN = table["BORROWER_COUNT"].to_numpy()
    paidN = table["REPAYER_COUNT"].to_numpy()


    col1, col2= st.columns(2)
    totbor = 5.4219e9

    col1.metric("Total borrowed", millify(totbor, precision=2) + " UST", "{:0.2f} %".format( ((bor[-1] - paid[-1]) / totbor) * 100))
    col2.metric("Positions above 1000 UST", "xxx", "+0%")

    st.write("Note: values above are placeholders. Also, should replace right side below by tracking actual positions, not what is done now which just counts transaction numbers")


    x = table["WEEK"].apply(lambda x: pd.Timestamp(x).date())

    start = st.slider("Start date", x.iloc[0], x.iloc[-1], x.iloc[0])
    end = st.slider("End date", start, x.iloc[-1], x.iloc[-1])


    cut = (x <= end) & (x >= start)
    x = x[cut]
    bor = bor[cut]
    paid = paid[cut]
    borN = borN[cut]
    paidN = paidN[cut]
    #  col2.write(table)

    if st.checkbox("Show data table"):
        st.write(table)

    col1, col2 = st.columns([3,3])
    
   
    cumu = np.cumsum(bor-paid)*100
    cumu /= 5.5e9

    #  plotdata = pd.DataFrame({"Date": x, "percent": cumu})
    #  chart = (
    #      alt.Chart(
    #          data=plotdata,
    #          title="Cumulative change of borrowed amount",
    #      )
    #      .mark_line().encode(x="Date", y="percent"))
    #
    #  st.altair_chart(chart)

   
    fig, ax0 = plt.subplots(1, figsize=(6,4))

    ax0.plot(x, cumu)
    ax0.plot(x, np.zeros(len(x)), "k--")
    ax0.set_title("Cumulative change of borrowed amount")
    ax0.set_ylabel("percent")
    ax0.grid()
    ax0.set_xticklabels(x, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax0.xaxis.set_major_formatter(formatter)
    
    col1.pyplot(fig)

    fig, ax1 = plt.subplots(1, figsize=(6,4))

    ax1.bar(x, -paid, width=2)
    ax1.bar(x, bor, width=2)
    ax1.legend(["Repaid", "Borrowed"],prop={'size': 8}) 
    ax1.set_ylabel("UST")
    ax1.set_title("Delta UST")
    ax1.grid()
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax1.xaxis.set_major_formatter(formatter)

    col1.pyplot(fig)

    fig, ax2 = plt.subplots(1, figsize=(6,4))

    ax2.bar(x, bor-paid, width=2)
    ax2.grid()
    ax2.set_ylabel("UST")
    ax2.set_title("Net Delta UST")
    ax2.set_xticklabels(x, rotation=45, ha='right')
    ax2.xaxis.set_major_formatter(formatter)
    #  locator = mdates.DayLocator()
    #  ax0.xaxis.set_major_locator(locator)

    col1.pyplot(fig)

    cumu = np.cumsum(borN-paidN)
    fig, ax0 = plt.subplots(1, figsize=(6,4))

    ax0.plot(x, cumu)
    ax0.plot(x, np.zeros(len(x)), "k--")
    ax0.set_title("Cumulative signed borrow/paid transaction number")
    #  ax0.set_ylabel("")
    ax0.grid()
    ax0.set_xticklabels(x, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax0.xaxis.set_major_formatter(formatter)
    
    col2.pyplot(fig)

    fig, ax1 = plt.subplots(1, figsize=(6,4))

    ax1.bar(x, -paidN, width=2)
    ax1.bar(x, borN, width=2)
    ax1.legend(["Repaid", "Borrowed"],prop={'size': 8}) 
    ax1.set_title("Delta Transactions")
    ax1.grid()
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax1.xaxis.set_major_formatter(formatter)

    col2.pyplot(fig)

    fig, ax2 = plt.subplots(1, figsize=(6,4))

    ax2.bar(x, borN-paidN, width=2)
    ax2.grid()
    ax2.set_title("Net Delta Transactions")
    ax2.set_xticklabels(x, rotation=45, ha='right')
    ax2.xaxis.set_major_formatter(formatter)
    #  locator = mdates.DayLocator()
    #  ax0.xaxis.set_major_locator(locator)

    col2.pyplot(fig)
#  chart_data = pd.DataFrame(
#       np.random.randn(50, 3),
#       columns=["a", "b", "c"])
#
#  st.bar_chart(chart_data)
