import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib import cm
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


@st.cache
def download_data():
    URL1 = 'https://api.flipsidecrypto.com/api/v2/queries/08c78cad-d2a6-4fa8-89cd-50884e55a389/data/latest'

    URL_CEX = 'https://api.flipsidecrypto.com/api/v2/queries/22362755-62b1-4a3a-92dc-3bf67aa9ea87/data/latest'
    URL_BRIDGE = 'https://api.flipsidecrypto.com/api/v2/queries/672a89ad-3485-4a2c-b437-cea1c65afca9/data/latest'
    URL_WALLETS = 'https://api.flipsidecrypto.com/api/v2/queries/8825fb7f-4bf7-4a7d-b00e-21c684b12d6c/data/latest'


    cex = pd.read_json(URL_CEX)
    bridge = pd.read_json(URL_BRIDGE)
    wallets = pd.read_json(URL_WALLETS)
    #  data["REPAYS_AMOUNT"] = -data["REPAYS_AMOUNT"]
    #  lowercase = lambda x: str(x).lower()
    #  data.rename(lowercase, axis='columns', inplace=True)
    return cex, bridge, wallets

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO

def google_drive(id:str) -> pd.DataFrame:
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_file(
        "service-account.json",
        scopes=SCOPES,
    )
    service = build(
        "drive",
        "v3",
        credentials=creds,
        cache_discovery=False,
    )
    request = service.files().get_media(fileId=id)
    file = BytesIO()
    downloader = MediaIoBaseDownload(file,request)
    done = False
    while done is False: _,done = downloader.next_chunk()
    file.seek(0)
    return pd.read_csv(file)

@st.cache
def load_data():
    #  path1 = 'tables/anchor_users_borrow_weekly.csv'
    #  anchBor = pd.read_csv(path1)
    #  anchBor.set_index("ADDRESS", inplace=True)
    #  path2 = 'tables/anchor_users_repay_weekly.csv'
    #  anchPay = pd.read_csv(path2)
    #  anchPay.set_index("ADDRESS", inplace=True)

    tables = [("anch", "anchor_users_positions.csv", "1CEQN35wh6imQeuXM_LSWsI3uZ5UL0etp")]
    #  from util import load_gdrive
    #  load_gdrive(tables[0][2],tables[0][1])   #  return anchBor, anchPay
    #  path = 'tables/anchor_users_positions.csv'
    #  anch = pd.read_csv(path)
    #  anch.set_index("ADDRESS", inplace=True)
    #  return anch

    from pathlib import Path
    from os.path import join

    data = []
    for t in tables:
        dest = Path('tables')
        dest.mkdir(exist_ok=True)
    
        dest = dest / t[1]

        if not dest.exists():
            with st.spinner("Downloading table " + t[1]):
            
                data += [ google_drive(t[2]) ]
                data[-1].to_csv(str(dest), index=False)

        else:
            data += [pd.read_csv(str(dest))]

    #  url = 'https://drive.google.com/file/d/1CEQN35wh6imQeuXM_LSWsI3uZ5UL0etp/view?usp=sharing'
    #  path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    #  anch = pd.read_csv(path)
    data[0].set_index("ADDRESS", inplace=True)


    cex, bridge, wallets = download_data()

    cex = cex.sort_values("MONTH")

    bridge = bridge.sort_values("MONTH")

    m = cex['CEX_LABEL'] == 'binance'
    cex.loc[m, 'CEX_LABEL'] = 'Binance'
    m = cex['CEX_LABEL'] == 'kucoin'
    cex.loc[m, 'CEX_LABEL'] = 'Kucoin'

    #  wallets.reset_index(inplace=False)
    #  wallets.columns = ['foo', 'bsar']
    #  wallets.index.name = 'Day'
    wallets.rename({'MIN_TIMESTAMP':'Day', 'WALLET_COUNT':'New wallets from CEX'}, axis=1, inplace=True)
    wallets['Day'] = wallets['Day'].map(lambda x: pd.Timestamp(x).date())
    wallets.set_index("Day", inplace=True)

    return data[0], cex, bridge, wallets
#  table = 0
#  anchBor = 0
#  anchPay = 0
anch = 0
cex = 0
bridge = 0
wallets = 0

def init():
    #  global table
    #  global anchBor
    #  global anchPay
    global anch
    global cex
    global bridge
    global wallets
    #  table = download_data()
    #  anchBor, anchPay = load_data()
    anch, cex, bridge, wallets = load_data()

     #  = cex[cex['CEX_LABEL'] == binance]
    #  st.write(binance)
    #  st.write(anch)

init()

#  data_load_state = st.text('Loading data...')
if 'section' not in st.session_state:
    st.session_state['section'] = 'flows'

def click_flows(): 
    st.session_state["section"] = "flows"
def click_anchor(): 
    st.session_state["section"] = "anchor"
def click_misc(): 
    st.session_state["section"] = "misc"

buttonFlows = st.sidebar.button("Flows in/out Terra", on_click = click_flows)#, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)
buttonCDP = st.sidebar.button("Anchor", on_click = click_anchor)
buttonMisc = st.sidebar.button("Misc", on_click = click_misc)

if st.session_state["section"] == "misc":
    st.title('Miscellaneous')

    st.write("#### First-time transfers from CEX to new wallet (LUNA+UST)")

    st.bar_chart(wallets)

if st.session_state["section"] == "flows":
    st.title('Flows')

    width = 10

    coin = st.radio(
     "",
     ('UST', 'LUNA', 'both'))


    col1, col2 = st.columns([3,3])
    col1.write("### CEX")
    col2.write("### Bridges")

    st.write("#### Inflows/Outflows into/from Terra (positive/negative, in USD)")
    
    col1, col2 = st.columns([3,3])

    exchangeLabels = ['Binance', 'Kucoin', 'Others']
    exchangeColors = ['#E8Ba41', '#52AC92', 'red']

    x0 = np.array( [pd.Timestamp(x).date() for x in cex.groupby('MONTH').sum().index.tolist() ] )

    fig, ax = plt.subplots(1, figsize=(6,4))
    for inout in [1.,-1.]:

        # bottom of bar plot: reset after doing one sign 
        y0 = np.zeros(x0.size)
        for ex, col in zip(exchangeLabels, exchangeColors):

            #  m = pd.Series(True, index=cex.index)
            m = (cex['CAPITAL_FLOW_AMOUNT']*inout > 0)

            if coin != 'both':
                m = (m & (cex['TYPE'] == coin))

            if ex == 'Others':
                # exclude all but last in list (last = Others)
                for ex2 in exchangeLabels[:-1]:
                    m = (m & (cex['CEX_LABEL'] != ex2))
            else:
                m = m & (cex['CEX_LABEL'] == ex)

            data = cex[m].groupby('MONTH', as_index=False)['CAPITAL_FLOW_AMOUNT'].sum()

            xid = np.array( [ np.argwhere(x0 == pd.Timestamp(x).date())[0][0] for x in data.MONTH] )
            y = np.zeros(x0.size)
            y[xid] = data['CAPITAL_FLOW_AMOUNT'].to_numpy()

            
            ax.bar(x0, y, width=width, bottom=y0, color=col)

            y0 += y

    ax.legend(exchangeLabels)
    ax.plot(x0, y*0, 'k--', lw=1)
    ax.set_xticklabels(x0, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.grid()
    col1.pyplot(fig)


    x0 = np.array( [pd.Timestamp(x).date() for x in bridge.groupby('MONTH').sum().index.tolist() ] )
    bridgeLabels = ['Wormhole', 'Shuttle']
    bridgeColors = ['#BF2952', '#6092F0']

    fig, ax = plt.subplots(1, figsize=(6,4))
    for inout in [1.,-1.]:

        # bottom of bar plot: reset after doing one sign 
        y0 = np.zeros(x0.size)
        for ex, col in zip(bridgeLabels, bridgeColors):

            #  m = pd.Series(True, index=cex.index)
            m = (bridge['CAPITAL_FLOW_AMOUNT']*inout > 0)

            if coin != 'both':
                m = (m & (bridge['CURRENCY'] == coin))

            m = m & (bridge['BRIDGE'] == ex)

            data = bridge[m].groupby('MONTH', as_index=False)['CAPITAL_FLOW_AMOUNT'].sum()

            xid = np.array( [ np.argwhere(x0 == pd.Timestamp(x).date())[0][0] for x in data.MONTH] )
            y = np.zeros(x0.size)
            y[xid] = data['CAPITAL_FLOW_AMOUNT'].to_numpy()

            
            ax.bar(x0, y, width=width, bottom=y0, color=col)

            y0 += y

    ax.legend(bridgeLabels)
    ax.plot(x0, y*0, 'k--', lw=1)
    ax.set_xticklabels(x0, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.grid()
    col2.pyplot(fig)



    st.write("#### Accumulated (total) net flows (positive: into Terra, negative: leaving Terra), in USD")
    
    col1, col2 = st.columns([3,3])

    fig, ax = plt.subplots(1, figsize=(6,4))

    for i, (ex, col) in enumerate(zip(exchangeLabels, exchangeColors)):

        y = np.zeros(x0.size)

        m = pd.Series(True, index=cex.index)

        if coin != 'both':
            m = (m & (cex['TYPE'] == coin))

        if ex == 'Others':
            # exclude all but last in list (last = Others)
            for ex2 in exchangeLabels[:-1]:
                m = (m & (cex['CEX_LABEL'] != ex2))
        else:
            m = m & (cex['CEX_LABEL'] == ex)

        data = cex[m].groupby('MONTH', as_index=False)['CAPITAL_FLOW_AMOUNT'].sum()

        xid = np.array( [ np.argwhere(x0 == pd.Timestamp(x).date())[0][0] for x in data.MONTH] )
        y[xid] = np.cumsum(data['CAPITAL_FLOW_AMOUNT'].to_numpy())

        
        ax.bar(x0+pd.tseries.offsets.DateOffset(8*i, 'day'), y, width=width, color=col)

    ax.legend(exchangeLabels)
    ax.plot(x0, y*0, 'k--', lw=1)
    ax.set_xticklabels(x0, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.grid()
    col1.pyplot(fig)


    fig, ax = plt.subplots(1, figsize=(6,4))

    for i, (ex, col) in enumerate(zip(bridgeLabels, bridgeColors)):

        y = np.zeros(x0.size)

        m = pd.Series(True, index=bridge.index)

        if coin != 'both':
            m = (m & (bridge['CURRENCY'] == coin))

        m = m & (bridge['BRIDGE'] == ex)

        data = bridge[m].groupby('MONTH', as_index=False)['CAPITAL_FLOW_AMOUNT'].sum()

        xid = np.array( [ np.argwhere(x0 == pd.Timestamp(x).date())[0][0] for x in data.MONTH] )
        y[xid] = np.cumsum(data['CAPITAL_FLOW_AMOUNT'].to_numpy())

        
        ax.bar(x0+pd.tseries.offsets.DateOffset(8*i, 'day'), y, width=width, color=col)

    ax.legend(bridgeLabels)
    ax.plot(x0, y*0, 'k--', lw=1)
    ax.set_xticklabels(x0, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    ax.grid()
    col2.pyplot(fig)
############ ANCHOR ##############

if st.session_state["section"] == "anchor":
    st.title('Anchor')
   
    pos = anch.to_numpy(copy=True)

    x = np.array( [ pd.Timestamp(x).date() for x in anch.columns] )


    st.write("##### Position size considered (10^x UST)")
    col1, col2 = st.columns([3,3])
    minsize = col1.slider("Cut shrimps (choose x)", 0, 11, -2)
    #  end = st.slider("End date", start, x.iloc[-1], x.iloc[-1])
    maxsize = col2.slider("Cut whales (choose x)", minsize, 10, 11)

    st.write("##### Choose time window")
    col1, col2 = st.columns([3,3])

    start = col1.slider("Start date", x[0], x[-1], x[0])
    #  end = st.slider("End date", start, x.iloc[-1], x.iloc[-1])
    end = col2.slider("End date", start, x[-1], x[-1])


    mask = (pos > 10**minsize) & (pos < 10**maxsize)

    mask = mask.astype(int)

    #  st.write(pd.DataFrame(pos))
    # from left to right, carry over from left and add 1 if 1, and reset to 0 else (carry over 0 and add 0 is zero)
    # this counts age of position in days starting at 1 every time
    #  pos = np.ones(pos.shape)
    for i in np.arange(1,mask.shape[1]):
        
        mask[:,i] += mask[:,i]*mask[:,i-1]

    datecut = (x <= end) & (x >= start)
    x = x[datecut]
    pos = pos[:,datecut]
    mask = mask[:,datecut]
    

    #### POSITIONS (SIZE) OVER TIME ####

    st.write("##### Heatmap")
    nSizeRes = 400
    positionplot = np.zeros((nSizeRes,pos.shape[1]))

    for i in np.arange(positionplot.shape[1]):
        #  st.write(pos[mask[:,i]>=1,i])
        indices = (np.log(pos[mask[:,i]>=1,i]) - np.log(10**minsize))/(np.log(10**maxsize)-np.log(10**minsize))*nSizeRes
        indices = indices.astype(int)
        indices = indices[indices<nSizeRes]
        np.add.at(positionplot[:,i], indices, 1)

    positionplot = (positionplot)**0.2
    fig, ax = plt.subplots(1, figsize=(6,4))
    pl = ax.pcolormesh(x, np.geomspace(10**minsize, 10**maxsize, nSizeRes), positionplot, cmap=cm.inferno)
    ax.set_ylabel('Dept position size (UST)')
    ax.set_yscale('log')
    ax.set_xticklabels(anch.columns, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)

    st.pyplot(fig)

    #### AGE OF POSITIONS ####

    st.write("##### Full tracking of positions by age")
    fig, ax0 = plt.subplots(1, figsize=(6,4))

    cuts = [14, 30, 60, 90, 180]
    plots = []
    left = 1
    counts = st.checkbox('Position counts (instead UST of amount)', value=True)
    for c in cuts:
        right = c
        maskage = (mask >= left) & (mask < right)
        if counts:
            plots += [np.sum(maskage, 0)]
        else:
            plots += [np.sum(pos*maskage, 0)]
        left = right

    maskage = mask >= cuts[-1]
    if counts:
        plots += [np.sum(maskage,0)]
    else:
        plots += [np.sum(pos*maskage,0)]
    
    cuts.insert(0, 1)
    reverseStacking = st.checkbox('Reverse stacking')
    if not reverseStacking:
        plots = plots[-1::-1]
        cuts = cuts[-1::-1]

    plotsum = np.zeros(plots[0].shape)
    
    for i,(p,c) in enumerate(zip(plots, cuts)):
        #  ax0.bar(anch.columns, p, bottom = plotsum)
        col = i/(len(plots)-1.)
        if reverseStacking:
            col = 1 - col
        col = cm.inferno(col)
        ax0.bar(x, p, bottom = plotsum, label= "at least " + str(c) + " days", color=col)
        plotsum += p
    ax0.legend(loc='best')
    #  every_nth = int(len(anch.columns)/10)
    #  for n, label in enumerate(ax0.xaxis.get_ticklabels()):
    #      if n % every_nth != 0:
    #          label.set_visible(False)
    ax0.set_xticklabels(anch.columns, rotation=45, ha='right')
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax0.xaxis.set_major_formatter(formatter)

    st.pyplot(fig)


    ##################################

    #  ageplot = np.zeros((mask.max(),pos.shape[1]))

    #  for i in np.arange(ageplot.shape[1]):
    #      indices = mask[:,i]-1
    #      indmask = indices >= 0
    #      indices = indices[indmask]
    #      if counts:
    #          np.add.at(ageplot[:,i], indices, 1)
    #      else:
    #          np.add.at(ageplot[:,i], indices, pos[indmask,i])
    #  if not counts:
    #      m = ageplot > 0
    #      ageplot[m] = np.log(ageplot[m])/np.log(10)
    #

    #  pl = ax.pcolormesh(x, np.arange(1,ageplot.shape[0]+1), ageplot, cmap= cm.inferno)
    #  ax.set_ylabel('Position age (days)')
    #  fig.colorbar(pl)


    st.write("(Not final results. The borrow rate computation used behind the scene here is not yet accurate.)") 


