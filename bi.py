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
    data = pd.read_json(URL1)
    data = data.sort_values("WEEK")
    #  data["REPAYS_AMOUNT"] = -data["REPAYS_AMOUNT"]
    #  lowercase = lambda x: str(x).lower()
    #  data.rename(lowercase, axis='columns', inplace=True)
    return data

@st.cache
def load_data():
    #  path1 = 'tables/anchor_users_borrow_weekly.csv'
    #  anchBor = pd.read_csv(path1)
    #  anchBor.set_index("ADDRESS", inplace=True)
    #  path2 = 'tables/anchor_users_repay_weekly.csv'
    #  anchPay = pd.read_csv(path2)
    #  anchPay.set_index("ADDRESS", inplace=True)
    #  return anchBor, anchPay
    path = 'tables/anchor_users_positions.csv'
    anch = pd.read_csv(path)
    anch.set_index("ADDRESS", inplace=True)
    return anch

#  table = 0
#  anchBor = 0
#  anchPay = 0
anch = 0

def init():
    #  global table
    #  global anchBor
    #  global anchPay
    global anch

    #  table = download_data()
    #  anchBor, anchPay = load_data()
    anch = load_data()


init()

#  data_load_state = st.text('Loading data...')
if 'section' not in st.session_state:
    st.session_state['section'] = 'flows'

def click_flows(): 
    st.session_state["section"] = "flows"
def click_anchor(): 
    st.session_state["section"] = "anchor"

buttonBasic = st.sidebar.button("Flows in/out Terra", on_click = click_flows)#, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)
buttonCDP = st.sidebar.button("Anchor", on_click = click_anchor)


if st.session_state["section"] == "flows":
    st.title('Flows')

    col1, col2 = st.columns([3,3])
    col1.write("#### CEX")
    col2.write("#### Bridges")

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

    st.write("##### Full tracking of position and their ages")
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

