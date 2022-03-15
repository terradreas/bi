import json
from functools import cmp_to_key
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#
#
#  WITH
#    liqs as (
#    select
#    date_trunc(second, block_timestamp) as mytime,
#    tx_id as id,
#    event_attributes::string as text
#    from terra.msg_events
#    where tx_status = 'SUCCEEDED'
#    and event_type = 'from_contract'
#    and event_attributes::string like '%liquidate_collateral%'
#    and (event_attributes::string like '%terra1e25zllgag7j9xsun3me4stnye2pcg66234je3u%' -- Anchor Liquidation queue
#    or event_attributes::string like '%terra1w9ky73v4g7v98zzdqpqgf3kjmusnx4d4mvnac6%') -- Anchor Liquidation queue legacy
#    and block_timestamp >= '2021-01-01'
#    )
#  select
#    mytime as t,
#    id,
#    concat('\"', replace(text, '"', ''''), '\"') as events
#    from liqs
#  order by 1 asc


print("Loading data...")

pathLiq = 'tables/anchor_liq.csv'
pathBor = 'tables/anchor_users_borrow_weekly.csv'
pathBor = 'tables/anchor_users_borrow_daily.csv'
pathPay = 'tables/anchor_users_repay_weekly.csv'
pathPay = 'tables/anchor_users_repay_daily.csv'
data_file_delimiter = ','

liq = pd.read_csv(pathLiq)

borrow = pd.read_csv(pathBor)
#  borrow = pd.read_excel(pathBor)
borrow = borrow.astype({'ADDRESS': 'string'})
borrow.set_index("ADDRESS", inplace=True)
repay = pd.read_csv(pathPay)
repay = repay.astype({'ADDRESS': 'string'})
repay.set_index("ADDRESS", inplace=True)


print("Processing data...")
# remove quotes around date 
borrow.columns = [c[1:11] for c in borrow.columns]
repay.columns = [c[1:11] for c in repay.columns]

# yyyy-mm-dd strings - this works
mindate = min(borrow.columns[0], repay.columns[0])
maxdate = max(borrow.columns[-1], repay.columns[-1])

borrow = borrow[ borrow.columns[(borrow.columns >= mindate) & (borrow.columns <= maxdate)] ]
repay =  repay[ repay.columns[(repay.columns >= mindate) & (repay.columns <= maxdate)] ]

# now both columns lists are identical, use either

#  interestFac = pd.DataFrame(np.ones((1,len(borrow.columns))), columns=borrow.columns)
interestFac = pd.Series(np.linspace(1,1.2,len(borrow.columns)), index=borrow.columns)
#  interestFac = pd.Series(np.linspace(1,1,len(borrow.columns)), index=borrow.columns)

#  print(borrow.head())
#  print(repay.head())
#  print()


users = borrow.copy()
#  users.set_index("ADDRESS", inplace=True)
print(users.head(40))
# this subtracts repays where available. Also, all values are converted to a kind of "aUST" that appreciates wrt to UST due to interest
# this is the quantity that can be added and subtracted and cumulative-summed
# at the end, we will multiply by interest again

print("Just borrow")
print(users.cumsum(axis=1).head(40))
print(users.cumsum(axis=1).tail(40))

def combine(r1):
    try:
        r2 = repay.loc[r1.name]
    except KeyError:
        return r1 / interestFac

    return (r1 - r2) / interestFac

users = users.apply(lambda x: combine(x), axis=1)

print("Minus repay")
print(users.cumsum(axis=1).mul(interestFac, axis=1).head(40))
print(users.cumsum(axis=1).mul(interestFac, axis=1).tail(40))
print("Negatives: ", ((users.cumsum(axis=1).mul(interestFac, axis=1))['2022-03-10'] < 0 ).sum() ) 

# the structure is that there is a liquidate_collateral action followed by borrower (among others), 
# then always an execute_bid action followed by irrelevant repay_amount (among others), 
# then repay_stable with both borrower (same again) and repay_amount -- we want the latter pair! 
# these are the only occurances of repay_amount and borrower
# there are other actions in between so the number in front of X_action key is irrelevant here
# there can be the first part (execute_bid and repay_stable followed by stuff each) for each collateral before a single repay_stable!

# we count liquidate_collateral actions and execute_bid actions once we see a repay_stable action we 


#  for i in np.arange(len(liq.index)-1, 0, -1):
#      tx = str(liq.ID[i])
#      print(tx)
#      req = Request("https://lcd.terra.dev/txs/" + tx, headers={'User-Agent': 'Mozilla/5.0'})
#
#      try:
#          with urlopen(req) as dat:
#              data = json.loads(dat.read().decode())
#      except HTTPError:
#          continue
        #  print(data.keys())
        #  print(data)
#
#      print(data)
#      for ev in data['logs'][0]['events']:
#          if ev['type'] == 'from_contract':
#              events = ev['attributes']
#              break
#
#      read = False
#      liqs = []
#      for i,ev in enumerate(events):
#          print(i, ev)
#          key = ev['key']
#          value = ev['value']
#
#          if 'repay_stable' in value: # action: repay_stable. then: borrower, repay_amount
#              read = True
#              print("Reading")
#          if 'borrower' in key and read:
#              address = value
#              print("Set borrower")
#          if 'repay_amount' in key and read:
#              amount = value
#              print("Set value")
#              read = False
#              liqs += [(address, amount)]
#      print(liqs)

print("Parsing {} liquidation events...".format(len(liq.index)-1))

isolatedLiqs = []
nLiqsOutside = 0

for i in np.arange(len(liq.index)-1,0,-1):
    #  print(i, liq['ID'][i])
    t = json.loads(liq['EVENTS'][i].replace('\'', '\"'))
    time = liq['T'][i]
    time = time[:10] 
    index = 0
    index2 = 0
    l = []

    # make a list first, to keep track of ordering
    for key,value in t.items():
        l += [(key, value)]

    # sorting properly, that is, 11 should come after 2 and not before as in present alphabetical ordering - convert string bit to number if present, ignore rest
    def com(lhs, rhs):
        a = lhs[0].split('_',1)[0]
        b = rhs[0].split('_',1)[0]
        try: 
            a = int(a)
        except ValueError:
            return 99999
        try: 
            b = int(b)
        except ValueError:
            return -99999
        return a-b

    l = sorted(l, key=cmp_to_key(com))
   

    # we care about 3 types of action and their order. there can be several liquidate_collaterals and execute_bid pairs for multiple collateral liqs
    # each pair brings a repay_amount event which we do not care about - these are skipped by incrementing the index for each pair
    # (thanks to sorting, these 3 actions are in the correct order because they all have an xx_action key that tracks it!)
    # We are interested in the repay_amount keys (with an amount value) that come after the list of pairs whenever there is a repay_stable action

    for (key, value) in l:
        if not isinstance(value, str):
            continue
        if 'liquidate_collateral' in value:
            index += 1 # one index  ...
        if 'execute_bid' in value:
            index2 += 1 # ... and another one -- since we expect pairs, they should catch up ...
        if 'repay_stable' in value: # ... by this time 
            if index != index2:
                # under the assumptions made by this parser this does not happen
                print("Indices do not agree")
                raise Exception("Indices do not agree")
            else:
                address = t[str(index)+"_borrower"]
                amount = t[str(index)+"_repay_amount"]*1e-6

                if time >= mindate and time <= maxdate:
                    isolatedLiqs += [(time, address, amount)]
                else: 
                    nLiqsOutside += 1

    if (i % int( len(liq.index) / 100) )== 0:
        print('#',end='', flush=True)

print()
print("Done.")
print("{} user liquidations extracted. {} liquidations were outside the borrow/repay date overlap".format(len(isolatedLiqs), nLiqsOutside))
print()
print("Processing liquidations...")

for i,l in enumerate(isolatedLiqs):
    try:
        #  print("yo")
    #  print(l[0])
    #  print(users.loc[l[1]])

        users.loc[l[1],l[0]] -= l[2]/interestFac[l[0]]
    except KeyError:
        nLiqsOutside += 1

    if (i % int( len(isolatedLiqs) / 100) )== 0:
        print('#',end='', flush=True)

print()
print()


users = users.cumsum(axis=1).mul(interestFac, axis=1)

users.to_csv("anchor_users_positions")
print(users.cumsum(axis=1).mul(interestFac, axis=1).tail(40))
print("Negatives: ", ((users.cumsum(axis=1).mul(interestFac, axis=1))['2022-03-10'] < 0 ).sum() ) 
print("Done.")
print("{} liquidations were outside the borrow/repay date overlap or had unknown wallet address".format(nLiqsOutside))

