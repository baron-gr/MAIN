import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fredapi import Fred
import time
plt.style.use('fivethirtyeight')
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

## Create Fred object
API_KEY = '576b41c7b0e5601263fca2ad59bd6817'
fred = Fred(api_key = API_KEY)

## Search for economic data
sp_search = fred.search('S&P', order_by = 'popularity')

## Pull raw data & plot
sp500 = fred.get_series('SP500')
# sp500.plot(figsize=(10,5), title="S&P 500", lw=2)

## Pull & join multiple datasets
unemp_results = fred.search('unemployment')
unrate = fred.get_series('UNRATE')

unemp_df = fred.search('unemployment rate state', filter=('frequency','Monthly'))
unemp_df = unemp_df.query('seasonal_adjustment == "Seasonally Adjusted" and units == "Percent"')
unemp_df = unemp_df.loc[unemp_df['title'].str.contains('Unemployment Rate')]

all_results = []
for myid in unemp_df.index:
    results = fred.get_series(myid)
    results = results.to_frame(name=myid)
    all_results.append(results)
    time.sleep(0.1)
uemp_results = pd.concat(all_results, axis=1)

cols_to_drop = []
for i in uemp_results:
    if len(i) > 4:
        cols_to_drop.append(i)
uemp_results = uemp_results.drop(columns = cols_to_drop, axis=1)

uemp_states = uemp_results.copy()
uemp_states = uemp_states.dropna()

# Using list comprehension to map ids
id_to_state = unemp_df['title'].str.replace('Unemployment Rate in ','').to_dict()
uemp_states.columns = [id_to_state[c] for c in uemp_states.columns]

## Plot States Unemployment Rates
# fig = px.line(uemp_states)
# fig.show()

## Pull May 2020 Unemployment Rate per State
# ax = uemp_states.loc[uemp_states.index == '2020-05-01'].T \
#     .sort_values('2020-05-01') \
#     .plot(kind='barh', figsize=(8,12), width=0.7, edgecolor='black' \
#           , title = 'Unemployment Rate by State, May 2020')
# ax.legend().remove()
# ax.set_xlabel('% Unemployed')
# plt.show()

## Pull Participation Rate
part_df = fred.search('participation rate state', filter=('frequency','Monthly'))
part_df = part_df.query('seasonal_adjustment == "Seasonally Adjusted" and units == "Percent"')
part_id_to_state = part_df['title'].str.replace('Labor Force Participation  Rate for ','').to_dict()

all_part_results = []
for myid in part_df.index:
    part_results = fred.get_series(myid)
    part_results = part_results.to_frame(name=myid)
    all_part_results.append(part_results)
    time.sleep(0.1)
part_states = pd.concat(all_part_results, axis=1)
part_states.columns = [part_id_to_state[c] for c in part_states.columns]

# Fix DC
uemp_states = uemp_states.rename(columns={'the District of Columbia':'District Of Columbia'})

## Plot Unemployment vs Participation Rate
fig, axs = plt.subplots(10, 5, figsize=(30, 30), sharex=True)
axs = axs.flatten()

index = 0
for state in unemp_df.columns:
    if state in ["District Of Columbia","Puerto Rico"]:
        continue
    ax2 = axs[i].twinx()
    uemp_states.query('index == 2020 and index < 2020')[state] \
        .plot(ax=axs[i], label='Unemployment')
    part_states.query('index == 2020 and index < 2020')[state] \
        .plot(ax=ax2, label='Participation', color=color_pal[1])
    ax2.grid(False)
    axs[i].set_title(state, fontsize=8)
    index += 1
plt.tight_layout()
plt.show()

# state = 'California'
# fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
# ax2 = ax.twinx()
# uemp_states2 = uemp_states.asfreq('MS')
# l1 = uemp_states2.query('index >= 2020 and index < 2022')[state] \
#     .plot(ax=ax, label='Unemployment')
# l2 = part_states.dropna().query('index >= 2020 and index < 2022')[state] \
#     .plot(ax=ax2, label='Participation', color=color_pal[1])
# ax2.grid(False)
# ax.set_title(state)
# fig.legend(labels=['Unemployment','Participation'])
# plt.show()