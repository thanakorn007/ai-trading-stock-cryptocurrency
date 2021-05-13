import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pandas as pd
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name')
args = parser.parse_args()

stock_name = args.name.upper()

sns.set_style('whitegrid')

df1 = pd.read_csv(f'data/{stock_name}.csv', parse_dates=True)
df1.set_index('Date', inplace=True)
df1 = df1[['stock_price', 'marker_buy', 'marker_sell', 'portfolio_value']]
df1.columns = ['Stock price', 'Buy', 'Sell', 'Portfolio value']
bt_returns = df1['Portfolio value'].pct_change().cumsum() + 1
price = df1['Stock price'].pct_change().cumsum() + 1
bt_returns = bt_returns.fillna(1)
price = price.fillna(1)


fig = plt.figure()

plt.xticks(rotation=45, ha="right", rotation_mode="anchor") #rotate the x-axis values
plt.subplots_adjust(bottom = 0.2, top = 0.9) #ensuring the dates (on the x-axis) fit in the screen
plt.ylabel('Cumulative return')
plt.xlabel('Days')
plt.title(stock_name)


def buildmebarchart(i=int):
    plt.legend(['Backtest', 'Price'], loc=3)

    plt.plot(price.values[:i], lw=3, c='#bab8b8', label='Price')
    plt.plot(bt_returns.values[:i], lw=3, c='#069c24', label='Backtest')
    plt.axhline(1, ls='--', c='black', lw=2)



animator = ani.FuncAnimation(fig, buildmebarchart, interval=20, frames=len(df1))
animator.save(f"anime/compare/{stock_name}.mp4", fps=30)
#plt.show()

