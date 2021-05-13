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

plt.style.use("dark_background")

df1 = pd.read_csv(f'data/{stock_name}.csv', parse_dates=True)
df1.set_index('Date', inplace=True)
df1 = df1[['stock_price', 'marker_buy', 'marker_sell', 'portfolio_value']]
df1.columns = ['Stock price', 'Buy', 'Sell', 'Portfolio value']


fig = plt.figure()

plt.xticks(rotation=45, ha="right", rotation_mode="anchor") #rotate the x-axis values
plt.subplots_adjust(bottom = 0.2, top = 0.9) #ensuring the dates (on the x-axis) fit in the screen
plt.ylabel('Price')
plt.xlabel('Days')
plt.title(stock_name)


def buildmebarchart(i=int):
    plt.legend(df1.columns[:-1], loc=3)
    #
    s = i-100
    if s < 0:
        s = 0
    plt.plot(df1['Stock price'][:i].values, color='#03fff7', lw='1.5', alpha=1);
    plt.plot(df1['Buy'][:i].values, ls=' ', mfc='#00ff04', lw='2', alpha=1, marker='o', ms=7);
    plt.plot(df1['Sell'][:i].values, ls=' ', mfc='red', lw='2', alpha=1, marker='o', ms=7);
    plt.xlabel(f'Days\n\nPortfolio value: {df1["Portfolio value"].values[i] : 8.2f}     Profit: {(df1["Portfolio value"].values[i] - 1000)*100/1000:8.2f} %')
    plt.xlim([s, i+20])


animator = ani.FuncAnimation(fig, buildmebarchart, interval=200, frames=len(df1))
#animator.save(f"anime/{stock_name}.mp4", fps=30)
plt.show()

