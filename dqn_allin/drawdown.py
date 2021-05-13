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



fig = plt.figure()

plt.xticks(rotation=45, ha="right", rotation_mode="anchor") #rotate the x-axis values
plt.subplots_adjust(bottom = 0.2, top = 0.9) #ensuring the dates (on the x-axis) fit in the screen
plt.ylabel('Drawdown (%)')
plt.xlabel('Days')
plt.title(stock_name)


def max_dd(DF):
    df = DF.copy()
    try:
        df['daily_ret'] = df['Portfolio value'].pct_change()
    except:
        df['daily_ret'] = df['Adj Close'].pct_change()
    df['cumulative_ret'] = (1 + df['daily_ret']).cumprod()
    df['cum_roll_max'] = df['cumulative_ret'].cummax()
    df['drawdown'] = df['cum_roll_max'] - df['cumulative_ret']
    df['drawdown_pct'] = df['drawdown'] / df['cum_roll_max']
    return df['drawdown_pct'] * -100

dd =max_dd(df1)[1:]


def buildmebarchart(i=int):

    plt.fill_between(range(len(dd[:i])), dd[:i], 0, color='pink')
    plt.plot(dd[:i].values, c='red', lw=1.5, )



animator = ani.FuncAnimation(fig, buildmebarchart, interval=20, frames=len(dd))
animator.save(f"anime/drawdown/{stock_name}.mp4", fps=30)
#plt.show()

