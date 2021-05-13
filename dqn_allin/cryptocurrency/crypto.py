from model import DQNAgent, SingleStockEnv
import torch
import pandas_datareader as web
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, types



stock_name = 'btc-usd'.upper()
industry = 'Cryptocurrentcy'

engine = create_engine('mysql://root:@localhost/dqn_crypto_'+stock_name.lower().split('-')[0])

stock_data = pd.read_csv('data/'+stock_name+'.csv', parse_dates=True)
stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])
stock_data.set_index('Datetime', inplace=True)

n_train = len(stock_data)//2
train_data = stock_data.iloc[:n_train]
test_data = stock_data.iloc[n_train:]



print(f'Stock name: {stock_name}')
print(f'Start: {stock_data.index[0]}, End: {stock_data.index[-1]}')
print(f'Training data: {len(train_data)} ')
print(f'Tsesting data: {len(test_data)} ')

sns.set_style('whitegrid')
train_data['Adj Close'].plot(label='training_data', figsize=(15,8));
test_data['Adj Close'].plot(label='testing_data');
plt.title('Stock data '+ stock_name)
plt.legend();
#plt.show()

stock_data[['rsi', 'mom', 'adx', 'macd', 'macd sig', 'cci']].plot(subplots=True, figsize=(15,10), title='Technical indicators');
plt.show()


plt.figure(figsize=(10,6))
sns.heatmap(stock_data.isnull(), cmap='viridis');
plt.title('Missing data');
#plt.show()

train_data['color'] = 'blue'
test_data['color'] = 'orange'

background  = pd.concat([train_data, test_data], axis=0)


background.columns = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj_Close', 'rsi', 'mom', 'adx', 'macd', 'macd_sig', 'cci', 'color']
background.to_sql('background', con=engine, if_exists='replace')



capital = 1000


# validation
val_env = SingleStockEnv(train_data,
                         capital,
                         trade_fee_bid_percent = 0.0025,
                         trade_fee_ask_percent = 0.0025)
state_size = val_env.state_dim
nb_action = val_env.action_space.n

agent = DQNAgent(state_size, nb_action, 0.9, ticker=stock_name, layers=[100, 100, 100])
with open(f'scaler/{stock_name}.pkl', 'rb') as f:
    scaler = pickle.load(f)

last_state = val_env.reset()
last_reward = 0
done = False
agent.load()

while not done:
    last_state = scaler.transform([last_state])
    last_state = torch.Tensor(last_state[0]).float().unsqueeze(0)
    with torch.no_grad():
        action = agent.network(last_state).argmax().item()
    last_state, last_reward, done, info = val_env.step(action)

print('Testing Model')
print('=======================================================================')
print('=======================================================================')
print()
print()
print()
print(f'Start: {test_data.index[0]}  End: {test_data.index[-1]}')
print(f'Since: {len(test_data)} days')
print(f"Begin portfolio value: {capital:8.2f}")
print(f"End portfolio value: {info['portfolio_value'][-1]:10.2f}")
print(f"Return {(info['portfolio_value'][-1] - capital) * 100 / capital:.2f} %")
print(f"Sell: {val_env.sell_buy.count('sell')} times, Buy: {val_env.sell_buy.count('buy')} times")
print('=======================================================================')

validat = pd.DataFrame(info)
validat.index = train_data.index


def marker_buy(col):
    price = col[0]
    sellbuy = col[1]

    if sellbuy == 'buy':
        return price
    else:
        return np.nan


def marker_sell(col):
    price = col[0]
    sellbuy = col[1]

    if sellbuy == 'sell':
        return price
    else:
        return np.nan

def color(col):
    if col == 'buy':
        return 'green'
    elif col == 'sell':
        return 'red'
    else:
        return np.nan


validat['marker_buy'] = validat[['stock_price', 'sell_buy']].apply(marker_buy, axis=1);
validat['marker_sell'] = validat[['stock_price', 'sell_buy']].apply(marker_sell, axis=1);

validat['stock_price'].plot(figsize=(10, 6), c='orange', lw='3');
validat['marker_buy'].plot(style='o', ms=7, label='buy', c='g', alpha=0.5);
validat['marker_sell'].plot(style='o', ms=7, label='sell', c='r', alpha=0.5);
plt.title('Stock Price ' + stock_name)
plt.legend();
#plt.show()

validat['portfolio_value'].plot(figsize=(10, 6), c='r', lw=3);
plt.title('Portfolio Value');
#plt.show()


validat['color'] = validat['sell_buy'].apply(color)
validat['daily_ret'] = validat['stock_price'].pct_change(1)
validat['bt_ret'] = validat['portfolio_value'].pct_change(1)
validat['Backtest'] = validat['bt_ret'].cumsum() + 1
validat['Closeprice'] = validat['daily_ret'].cumsum() + 1
validat['Threshold'] = 1



validat.to_sql('validation', con=engine, if_exists='replace')
#validat.to_csv('file/validation.csv')



val_result = pd.DataFrame()
val_result['start_date'] = [validat.index[0]]
val_result['end_date'] = [validat.index[-1]]
val_result['portfolio_value'] = [validat['portfolio_value'][-1]]
val_result['returns'] = [(validat['portfolio_value'][-1] - capital) * 100/capital]
val_result['order_sell'] = [validat['sell_buy'].value_counts()[1]]
val_result['order_buy'] = [validat['sell_buy'].value_counts()[2]]


val_result.index.name = 'num'
val_result.to_sql('val_result', con=engine, if_exists='replace')
#val_result.to_csv('file/val_result.csv')


##############################################################################################
##############################################################################################



test_env = SingleStockEnv(test_data,
                          capital,
                          trade_fee_bid_percent = 0.0025,
                          trade_fee_ask_percent = 0.0025)
state_size = test_env.state_dim
nb_action = test_env.action_space.n
agent = DQNAgent(state_size, nb_action, 0.9, ticker=stock_name, layers=[100, 100, 100])
with open(f'scaler/{stock_name}.pkl', 'rb') as f:
    scaler = pickle.load(f)

last_state = test_env.reset()
last_reward = 0
done = False
agent.load()

while not done:
    last_state = scaler.transform([last_state])
    last_state = torch.Tensor(last_state[0]).float().unsqueeze(0)
    with torch.no_grad():
        action = agent.network(last_state).argmax().item()
    last_state, last_reward, done, info = test_env.step(action)

print('Testing Model')
print('=======================================================================')
print('=======================================================================')
print()
print()
print()
print(f'Start: {test_data.index[0]}  End: {test_data.index[-1]}')
print(f'Since: {len(test_data)} days')
print(f"Begin portfolio value: {capital:8.2f}")
print(f"End portfolio value: {info['portfolio_value'][-1]:10.2f}")
print(f"Return {(info['portfolio_value'][-1] - capital) * 100 / capital:.2f} %")
print(f"Sell: {test_env.sell_buy.count('sell')} times, Buy: {test_env.sell_buy.count('buy')} times")
print('=======================================================================')

result = pd.DataFrame(info)
result.index = test_data.index

result['marker_buy'] = result[['stock_price', 'sell_buy']].apply(marker_buy, axis=1);
result['marker_sell'] = result[['stock_price', 'sell_buy']].apply(marker_sell, axis=1);

result['stock_price'].plot(figsize=(10, 6), c='orange', lw='3');
result['marker_buy'].plot(style='o', ms=7, label='buy', c='b', alpha=0.5);
result['marker_sell'].plot(style='o', ms=7, label='sell', c='r', alpha=0.5);
plt.title('Stock Price ' + stock_name)
plt.legend();
#plt.show()

result['portfolio_value'].plot(figsize=(10, 6), c='r', lw=3);
plt.title('Portfolio Value');
#plt.show()

result['stock_owned'].plot(figsize=(10, 6), c='purple', lw=3);
plt.title('Position')
#plt.show()

result[['cash_in_hand', 'stock_value']].plot(figsize=(10, 6), lw=3);
plt.title('Cash in Hand and Stock Value');
#plt.show()


result['color'] = result['sell_buy'].apply(color)

result.to_sql('test', con=engine, if_exists='replace')
#result.to_csv('file/test.csv')

##############################################################################################
##############################################################################################




def CAGR(DF):
    df = DF.copy()
    try:
        df['daily_ret'] = df['portfolio_value'].pct_change()
    except:
        df['daily_ret'] = df['Adj Close'].pct_change()

    df['cumulative_ret'] = (1 + df['daily_ret']).cumprod()
    n = len(df) / 252
    cagr = (df['cumulative_ret'][-1]) ** (1 / n) - 1
    return cagr


def volatility(DF):
    df = DF.copy()
    try:
        df['daily_ret'] = df['portfolio_value'].pct_change()
    except:
        df['daily_ret'] = df['Adj Close'].pct_change()
    vol = df['daily_ret'].std() * np.sqrt(252)
    return vol


def sharpe(DF, rf):
    df = DF.copy()
    sr = (CAGR(df) - rf) / volatility(df)
    return sr


def sortino(DF, rf):
    df = DF.copy()
    try:
        df['daily_ret'] = df['portfolio_value'].pct_change()
    except:
        df['daily_ret'] = df['Adj Close'].pct_change()
    neg_vol = df[df['daily_ret']<0]['daily_ret'].std() * np.sqrt(252)
    sr = (CAGR(df) - rf)/neg_vol
    return sr

def max_dd(DF):
    df = DF.copy()
    try:
        df['daily_ret'] = df['portfolio_value'].pct_change()
    except:
        df['daily_ret'] = df['Adj Close'].pct_change()
    df['cumulative_ret'] = (1 + df['daily_ret']).cumprod()
    df['cum_roll_max'] = df['cumulative_ret'].cummax()
    df['drawdown'] = df['cum_roll_max'] - df['cumulative_ret']
    df['drawdown_pct'] = df['drawdown'] / df['cum_roll_max']
    max_dd = df['drawdown_pct'].max()
    return max_dd, df['drawdown_pct']


def calmer(DF):
    df = DF.copy()
    clmr = CAGR(df) / max_dd(df)[0]
    return clmr


print(f'Stock name: {stock_name}')
print(f'Start: {result.index[0]}  End: {result.index[-1]}')
print(f'Compound Annual Growth Rate: {CAGR(result) * 100:.2f} %')
print(f'Volatility: {volatility(result):.4f}')
print(f'shape ratio: {sharpe(result, 0.011):.4f}')
print(f'Sortino ratio: {sortino(result, 0.011):.4f}')
print(f'Maximun drawdown: {max_dd(result)[0] * -100:.2f} %')
print(f'Calmar ratio: {calmer(result):.4f}')
print('-----------------------------------------------------------')

print('-----------------------------------------------------------')
print('Comparing with Adj close')
print(f'Compound Annual Growth Rate (Adj close): {CAGR(test_data) * 100:.2f} %')
print(f'Volatility (benchmark): {volatility(test_data):.4f}')
print(f'shape ratio: {sharpe(test_data, 0.011):.4f}')
print(f'Sortino ratio: {sortino(test_data, 0.011):.4f}')
print(f'Maximun drawdown: {max_dd(test_data)[0] * -100:.2f} %')
print(f'Calmar ratio (benchmark): {calmer(test_data):.4f}')


kpi = pd.DataFrame()
kpi['name'] = ['backtest', 'price']
kpi['cagr'] = [CAGR(result) * 100, CAGR(test_data) * 100]
kpi['volality'] = [volatility(result), volatility(test_data)]
kpi['shape_ratio'] = [sharpe(result, 0.011), sharpe(test_data, 0.011)]
kpi['sortino_ratio'] = [sortino(result, 0.011), sortino(test_data, 0.011)]
kpi['max_dd'] = [max_dd(result)[0] * -100, max_dd(test_data)[0] * -100]
kpi['calmar'] = [calmer(result), calmer(test_data)]

kpi.index.name = 'num'
kpi.to_sql('kpi', con=engine, if_exists='replace')
#kpi.to_csv('file/kpi.csv')





performance = pd.DataFrame()
performance['portfolio_value'] = result['portfolio_value']
performance['stock_price'] = result['stock_price']

performance['daily_ret'] = performance['stock_price'].pct_change(1)
performance['bt_ret'] = performance['portfolio_value'].pct_change(1)
performance['Backtest'] = performance['bt_ret'].cumsum() + 1
performance['Closeprice'] = performance['daily_ret'].cumsum() + 1
performance['Threshold'] = 1
performance['Drawdown'] = max_dd(result)[1] * -100

performance.dropna(inplace=True, axis=0)
performance.to_sql('performance', con=engine, if_exists='replace')
#performance.to_csv('file/performance.csv')

performance['date'] = performance.index
performance['month_year'] = performance['date'].apply(lambda date: str(date.year) +'-'+ str(date.month))
performance['year'] = performance['date'].apply(lambda date: str(date.year))


month_return = performance.groupby('month_year')['bt_ret'].sum() * 100
month_return = pd.DataFrame(month_return)

month_return.reset_index(inplace=True)
month_return.index.name = 'num'
month_return.to_sql('month_return', con=engine, if_exists='replace')
#month_return.to_csv('file/month_return.csv')

year_return = performance.groupby('year')['bt_ret'].sum() * 100
year_return = pd.DataFrame(year_return)
year_return.reset_index(inplace=True)
year_return.index.name = 'num'
year_return.to_sql('year_return', con=engine, if_exists='replace')
#year_return.to_csv('file/year_return.csv')


score = (sharpe(result, 0.011) - sharpe(test_data, 0.011))*10
with open(f'score/{stock_name}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(score, f)


ticker = stock_name
url = 'https://finance.yahoo.com/quote/'+ ticker + '/profile?p=' + ticker
page = requests.get(url)
page_content = page.content
soup = BeautifulSoup(page_content, 'html.parser')
rows = soup.find_all('div', {"class": 'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'})
name = rows[0].find_all('div')[0].find_all('h1')[0].get_text()
disc = rows[0].find_all('div')[1].find_all('span')[0].get_text()


inform = pd.DataFrame()
inform['fullname'] = [name]
inform['subtitle'] = [disc]
inform['stock_name'] = [stock_name]
inform['industry'] = [industry]
inform['value'] = [result['portfolio_value'][-1]]
inform['score'] = score
inform.index.name = 'num'
inform.to_sql('inform', con=engine, if_exists='replace')
#inform.to_csv('file/inform.csv')

test_result = pd.DataFrame()
test_result['start_date'] = [result.index[0]]
test_result['end_date'] = [result.index[-1]]
test_result['portfolio_value'] = [result['portfolio_value'][-1]]
test_result['returns'] = [(result['portfolio_value'][-1] - capital) * 100/capital]
test_result['order_sell'] = [result['sell_buy'].value_counts()[1]]
test_result['order_buy'] = [result['sell_buy'].value_counts()[2]]
test_result['month_1'] = month_return.mean()
test_result['year_1'] = year_return.mean()
test_result['annual_ret'] = test_result['returns']/3


test_result.index.name = 'num'
test_result.to_sql('test_result', con=engine, if_exists='replace')
#test_result.to_csv('file/test_result.csv')


url = 'https://finance.yahoo.com/quote/' + ticker + '/profile?p=' + ticker
page = requests.get(url)
page_content = page.content
soup = BeautifulSoup(page_content, 'html.parser')
tabl = soup.find_all('section', {"class": "Pb(30px) smartphone_Px(20px) undefined"})
about = []
about.append(tabl[0].find_all('span')[0].text)
rows = tabl[0].find_all('div', {"class": "W(100%)"})[0]
rows = rows.find_all('div', {"class": "W(50%) D(ib) Va(t) smartphone_W(100%) smartphone_D(b)"})
for row in rows:
    about.append(row.get_text(separator='|').split('|'))

df = pd.DataFrame()
df['head'] = [about.pop(0)]
for col in about:
    df[col[0]] = [col[2]]
    df[col[3]] = [col[5]]
    df[col[6]] = [col[8]]
    df[col[9]] = [col[11]]

df['Description'] = soup.find_all('div', {"data-test": "prof-desc"})[0].get_text()
df.set_index('head', inplace=True)
df.to_sql('about', con=engine, if_exists='replace')

