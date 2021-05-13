
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Style
from tkinter.font import Font
from tkscrolledframe import ScrolledFrame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import ImageTk, Image
from statsmodels.tsa.seasonal import seasonal_decompose
from bs4 import BeautifulSoup
import requests
import pyfolio as pf
import matplotlib.animation as ani
from perform import CAGR, volatility, sharpe, max_dd, calmer
import pickle
sns.set_style('white')

window = Tk()
window.title('AI trading')
window.geometry('1350x780')
window.resizable(0,0)


bg = PhotoImage(file='tee (1) (1).png')


def go_to_first():
    canvas.destroy()



canvas = Canvas(window, width=1350, height=780)
canvas.pack(fill='both', expand=True)

canvas.create_image(0, 0, image=bg, anchor='nw')




bt = Button(window, text='GET START',
            width=16, height=2,
            bg='#fd2b91', relief=RAISED,
            font=('Berlin Sans FB Demi', 15, 'bold'),
            fg='white',
            command=go_to_first)

bt_window = canvas.create_window(1000, 660, anchor='nw', window=bt)



# --------------------Window-----------------------------------


# window.iconphoto(False, PhotoImage(file='commission.png'))
window.title('Alpha trading')
window.geometry('1350x780')
window.resizable(0, 0)
window.config(padx=0, pady=0, bg='#0d1d6b')
window.option_add('*font', ('Calibri', 12))


# ---------------------Notebook & Tabs -------------------------
notebook = ttk.Notebook(window, width=1300, height=750)
tab_watch = Frame(notebook)
tab_backtest = Frame(notebook)
tab_preformance = Frame(notebook)
tab_portfolio = Frame(notebook)
tab_predict = Frame(notebook)


notebook.add(tab_watch, text='Watch')
notebook.add(tab_backtest, text='Backtest')
notebook.add(tab_preformance, text='Performance')
notebook.add(tab_portfolio, text='Portfolio')
notebook.add(tab_predict, text='Prediction')
notebook.pack()

style = ttk.Style()
font = Font(family='Calibri', size=12, weight='bold')
style.configure('TNotebook.Tab', font=font)
style.configure('TNotebook.Tab', padding=[5, 2])
style.configure('TNotebook.Tab', foreground='black')
win_bg = window.cget('bg')
style.configure('TNotebook', background=win_bg)

# -------------------------------Tab Watch---------------------------

main_frame = Frame(master=tab_watch)
main_frame.pack(fill=BOTH, expand=1)

sf = ScrolledFrame(main_frame)
sf.pack(side="top", expand=1, fill="both")
sf.bind_arrow_keys(main_frame)
sf.bind_scroll_wheel(main_frame)

frame = sf.display_widget(Frame)

_data = {'Basic materials': ['BBL', 'BHP', 'IFF', 'LIN', 'MT', 'PKX', 'RIO', 'SCCO', 'VALE', 'VMC'],
         'Communication service': ['DIS', 'DISCA', 'EA', 'GOOG', 'NFLX', 'OMC', 'RELX', 'TU', 'VZ', 'WPP'],
         'Consumer cyclical': ['AMZN', 'EBAY', 'FORD', 'GM', 'HD', 'LVS', 'MCD', 'NKE', 'SBUX', 'TM'],
         'Consumer defensive': ['BG', 'BTI', 'CCEP', 'CPB', 'DEO', 'GIS', 'K', 'KR', 'NWL', 'UL'],
         'Energy': ['BP', 'COP', 'CVX', 'ENB', 'EQNR', 'PTR', 'RDS-B', 'SNP', 'TOT', 'XOM'],
         'Financial service': ['BAC-PL', 'BRK-A', 'C', 'GS', 'JPM', 'MA', 'MS', 'TD', 'V', 'WFC-PL'],
         'Industrial': ['ABB', 'CAT', 'CMI', 'DAL', 'DE', 'GD', 'GE', 'JCI', 'LUV', 'PCAR'],
         'Healthcare': ['ANTM', 'CAH', 'CI', 'CNC', 'FMS', 'GSK', 'JNJ', 'PFE', 'SNY', 'ZBH'],
         'Real estate': ['ARE', 'AVB', 'HST', 'MPW', 'NLY', 'O', 'SPG', 'VTR', 'WPC', 'WY'],
         'Technology': ['AAPL', 'CSCO', 'HPQ', 'IBM', 'INTC', 'MU', 'SAP', 'STM', 'TSM', 'TXN'],
         'Cryptocurrency': ['BTC-USD', 'DASH-USD', 'DCR-USD', 'DGB-USD', 'ETC-USD', 'ETH-USD','LTC-USD',
                            'SC-USD', 'USDT-USD', 'WAVES-USD', 'XEM-USD', 'XLM-USD', 'XMR-USD', 'XRP-USD', 'ZEC-USD']}

# ---------------------------Label Frame---------------------------------------------------



fm1 = LabelFrame(master=frame, text='Symbol')
fm1.grid(row=0, column=0, pady=10, padx=10, sticky=W)

lb1 = Label(master=fm1, text='Sector:')
lb1.grid(row=0, column=0, padx=10)

sector = []

for sec in _data.keys():
    sector.append(sec)

combo_sector = ttk.Combobox(master=fm1, values=sector)
combo_sector.grid(row=0, column=1, pady=5)
combo_sector.bind('<<ComboboxSelected>>', lambda e: combo_sector_selected())
combo_sector.bind('<Key>', 'break')

lb2 = Label(master=fm1, text='Symbol:')
lb2.grid(row=0, column=2, padx=10)

combo_symbol = ttk.Combobox(master=fm1)
combo_symbol.grid(row=0, column=3)
combo_sector.bind('<<ComboboxSelected>>')
combo_symbol.bind('<Key>', 'break')

bt1 = Button(master=fm1, text='Search', width=10, command=lambda: button_search(), bg='yellow', font='Calibri 10 bold')
bt1.grid(row=0, column=4, padx=20)

# --------------------------------------------------------------------------------------------------------


fm2 = LabelFrame(master=frame, text='Stock price', width=1, height=1)
fm2.grid(row=1, column=0, pady=10, padx=10, sticky=NW)
fm2.config(padx=10, pady=10)

# --------------------------------------------------------------------------------------------------------


fm3 = LabelFrame(master=frame, text='Technical indicator')
fm3.grid(row=2, column=0, pady=10, padx=10, sticky=NW)
fm3.config(padx=10, pady=10)

# --------------------------------------------------------------------------------------------------------


fm4 = LabelFrame(master=frame, text='Summary')
fm4.grid(row=1, column=1, pady=10, padx=10)
fm4.config(padx=10, pady=10)

# --------------------------------------------------------------------------------------------------------


fm5 = LabelFrame(master=frame, text='Seasonal Decomposition')
fm5.grid(row=2, column=1, pady=10, padx=10, sticky=NW, columnspan=2)
fm5.config(padx=10, pady=10)

# --------------------------------------------------------------------------------------------------------


fm6 = LabelFrame(master=frame, text='Exponentially-weighted Moving Average')
fm6.grid(row=3, column=0, pady=10, padx=10, sticky=NW, columnspan=2)
fm6.config(padx=10, pady=10)

# --------------------------------------------------------------------------------------------------------


fm7 = LabelFrame(master=frame, text='Bollinger Band')
fm7.grid(row=3, column=1, pady=10, padx=10, sticky=NW, columnspan=2)
fm7.config(padx=10, pady=10)








# =========================================================================================
# =========================================================================================


# ------------------------Tab Watch ------------------------------------------------------------

def combo_sector_selected():
    sel_sector = combo_sector.get()
    symbols = _data[sel_sector]
    combo_symbol.delete(0, END)
    combo_symbol.config(values=symbols)
    combo_symbol.current(0)


def button_search():
    sel_symbol = combo_symbol.get()


    for widget in fm2.winfo_children():
        widget.destroy()

    fm2.pack_forget()

    for widget in fm3.winfo_children():
        widget.destroy()

    fm3.pack_forget()

    for widget in fm4.winfo_children():
        widget.destroy()

    fm4.pack_forget()

    for widget in fm5.winfo_children():
        widget.destroy()

    fm5.pack_forget()

    for widget in fm6.winfo_children():
        widget.destroy()

    fm6.pack_forget()

    for widget in fm7.winfo_children():
        widget.destroy()

    fm7.pack_forget()


    df = pd.read_csv(f'project_2/dqn_allin/data/{sel_symbol}.csv', index_col='Date', parse_dates=True)
    df.index = pd.to_datetime(df.index)
    with sns.axes_style('whitegrid'):
        figure1 = plt.Figure(figsize=(6, 4), dpi=100)
        figure1.patch.set_facecolor('#F0F0F0')
        ax1 = figure1.add_subplot(111)
        ax1.set_facecolor('white')
        line1 = FigureCanvasTkAgg(figure1, fm2)

        toolbar = NavigationToolbar2Tk(line1, fm2)
        toolbar.update()
        line1.get_tk_widget().pack()

        line1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df['stock_price'].plot(ax=ax1, color='#0d1d6b')
        ax1.set_title(f'Stock prick {sel_symbol}')

# =========================================================================================


    if combo_sector.get() == 'Cryptocurrency':
        train_test_data = pd.read_csv(f'project_2/dqn_allin/train_test_data/{sel_symbol}.csv', index_col='Datetime',
                                      parse_dates=True)
        train_test_data = train_test_data.iloc[len(train_test_data)//2:]
        train_test_data.index = pd.to_datetime(train_test_data.index)

    else:

        train_test_data = pd.read_csv(f'project_2/dqn_allin/train_test_data/{sel_symbol}.csv', index_col='Date', parse_dates=True).loc['2018-01-02':]
        train_test_data.index = pd.to_datetime(train_test_data.index)

    fig, ax = plt.subplots(5, figsize=(6, 5))
    fig.patch.set_facecolor('#F0F0F0')
    line3 = FigureCanvasTkAgg(fig, fm3)
    toolbar = NavigationToolbar2Tk(line3, fm3)
    toolbar.update()
    line3.get_tk_widget().pack()
    line3.get_tk_widget().pack(side=LEFT, fill=BOTH)

    ax[0].plot(train_test_data['rsi'], c='tab:blue', lw=2)
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_title(' Relative Strength Index (RSI)')
    ax[0].set_facecolor('white')


    ax[1].plot(train_test_data['mom'], c='tab:orange', lw=2)
    ax[1].get_xaxis().set_visible(False)
    ax[1].set_title('Momentum Indicator (MOM)')
    ax[1].set_facecolor('white')


    ax[2].plot(train_test_data['adx'], c='tab:green', lw=2)
    ax[2].get_xaxis().set_visible(False)
    ax[2].set_title('Average Directional Index (ADX)')
    ax[2].set_facecolor('white')


    ax[3].plot(train_test_data['macd'], c='tab:red', lw=2)
    ax[3].get_xaxis().set_visible(False)
    ax[3].set_title('Moving Average Convergence Divergence (MACD)')
    ax[3].set_facecolor('white')


    ax[4].plot(train_test_data['cci'], c='tab:purple', lw=2)
    ax[4].get_xaxis().set_visible(False)
    ax[4].set_title(' Commodity channel index (CCI)')
    ax[4].set_facecolor('white')


    plt.tight_layout()

    plt.close(fig)

# =========================================================================================



    url = f'https://finance.yahoo.com/quote/{sel_symbol}?p={sel_symbol}'
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    rows = soup.find_all('div', {
        "class": 'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'})
    st_name = rows[0].find_all('div')[0].find_all('h1')[0].get_text()
    disc = rows[0].find_all('div')[1].find_all('span')[0].get_text()
    tabl = soup.find_all('div', {
        'class': 'Bxz(bb) D(ib) Va(t) Mih(250px)!--lgv2 W(100%) Mt(-6px) Mt(0px)--mobp Mt(0px)--mobl W(50%)!--lgv2 Mend(20px)!--lgv2 Pend(10px)!--lgv2'})
    prices = soup.find_all('div', {'class', 'D(ib) Va(m) Maw(65%) Ov(h)'})[0].get_text(separator='|').split('|')
    stat = []
    for t in tabl:
        rows = t.find_all('div')
        for row in rows:
            data = row.get_text(separator='|').split('|')
            stat.append(data)

    col1 = stat[0][:-1:2]
    col2 = stat[0][1::2]
    col3 = stat[1][:-1:2]
    col3.pop(5)
    col4 = stat[1][1::2]
    add = col4.pop(5)
    col4[4] = col4[4] + ' -\n' + add

    st_name = Label(fm4, text=st_name)
    st_name.config(font='Calibri 18 bold')
    st_name.pack()

    subname = Label(fm4, text=disc)
    subname.config(font='Calibri 14')
    subname.pack()

    stock_price = Label(fm4, text=prices[0])
    stock_price.config(font='Calibri 20 bold')
    stock_price.pack()

    sub_price = Label(fm4, text=prices[1])
    if prices[1][0] == '+':
        sub_price.config(font='Calibri 14 bold', fg='green')
    else:
        sub_price.config(font='Calibri 14 bold', fg='red')
    sub_price.pack()

    attime = Label(fm4, text=prices[2])
    attime.config(font='Calibri 10', fg='grey')
    attime.pack()

    tabl = Frame(fm4)
    tabl.pack()

    for i in range(len(col1)):
        lb = Label(tabl, text=col1[i])
        lb.grid(row=i, column=0, sticky=W, padx=10, pady=5)

    for i in range(len(col2)):
        lb = Label(tabl, text=col2[i])
        lb.grid(row=i, column=1, sticky=E, padx=5, pady=5)
        lb.config(font='Calibri 12 bold')

    for i in range(len(col3)):
        lb = Label(tabl, text=col3[i])
        lb.grid(row=i, column=2, sticky=W, padx=10, pady=5)

    for i in range(len(col4)):
        lb = Label(tabl, text=col4[i])
        lb.grid(row=i, column=3, sticky=E, padx=5, pady=5)
        lb.config(font='Calibri 12 bold')

# =========================================================================================




    components = seasonal_decompose(df['stock_price'], model='multiplicative', period=21)

    ts = (df['stock_price'].to_frame('Original')
          .assign(Trend=components.trend)
          .assign(Seasonality=components.seasonal)
          .assign(Residual=components.resid))

    with sns.axes_style('white'):
        fig, ax = plt.subplots(4, figsize=(6, 5))
        fig.patch.set_facecolor('#F0F0F0')
        line2 = FigureCanvasTkAgg(fig, fm5)
        toolbar = NavigationToolbar2Tk(line2, fm5)
        toolbar.update()
        line2.get_tk_widget().pack()
        line2.get_tk_widget().pack(side=LEFT, fill=BOTH)


        ax[0].plot(ts['Original'], c='tab:blue', lw=2, alpha=0.9)
        ax[0].get_xaxis().set_visible(False)
        ax[0].set_title('Original')
        ax[0].set_facecolor('white')


        ax[1].plot(ts['Trend'], c='tab:orange', lw=2, alpha=0.9)
        ax[1].get_xaxis().set_visible(False)
        ax[1].set_title('Trend')
        ax[1].set_facecolor('white')


        ax[2].plot(ts['Seasonality'], c='tab:green', lw=2, alpha=0.9)
        ax[2].get_xaxis().set_visible(False)
        ax[2].set_title('Seasonality')
        ax[2].set_facecolor('white')


        ax[3].plot(ts['Residual'], c='tab:red', lw=2, alpha=0.9)
        ax[3].get_xaxis().set_visible(False)
        ax[3].set_title('Residual')
        ax[3].set_facecolor('white')


        plt.tight_layout()

        plt.close(fig)


# =========================================================================================



    df['10-day-EWA'] = df['stock_price'].ewm(span=10).mean()
    df['20-day-EWA'] = df['stock_price'].ewm(span=20).mean()
    df['50-day-EWA'] = df['stock_price'].ewm(span=50).mean()
    df['100-day-EWA'] = df['stock_price'].ewm(span=100).mean()

    with sns.axes_style('whitegrid'):
        figure6 = plt.Figure(figsize=(6, 4), dpi=100)
        figure6.patch.set_facecolor('#F0F0F0')
        ax6 = figure6.add_subplot(111)
        ax6.set_facecolor('white')
        line6 = FigureCanvasTkAgg(figure6, fm6)
        toolbar = NavigationToolbar2Tk(line6, fm6)
        toolbar.update()
        line6.get_tk_widget().pack()
        line6.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df[['stock_price', '10-day-EWA', '20-day-EWA', '50-day-EWA', '100-day-EWA']].plot(ax=ax6, lw=2, cmap='coolwarm')
        ax6.set_title(f'EWMA {sel_symbol}')


# =========================================================================================



    df['Upper band'] = df['stock_price'].rolling(20).mean() + 2*df['stock_price'].rolling(20).std()
    df['Lower band'] = df['stock_price'].rolling(20).mean() - 2*df['stock_price'].rolling(20).std()


    with sns.axes_style('whitegrid'):
        figure7 = plt.Figure(figsize=(6, 4), dpi=100)
        figure7.patch.set_facecolor('#F0F0F0')
        ax7 = figure7.add_subplot(111)
        ax7.set_facecolor('white')
        line7 = FigureCanvasTkAgg(figure7, fm7)
        toolbar = NavigationToolbar2Tk(line7, fm7)
        toolbar.update()
        line7.get_tk_widget().pack()
        line7.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df[['stock_price', 'Upper band', 'Lower band']].plot(ax=ax7)
        ax7.set_title(f'Bollinger Band {sel_symbol}')



# ------------------------------------------------------------------------------------------------








# -----------------------Tab Backtest----------------------------------------------------------------------------




main_frame2 = Frame(master=tab_backtest)
main_frame2.pack(fill=BOTH, expand=1)

sf = ScrolledFrame(main_frame2)
sf.pack(side="top", expand=1, fill="both")

sf.bind_arrow_keys(main_frame2)
sf.bind_scroll_wheel(main_frame2)

frame2 = sf.display_widget(Frame)


# --------------------------------------------------------------------------------------------------------

backtest_fm1 = Frame(master=frame2)
backtest_fm1.grid(row=0, column=0, pady=15, padx=10, sticky=W, columnspan=2)

lb1 = Label(master=backtest_fm1, text='Sector:')
lb1.grid(row=0, column=0, padx=10)

sector = []

for sec in _data.keys():
    sector.append(sec)

combo_sector_bt = ttk.Combobox(master=backtest_fm1, values=sector)
combo_sector_bt.grid(row=0, column=1, pady=5)
combo_sector_bt.bind('<<ComboboxSelected>>', lambda e: combo_sector_selected_bt())
combo_sector_bt.bind('<Key>', 'break')

lb2 = Label(master=backtest_fm1, text='Symbol:')
lb2.grid(row=0, column=2, padx=10)

combo_symbol_bt = ttk.Combobox(master=backtest_fm1)
combo_symbol_bt.grid(row=0, column=3)
combo_sector_bt.bind('<<ComboboxSelected>>')
combo_symbol_bt.bind('<Key>', 'break')


lb3 = Label(master=backtest_fm1, text='Capital:')
lb3.grid(row=0, column=4, padx=10)

cap_ent = Entry(master=backtest_fm1)
cap_ent.grid(row=0, column=5)


bt2 = Button(master=backtest_fm1, text='Backtest', width=10, command=lambda: simulation(), bg='#0ac73f',
             font='Calibri 12 bold', fg='white')
bt2.grid(row=0, column=6, padx=20)


bt1 = Button(master=backtest_fm1, text='Result', width=10, command=lambda: button_search_bt(), bg='red', font='Calibri 12 bold', fg='white')
bt1.grid(row=0, column=7, padx=20)






# --------------------------------------------------------------------------------------

backtest_fm2 = LabelFrame(master=frame2, text='Backtest')
backtest_fm2.grid(row=1, column=0, pady=10, padx=10, sticky=NW, columnspan=2)
backtest_fm2.config(padx=10, pady=10)


# --------------------------------------------------------------------------------------

kpi = Frame(master=frame2)
kpi.grid(row=2, column=0, pady=10, padx=10, columnspan=2)
kpi.config(padx=10, pady=10)




# --------------------------------------------------------------------------------------

backtest_fm3 = LabelFrame(master=frame2, text='Portfolio Value')
backtest_fm3.grid(row=3, column=0, pady=10, padx=10, sticky=NW)
backtest_fm3.config(padx=10, pady=10)




# --------------------------------------------------------------------------------------

backtest_fm4 = LabelFrame(master=frame2, text='Compare with close price')
backtest_fm4.grid(row=3, column=1, pady=10, padx=10, sticky=NW)
backtest_fm4.config(padx=10, pady=10)


# --------------------------------------------------------------------------------------

backtest_fm5 = Frame(master=frame2)
backtest_fm5.grid(row=4, column=0, pady=10, padx=10, sticky=NW,  columnspan=2)
backtest_fm5.config(padx=10, pady=10)



# --------------------------------------------------------------------------------------

backtest_fm6 = Frame(master=frame2)
backtest_fm6.grid(row=5, column=0, pady=10, padx=10, sticky=NW)
backtest_fm6.config(padx=10, pady=10)

# --------------------------------------------------------------------------------------

backtest_fm7 = Frame(master=frame2)
backtest_fm7.grid(row=5, column=1, pady=10, padx=10, sticky=NW,  columnspan=2)
backtest_fm7.config(padx=10, pady=10)

# --------------------------------------------------------------------------------------

backtest_fm8 = Frame(master=frame2)
backtest_fm8.grid(row=6, column=0, pady=10, padx=10, sticky=NW,  columnspan=2)
backtest_fm8.config(padx=10, pady=10)


# --------------------------------------------------------------------------------------

backtest_fm9 = Frame(master=frame2)
backtest_fm9.grid(row=6, column=1, pady=10, padx=10, sticky=NW,  columnspan=2)
backtest_fm9.config(padx=10, pady=10)



def combo_sector_selected_bt():
    sel_sector_bt = combo_sector_bt.get()
    symbols = _data[sel_sector_bt]
    combo_symbol_bt.delete(0, END)
    combo_symbol_bt.config(values=symbols)
    combo_symbol_bt.current(0)



def simulation():


    stock_name = combo_symbol_bt.get()

    capital = float(cap_ent.get())

    with plt.style.context('dark_background'):

        df1 = pd.read_csv(f'project_2/dqn_allin/data/{stock_name}.csv', parse_dates=True)
        df1.set_index('Date', inplace=True)
        df1 = df1[['stock_price', 'marker_buy', 'marker_sell', 'portfolio_value']]
        df1.columns = ['Stock price', 'Buy', 'Sell', 'Portfolio value']
        df1['Portfolio value'] = (df1['Portfolio value']/1000) * capital

        fig = plt.figure()

        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")  # rotate the x-axis values
        plt.subplots_adjust(bottom=0.2, top=0.9)  # ensuring the dates (on the x-axis) fit in the screen
        plt.ylabel('Price')
        plt.xlabel('Days')
        plt.title(stock_name)

        def buildmebarchart(i=int):
            plt.legend(df1.columns[:-1], loc=3)
            #
            s = i - 100
            if s < 0:
                s = 0
            plt.plot(df1['Stock price'][:i].values, color='#03fff7', lw='1.5', alpha=1);
            plt.plot(df1['Buy'][:i].values, ls=' ', mfc='#00ff04', lw='2', alpha=1, marker='o', ms=7);
            plt.plot(df1['Sell'][:i].values, ls=' ', mfc='red', lw='2', alpha=1, marker='o', ms=7);
            plt.xlabel(
                f'Days\n\nPortfolio value: {df1["Portfolio value"].values[i] : 8.2f}     Profit: {(df1["Portfolio value"].values[i] - df1["Portfolio value"].iloc[0]) * 100 / df1["Portfolio value"].iloc[0]:8.2f} %')
            plt.xlim([s, i + 20])

        animator = ani.FuncAnimation(fig, buildmebarchart, interval=50, frames=len(df1))
        # animator.save(f"anime/{stock_name}.mp4", fps=30)
        plt.show()




def button_search_bt():

    capital = float(cap_ent.get())

    sel_symbol_bt = combo_symbol_bt.get()
    sector = combo_sector_bt.get()



    for widget in kpi.winfo_children():
        widget.destroy()

    kpi.pack_forget()




    for widget in backtest_fm2.winfo_children():
        widget.destroy()

    backtest_fm2.pack_forget()

    backtest_data = pd.read_csv(f'project_2/dqn_allin/data/{sel_symbol_bt}.csv', index_col='Date', parse_dates=True)
    backtest_data.index = pd.to_datetime(backtest_data.index)

    figure1 = plt.Figure(figsize=(12.4, 8), dpi=100)
    figure1.patch.set_facecolor('#F0F0F0')
    ax1 = figure1.add_subplot(111)
    ax1.set_facecolor('black')
    line1 = FigureCanvasTkAgg(figure1, backtest_fm2)

    toolbar = NavigationToolbar2Tk(line1, backtest_fm2)
    toolbar.update()

    line1.get_tk_widget().pack(side=LEFT, fill=BOTH)
    backtest_data['stock_price'].plot(ax=ax1, label='Stock price', color='#03fff7', alpha=0.7)
    backtest_data['marker_sell'].plot(ax=ax1, style='o', ms=5, label='Sell', c='r', mfc='red')
    backtest_data['marker_buy'].plot(ax=ax1, style='o', ms=5, label='Buy', c='g', mfc='#00ff04')
    ax1.legend()
    ax1.set_title(f'Stock prick {sel_symbol_bt}')






    backtest_data['portfolio_value'] = (backtest_data['portfolio_value']/1000) * capital

    returns = (backtest_data["portfolio_value"].iloc[-1] - capital) * 100 / capital
    cagr = CAGR(backtest_data, sector)
    vol = volatility(backtest_data, sector)
    sr = sharpe(backtest_data, 0.011, sector)
    dd = max_dd(backtest_data)
    clmr = calmer(backtest_data, sector)






    btlb = Label(master=kpi, text='Backtest', font='Calibri 16', fg='grey')
    btlb.grid(row=0, column=0, sticky=W)

    plb = Label(master=kpi, text='Price', font='Calibri 16', fg='grey')
    plb.grid(row=2, column=0, sticky=W)



    returns = (backtest_data['stock_price'].iloc[-1] - backtest_data['stock_price'].iloc[0]) * 100 / backtest_data['stock_price'].iloc[0]
    cagr = CAGR(backtest_data.drop('portfolio_value', axis=1), sector)
    vol = volatility(backtest_data.drop('portfolio_value', axis=1), sector)
    sr = sharpe(backtest_data.drop('portfolio_value', axis=1), 0.011, sector)
    dd = max_dd(backtest_data.drop('portfolio_value', axis=1))
    clmr = calmer(backtest_data.drop('portfolio_value', axis=1), sector)



    ret = Label(master=kpi, text = f'{returns:.2f}%', font='Calibri 22', fg='black')
    ret.grid(row=2, column=1, padx=15, pady=10)


    cag = Label(kpi, text=f'{cagr*100:.2f}%', font='Calibri 22', fg='black')
    cag.grid(row=2, column=2, padx=15, pady=10)

    vollb = Label(kpi, text=f'{vol:.2f}', font='Calibri 22', fg='black')
    vollb.grid(row=2, column=3, padx=15, pady=10)

    srlb = Label(kpi, text=f'{sr:.2f}', font='Calibri 22', fg='black')
    srlb.grid(row=2, column=4, padx=15, pady=10)


    ddlb = Label(kpi, text=f'{dd*-100:.2f}%', font='Calibri 22', fg='black')
    ddlb.grid(row=2, column=5, padx=15, pady=10)


    clmrlb = Label(kpi, text=f'{clmr:.4f}', font='Calibri 22', fg='black')
    clmrlb.grid(row=2, column=6, padx=15, pady=10)






    btreturns = (backtest_data["portfolio_value"].iloc[-1] - capital) * 100 / capital
    btcagr = CAGR(backtest_data, sector)
    btvol = volatility(backtest_data, sector)
    btsr = sharpe(backtest_data, 0.011, sector)
    btdd = max_dd(backtest_data)
    btclmr = calmer(backtest_data, sector)



    retbt = Label(master=kpi, text = f'{btreturns:.2f}%', font='Calibri 22')
    if btreturns > returns:
        retbt.config(fg='green')
    else:
        retbt.config(fg='red')
    retbt.grid(row=0, column=1, padx=15, pady=10)
    ret_dis = Label(master=kpi, text = 'Return', font='Calibri 16', fg='grey')
    ret_dis.grid(row=1, column=1, padx=15, pady=10)


    cagbt = Label(kpi, text=f'{btcagr*100:.2f}%', font='Calibri 22')
    if btcagr > cagr:
        cagbt.config(fg='green')
    else:
        cagbt.config(fg='red')

    cagbt.grid(row=0, column=2, padx=15, pady=10)
    cag_dis = Label(master=kpi, text = 'Compound Annual Growth Rate', font='Calibri 16', fg='grey')
    cag_dis.grid(row=1, column=2, padx=15, pady=10)


    vollbbt = Label(kpi, text=f'{btvol:.2f}', font='Calibri 22', fg='black')
    if btvol < vol:
        vollbbt.config(fg='green')
    else:
        vollbbt.config(fg='red')
    vollbbt.grid(row=0, column=3, padx=15, pady=10)
    vollb_dis = Label(master=kpi, text = 'Volatility', font='Calibri 16', fg='grey')
    vollb_dis.grid(row=1, column=3, padx=15, pady=10)


    srlbbt = Label(kpi, text=f'{btsr:.2f}', font='Calibri 22', fg='black')
    if btsr > sr:
        srlbbt.config(fg='green')
    else:
        srlbbt.config(fg='red')
    srlbbt.grid(row=0, column=4, padx=15, pady=10)
    srlb_dis = Label(master=kpi, text = 'Sharp ratio', font='Calibri 16', fg='grey')
    srlb_dis.grid(row=1, column=4, padx=15, pady=10)


    ddlbbt = Label(kpi, text=f'{btdd*-100:.2f}%', font='Calibri 22', fg='red')
    if btdd < dd:
        ddlbbt.config(fg='green')
    else:
        ddlbbt.config(fg='red')
    ddlbbt.grid(row=0, column=5, padx=15, pady=10)
    ddlb_dis = Label(master=kpi, text = 'Maximun drawdown', font='Calibri 16', fg='grey')
    ddlb_dis.grid(row=1, column=5, padx=15, pady=10)


    clmrlbbt = Label(kpi, text=f'{btclmr:.4f}', font='Calibri 22', fg='black')
    if btclmr > clmr:
        clmrlbbt.config(fg='green')
    else:
        clmrlbbt.config(fg='red')
    clmrlbbt.grid(row=0, column=6, padx=15, pady=10)
    clmrlb_dis = Label(master=kpi, text = 'Calmar ratio', font='Calibri 16', fg='grey')
    clmrlb_dis.grid(row=1, column=6, padx=15, pady=10)




    for widget in backtest_fm3.winfo_children():
        widget.destroy()

    backtest_fm3.pack_forget()


    with sns.axes_style('whitegrid'):
        figure2 = plt.Figure(figsize=(6, 4), dpi=100)
        figure2.patch.set_facecolor('#F0F0F0')
        ax2 = figure2.add_subplot(111)
        ax2.set_facecolor('white')
        line2 = FigureCanvasTkAgg(figure2, backtest_fm3)
        toolbar = NavigationToolbar2Tk(line2, backtest_fm3)
        toolbar.update()
        line2.get_tk_widget().pack()

        line2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        backtest_data['portfolio_value'].plot(ax=ax2, color='#fc039d', alpha=1, lw=2.5)
        ax2.set_title(f'Portfolio value {sel_symbol_bt}')
        ax2.set_ylabel('Value')


    for widget in backtest_fm4.winfo_children():
        widget.destroy()

    backtest_fm4.pack_forget()


    bt_returns = backtest_data['portfolio_value'].pct_change()

    backtest_data['Close price'] = backtest_data['stock_price']
    price = backtest_data['Close price'].pct_change()


    with sns.axes_style('whitegrid'):
        figure3 = plt.figure(figsize=(6, 4), dpi=100)
        figure3.patch.set_facecolor('#F0F0F0')
        pf.plotting.plot_rolling_returns(bt_returns, price)
        plt.close(figure3)
        line3 = FigureCanvasTkAgg(figure3, backtest_fm4)
        toolbar = NavigationToolbar2Tk(line3, backtest_fm4)
        toolbar.update()
        line3.get_tk_widget().pack()





    for widget in backtest_fm5.winfo_children():
        widget.destroy()

    backtest_fm5.pack_forget()



    with sns.axes_style('whitegrid'):
        figure4 = plt.figure(1)
        figure4.patch.set_facecolor('#F0F0F0')
        plt.subplot(1, 3, 1)
        pf.plot_annual_returns(bt_returns)
        plt.subplot(1, 3, 2)
        pf.plot_monthly_returns_dist(bt_returns)
        plt.subplot(1, 3, 3)
        pf.plot_monthly_returns_heatmap(bt_returns)
        plt.tight_layout()
        figure4.set_size_inches(12, 4)
        plt.close(figure4)

        line4 = FigureCanvasTkAgg(figure4, backtest_fm5)
        line4.get_tk_widget().pack()



    for widget in backtest_fm6.winfo_children():
        widget.destroy()

    backtest_fm6.pack_forget()


    with sns.axes_style('whitegrid'):
        figure5 = plt.figure(figsize=(6, 4), dpi=100)
        figure5.patch.set_facecolor('#F0F0F0')
        pf.plot_return_quantiles(bt_returns)
        plt.close(figure5)

        line5 = FigureCanvasTkAgg(figure5, backtest_fm6)
        line5.get_tk_widget().pack()



    for widget in backtest_fm7.winfo_children():
        widget.destroy()

    backtest_fm7.pack_forget()


    with sns.axes_style('whitegrid'):
        figure6 = plt.figure(figsize=(6, 4.5), dpi=100)
        figure6.patch.set_facecolor('#F0F0F0')
        pf.plot_rolling_beta(bt_returns, price);
        plt.close(figure6)

        line6 = FigureCanvasTkAgg(figure6, backtest_fm7)
        line6.get_tk_widget().pack()




    for widget in backtest_fm8.winfo_children():
        widget.destroy()

    backtest_fm8.pack_forget()


    with sns.axes_style('whitegrid'):
        figure7 = plt.figure(figsize=(6, 4.5), dpi=100)
        figure7.patch.set_facecolor('#F0F0F0')
        pf.plot_rolling_sharpe(bt_returns)
        plt.close(figure7)

        line7 = FigureCanvasTkAgg(figure7, backtest_fm8)
        line7.get_tk_widget().pack()



    for widget in backtest_fm9.winfo_children():
        widget.destroy()

    backtest_fm9.pack_forget()


    with sns.axes_style('whitegrid'):
        figure8 = plt.figure(figsize=(6, 4.5), dpi=100)
        figure8.patch.set_facecolor('#F0F0F0')
        pf.plot_drawdown_underwater(bt_returns)
        plt.close(figure8)

        line8 = FigureCanvasTkAgg(figure8, backtest_fm9)
        line8.get_tk_widget().pack()








main_frame3 = Frame(master=tab_preformance)
main_frame3.pack(fill=BOTH, expand=1)

sf = ScrolledFrame(main_frame3)
sf.pack(side="top", expand=1, fill="both")
sf.bind_arrow_keys(main_frame3)
sf.bind_scroll_wheel(main_frame3)

frame3 = sf.display_widget(Frame)

sector = {'basic materials': ['VMC', 'VALE', 'SCCO', 'RIO', 'PKX', 'IFF', 'MT', 'LIN', 'BHP', 'BBL'],
          'communication service': ['WPP', 'VZ', 'TU', 'RELX', 'OMC', 'NFLX', 'GOOG', 'EA', 'DISCA', 'DIS'],
          'consumer cyclical' : ['TM', 'SBUX', 'NKE', 'MCD', 'LVS', 'HD', 'GM', 'FORD', 'EBAY', "AMZN"],
          'consumer defensive': ['UL', 'NWL', 'KR', 'K', 'GIS', 'DEO', 'CPB', 'CCEP', 'BTI', "BG"],
          'energy' : ['XOM', 'TOT', 'SNP', 'RDS-B', 'PTR', 'EQNR', 'CVX', 'COP', "BP", 'ENB'],
          'financial service' : ['WFC-PL', 'V', 'TD', 'MS', 'MA', 'JPM', 'GS', 'C', 'BRK-A', 'BAC-PL'],
          'industrial' : ['PCAR', 'LUV', 'JCI', 'GE', 'GD', 'DE', 'DAL', 'CMI', 'CAT', 'ABB'],
          'health care' : ['ZBH', 'SNY', 'PFE', 'JNJ', 'GSK', 'FMS', 'CNC', 'CI', 'CAH', 'ANTM'],
          'real estate' : ['WY', 'WPC', 'VTR', 'SPG', 'O', 'NLY', 'MPW', 'HST', 'AVB', "ARE"],
          'technology' : ['TXN', 'TSM', 'STM', 'SAP', 'MU', 'INTC', 'IBM', 'HPQ', 'CSCO', 'AAPL']}


temp_dir = {'SYMBOL': [],
            'INDUSTRIES': [],
            'RETURN': [],
            'MARKET RETURN': [],
            'SCORE': []}



for sec in sector.keys():
    for ticker in sector[sec]:
        with open(f'project_2/dqn_allin/{sec}/score/{ticker}.pkl', 'rb') as f:
            rank = pickle.load(f)
            df = pd.read_csv(f'project_2/dqn_allin/data/{ticker}.csv')['stock_price']
            ret = ((df.iloc[-1] - df.iloc[0]) / df.iloc[0]) * 100

            temp_dir['SYMBOL'].append(rank[0])
            temp_dir['INDUSTRIES'].append(rank[1])
            temp_dir['RETURN'].append(float(f'{rank[3]:.2f}'))
            temp_dir['MARKET RETURN'].append(float(f'{ret:.2f}'))
            temp_dir['SCORE'].append(float(f'{rank[2]:.2f}'))

ranking = pd.DataFrame(temp_dir)
ranking = ranking.sort_values('SCORE', ascending=False)
ranking['RANK'] = [i for i in range(1, len(ranking) + 1)]
ranking.set_index('RANK', inplace=True)


fmper1 = Frame(master=frame3)
fmper1.grid(row=0, column=0, rowspan=2, padx=10, pady=10)


with sns.axes_style('white'):
    perfig1 = plt.figure(figsize=(7.5, 6), dpi=100)
    ax1 = perfig1.add_subplot(111)
    ax1.set_facecolor('#F0F0F0')
    perfig1.patch.set_facecolor('#F0F0F0')
    ranking.groupby('INDUSTRIES').mean().sort_values(by='SCORE', ascending=False).drop('SCORE', axis=1).plot.bar(
        color=['#0040ff', 'grey'], ax=ax1)
    ax1.set_ylabel('PERCENT')
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.close(perfig1)

    line1 = FigureCanvasTkAgg(perfig1, fmper1)
    line1.get_tk_widget().pack()


fmper2 = Frame(master=frame3)
fmper2.grid(row=0, column=1, padx=10, pady=10)


fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
ax.axis('equal')
width = 0.35
kwargs = dict(colors=['#0040ff', 'grey'], startangle=180)
outside, _ = ax.pie([sum(ranking["SCORE"] >= 0), 100-sum(ranking["SCORE"] > 0)], radius=1, pctdistance=1 - width / 2, labels=['Backtest', 'Market return'], **kwargs)
plt.setp(outside, width=width, edgecolor='white')

kwargs = dict(size=20, fontweight='bold', va='center')
ax.text(0, 0, f'{sum(ranking["SCORE"] >= 0)}%', ha='center', **kwargs)
ax.set_facecolor('#F0F0F0')
fig.patch.set_facecolor('#F0F0F0')
ax.set_xlabel('Win rate', fontsize=12)
plt.close(fig)
line2 = FigureCanvasTkAgg(fig, fmper2)
line2.get_tk_widget().pack()




fmper3 = Frame(master=frame3)
fmper3.grid(row=1, column=1, padx=10, pady=10)

with sns.axes_style('white'):
    figure3, ax3 = plt.subplots(figsize=(5, 3), dpi=100)
    figure3.patch.set_facecolor('#F0F0F0')
    sns.kdeplot(
        x=ranking["RETURN"],
        fill=True, common_norm=False, color="#0040ff",
        alpha=.8, linewidth=0, label='RETURN'
    )
    sns.kdeplot(
        data=ranking, x="MARKET RETURN",
        fill=True, common_norm=False, color="grey",
        alpha=.5, linewidth=0, label='MARKET RETURN'
    )
    plt.tight_layout()
    plt.close(figure3)

    ax3.set_facecolor('#F0F0F0')
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    line3 = FigureCanvasTkAgg(figure3, fmper3)
    line3.get_tk_widget().pack()







fmper5 = Frame(master=frame3)
fmper5.grid(row=2, column=0, padx=10, pady=10, columnspan=1)



def show1():

    for i, (name, ind, ret, mar_ret, score) in enumerate(ranking.values, start=1):
        listBox1.insert("", "end", values=(i, name, ind, ret, mar_ret, score))


label = Label(fmper5, text="High Scores", font=("Calibri",14)).grid(row=0, columnspan=6)
# create Treeview with 3 columns
cols = ['RANK', 'SYMBOL', 'INDUSTRIES', 'RETURN', 'MARKET RETURN', 'SCORE']
listBox1 = ttk.Treeview(fmper5,  show='headings', selectmode='none')
listBox1['columns'] = ['RANK', 'SYMBOL', 'INDUSTRIES', 'RETURN', 'MARKET RETURN', 'SCORE']

for col in cols:
    listBox1.column(col, width=125, anchor=E)


# set column headings
for col in cols:
    listBox1.heading(col, text=col)
listBox1.grid(row=1, column=0, sticky='e')

showScores1 = Button(fmper5, text="Show scores", width=15, command=show1, bg="#0040ff", fg='white').grid(row=4, column=0, pady=10)




fmper4 = Frame(master=frame3)
fmper4.grid(row=2, column=1, padx=10, pady=10, columnspan=1)

group_rank = ranking.groupby('INDUSTRIES').mean().sort_values(by='SCORE', ascending=False).reset_index()


def show2():
    for i, (ind, ret, mar_ret, score) in enumerate(group_rank.values, start=1):
        listBox2.insert("", "end", values=(i, ind, float(f'{ret:.2f}'), float(f'{mar_ret:.2f}'), float(f'{score:.2f}')))

label = Label(fmper4, text="Group by INDUSTRIES", font=("Calibri",14)).grid(row=0, columnspan=6)
cols = ['RANK', 'INDUSTRIES', 'RETURN', 'MARKET RETURN', 'SCORE']
listBox2 = ttk.Treeview(fmper4,  show='headings', selectmode='none')
listBox2['columns'] = ['RANK', 'INDUSTRIES', 'RETURN', 'MARKET RETURN', 'SCORE']



listBox2.column(cols[0], width=50, anchor=NE)
listBox2.column(cols[1], width=135, anchor=NE)
listBox2.column(cols[2], width=60, anchor=NE)
listBox2.column(cols[3], width=100, anchor=NE)
listBox2.column(cols[4], width=70, anchor=NE)

# set column headings
for col in cols:
    listBox2.heading(col, text=col)
listBox2.grid(row=1, column=0, sticky='e')

showScores2 = Button(fmper4, text="Show scores", width=15, command=show2, bg="#0040ff", fg='white').grid(row=4, column=0, pady=10)




fmper6 = Frame(master=frame3)
fmper6.grid(row=3, column=0, padx=10, pady=10, rowspan=2)


tickers = ['btc', 'eth', 'xrp', 'usdt', 'ltc', 'xlm', 'xem', 'xmr', 'dash', 'dcr', 'zec', 'etc', 'waves', 'dgb', 'sc']

temp_dir = {'SYMBOL': [],
            'INDUSTRIES': [],
            'RETURN': [],
            'MARKET RETURN': [],
            'SCORE': []}

for ticker in tickers:
    with open(f'project_2/dqn_allin/cryptocurrency/score/{ticker.upper()}-USD.pkl', 'rb') as f:
        rank = pickle.load(f)
        df = pd.read_csv(f'project_2/dqn_allin/data/{ticker.upper()}-USD.csv')['stock_price']
        ret = ((df.iloc[-1] - df.iloc[0]) / df.iloc[0]) * 100

        temp_dir['SYMBOL'].append(rank[0])
        temp_dir['INDUSTRIES'].append(rank[1])
        temp_dir['RETURN'].append(float(f'{rank[3]:.2f}'))
        temp_dir['MARKET RETURN'].append(float(f'{ret:.2f}'))
        temp_dir['SCORE'].append(float(f'{rank[2]:.2f}'))

ranking2 = pd.DataFrame(temp_dir)
ranking2 = ranking2.sort_values('SCORE', ascending=False)
ranking2['RANK'] = [i for i in range(1, len(ranking2) + 1)]
ranking2.set_index('RANK', inplace=True)

with sns.axes_style('white'):
    perfig6 = plt.figure(figsize=(7.5, 6), dpi=100)
    ax6 = perfig6.add_subplot(111)
    ax6.set_facecolor('#F0F0F0')
    perfig6.patch.set_facecolor('#F0F0F0')
    ranking2.groupby('SYMBOL').mean().sort_values(by='SCORE', ascending=False).drop('SCORE', axis=1).plot.bar(
        color=['orange', 'grey'], ax=ax6)
    ax6.set_ylabel('PERCENT')
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.close(perfig6)

    line6 = FigureCanvasTkAgg(perfig6, fmper6)
    line6.get_tk_widget().pack()


fmper7 = Frame(master=frame3)
fmper7.grid(row=3, column=1, padx=10, pady=10)


fig7, ax7 = plt.subplots(figsize=(3, 3), dpi=100)
ax7.axis('equal')
width = 0.35
kwargs = dict(colors=['orange', 'grey'], startangle=180)
outside, _ = ax7.pie([(sum(ranking2["SCORE"] >= 0)*100)/15, 100-(sum(ranking2["SCORE"] >= 0)*100)/15], radius=1, pctdistance=1 - width / 2, labels=['Backtest', 'Market return'], **kwargs)
plt.setp(outside, width=width, edgecolor='white')

kwargs = dict(size=20, fontweight='bold', va='center')
ax7.text(0, 0, f'{(sum(ranking2["SCORE"] >= 0)*100)/15:.2f}%', ha='center', **kwargs)
ax7.set_facecolor('#F0F0F0')
fig7.patch.set_facecolor('#F0F0F0')
ax7.set_xlabel('Win rate', fontsize=12)
plt.close(fig7)
line7 = FigureCanvasTkAgg(fig7, fmper7)
line7.get_tk_widget().pack()




fmper8 = Frame(master=frame3)
fmper8.grid(row=4, column=1, padx=10, pady=10)

with sns.axes_style('white'):
    figure8, ax8 = plt.subplots(figsize=(5, 3), dpi=100)
    figure8.patch.set_facecolor('#F0F0F0')
    sns.kdeplot(
        x=ranking2["RETURN"],
        fill=True, common_norm=False, color="orange",
        alpha=.8, linewidth=0, label='RETURN'
    )
    sns.kdeplot(
        data=ranking2, x="MARKET RETURN",
        fill=True, common_norm=False, color="grey",
        alpha=.5, linewidth=0, label='MARKET RETURN'
    )
    plt.tight_layout()
    plt.close(figure8)

    ax8.set_facecolor('#F0F0F0')
    ax8.spines['bottom'].set_visible(False)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.spines['left'].set_visible(False)

    line8 = FigureCanvasTkAgg(figure8, fmper8)
    line8.get_tk_widget().pack()


fmper9 = Frame(master=frame3)
fmper9.grid(row=5, column=0, padx=10, pady=10, columnspan=2)



def show3():

    for i, (name, ind, ret, mar_ret, score) in enumerate(ranking2.values, start=1):
        listBox3.insert("", "end", values=(i, name, ind, ret, mar_ret, score))


label = Label(fmper9, text="High Scores", font=("Calibri",14)).grid(row=0, columnspan=6)
# create Treeview with 3 columns
cols = ['RANK', 'SYMBOL', 'INDUSTRIES', 'RETURN', 'MARKET RETURN', 'SCORE']
listBox3 = ttk.Treeview(fmper9,  show='headings', selectmode='none')
listBox3['columns'] = ['RANK', 'SYMBOL', 'INDUSTRIES', 'RETURN', 'MARKET RETURN', 'SCORE']

for col in cols:
    listBox3.column(col, width=160, anchor=E)


# set column headings
for col in cols:
    listBox3.heading(col, text=col)
listBox3.grid(row=1, column=0, sticky='e')

showScores3 = Button(fmper9, text="Show scores", width=15, command=show3, bg="orange", fg='white').grid(row=4, column=0, pady=10)








main_frame4 = Frame(master=tab_portfolio)
main_frame4.pack(fill=BOTH, expand=1)

sf = ScrolledFrame(main_frame4)
sf.pack(side="top", expand=1, fill="both")
sf.bind_arrow_keys(main_frame4)
sf.bind_scroll_wheel(main_frame4)

frame4 = sf.display_widget(Frame)



fmport1 = Frame(master=frame4, width=1200)
fmport1.grid(row=0, column=0, padx=20, pady=30)




label = Label(fmport1, text="Symbol", font=("Calibri",14, 'bold')).grid(row=0, columnspan=1, pady=10)
cols = ['Symbol', 'Industries']
listBox4 = ttk.Treeview(fmport1,  show='headings', selectmode='extended')
listBox4['columns'] = cols

listBox4.column(cols[0], width=70, anchor=E)
listBox4.column(cols[1], width=140, anchor=E)

for col in cols:
    listBox4.heading(col, text=col)
listBox4.grid(row=1, column=0, sticky='e', rowspan=3, pady=10)


for sec in _data.keys():
    for item in _data[sec]:
        listBox4.insert("", "end", values=(item, sec))


scroll_y = Scrollbar(master=fmport1, orient=VERTICAL, command=listBox4.yview)
scroll_y.grid(row=1, column=1, sticky=N+S, rowspan=3)
listBox4.config(yscrollcommand=scroll_y.set)


addbut = Button(master=fmport1, text='Add to portfolio >>', command=lambda : add_port(), bg='#eacdcf', width=18, height=2, relief=RAISED)
addbut.grid(row=1, column=2, padx=30, pady=0)

clrbut = Button(master=fmport1, text='Clear portfolio', command=lambda : clr_port(), bg='#eacdcf', width=18, height=2, relief=RAISED)
clrbut.grid(row=2, column=2, padx=30, pady=0)

okbut = Button(master=fmport1, text='Confirm', command=lambda : ok_buttom(), bg='#eacdcf', width=18, height=2, relief=RAISED)
okbut.grid(row=3, column=2, padx=30, pady=0)



label = Label(fmport1, text="Portfolio", font=("Calibri",14, 'bold')).grid(row=0, column=3, pady=10)
cols = ['Symbol', 'Industries']
listBox5 = ttk.Treeview(fmport1,  show='headings', selectmode='none')
listBox5['columns'] = cols

listBox5.column(cols[0], width=70, anchor=E)
listBox5.column(cols[1], width=140, anchor=E)

for col in cols:
    listBox5.heading(col, text=col)
listBox5.grid(row=1, column=3, sticky='e', rowspan=3)


subfm = Frame(frame4)
subfm.grid(row=0, column=1, padx=30)


fmport2 = Frame(master=frame4)
fmport2.grid(row=1, column=0, pady=10, padx=10, columnspan=2, sticky=W)

df1 = pd.read_csv(f'project_2/dqn_allin/data/BBL.csv', index_col='Date', parse_dates=True)

with sns.axes_style('whitegrid'):
    figure1 = plt.Figure(figsize=(13, 4), dpi=100)
    figure1.patch.set_facecolor('#F0F0F0')
    ax1 = figure1.add_subplot(111)
    ax1.set_facecolor('white')
    line1 = FigureCanvasTkAgg(figure1, fmport2)
    line1.get_tk_widget().pack(side=LEFT)
    (df1['portfolio_value']/1000).plot(ax=ax1, alpha=0.7, lw=3, ls=' ')
    ax1.axhline(1, ls='--', c='black', lw=2)
    ax1.set_title('Cumulative returns')



fmport3 = Frame(master=frame4, height=200)
fmport3.grid(row=2, column=0, padx=10, pady=10, columnspan=2, sticky=W)


fmport4 = Frame(master=frame4)
fmport4.grid(row=3, column=0, padx=10, pady=10, columnspan=2)


fmport5 = Frame(master=frame4)
fmport5.grid(row=4, column=0, padx=10, pady=10, columnspan=2)


fmport6 = Frame(master=frame4)
fmport6.grid(row=5, column=0, padx=10, pady=10, columnspan=2)

simbut = Button(fmport6, text='Simulation', bg='#fdc428', width=10, height=2, command=lambda : compare(port, df1))
simbut.grid(row=0, column=0, padx=10, pady=10)

testbut = Button(fmport6, text='Backtest', bg='#1d068e', fg='white',width=10, height=2, command=lambda : testbutton(port, df1))
testbut.grid(row=0, column=1, padx=10, pady=10)

fmport7 = Frame(master=frame4)
fmport7.grid(row=6, column=0, padx=10, pady=10, columnspan=2)

fmport8 = Frame(master=frame4)
fmport8.grid(row=7, column=0, padx=10, pady=10, columnspan=2)

fmport9 = Frame(master=frame4)
fmport9.grid(row=8, column=0, padx=10, pady=10)

fmport10 = Frame(master=frame4)
fmport10.grid(row=8, column=1, padx=10, pady=10)


fmport11 = Frame(master=frame4)
fmport11.grid(row=9, column=0, padx=10, pady=10)

fmport12 = Frame(master=frame4)
fmport12.grid(row=9, column=1, padx=10, pady=10)










def add_port():

    ind = listBox4.selection()
    listBox5_item = listBox5.get_children()


    for item in ind:
        if item not in listBox5_item:
            listBox5.insert('', 'end', item, values=listBox4.item(item)['values'])


def clr_port():
    for record in listBox5.get_children():
        listBox5.delete(record)


def ok_buttom():

    listBox5_item = listBox5.get_children()

    tickers = []

    for item in listBox5_item:
        tickers.append(listBox4.item(item)['values'][0])

    global port

    port = pd.DataFrame()

    for ticker in tickers:
        temp = pd.read_csv(f'project_2/dqn_allin/data/{ticker}.csv', index_col='Date', parse_dates=True)
        port[ticker] = temp['portfolio_value']

    port.index = pd.to_datetime(temp.index)
    port = port/1000

    for widget in subfm.winfo_children():
        widget.destroy()

    subfm.pack_forget()

    color = ['#206a5d', '#81b214', '#ffcc29', '#f58634', '#d44000', '#864000']*2

    for i in range(len(tickers)):
        lb = Label(subfm, text=tickers[i], font=('Calibri', '18', 'bold'), fg=color[i]).grid(row=i, column=0)
        df = pd.read_csv(f'project_2/dqn_allin/data/{tickers[i]}.csv')['stock_price']
        figure1 = plt.Figure(figsize=(5.5, 0.6), dpi=100)
        figure1.patch.set_facecolor('#F0F0F0')
        ax1 = figure1.add_subplot(111)
        ax1.set_facecolor('#F0F0F0')
        line1 = FigureCanvasTkAgg(figure1, subfm)
        line1.get_tk_widget().grid(row=i, column=1)
        df.plot(ax=ax1, alpha=0.7, lw=2, color=color[i])
        ax1.fill_between(range(len(df)), df, df.min(), color=color[i], alpha=0.5)
        ax1.autoscale(axis='x', tight=True)
        ax1.axis('off')
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)


    for widget in fmport2.winfo_children():
        widget.destroy()

    fmport2.pack_forget()


    with sns.axes_style('whitegrid'):
        figure1 = plt.Figure(figsize=(13, 4), dpi=100)
        figure1.patch.set_facecolor('#F0F0F0')
        ax1 = figure1.add_subplot(111)
        ax1.set_facecolor('white')
        line1 = FigureCanvasTkAgg(figure1, fmport2)
        line1.get_tk_widget().pack(side=LEFT)
        port.plot(ax=ax1, alpha=0.7, lw=3)
        ax1.axhline(1, ls='--', c='black', lw=2)
        ax1.set_title('Cumulative returns')






for widget in fmport3.winfo_children():
    widget.destroy()

fmport3.pack_forget()

subframe = Frame(fmport3)
subframe.grid(row=0, column=0, rowspan=len(tickers))

num_port = Scale(master=subframe, from_=0, to=10000, resolution=500, label='Episode',
                  showvalue=TRUE, orient=VERTICAL, tickinterval=2000, length=300, troughcolor='#eacdcf')
num_port.grid(row=0, column=0, padx=30)

# cap = Scale(master=subframe, from_=0, to=10000, resolution=500, label='Capital',
#                   showvalue=TRUE, orient=VERTICAL,  length=300, troughcolor='#eacdcf')
# cap.grid(row=0, column=1)


allobut = Button(subframe, text='Allocate', bg='#eacdcf', width=10, height=2 ,command=lambda : search(port)).grid(row=1, column=0, pady=30, columnspan=2)



def search(df):



    log_ret = np.log(df / df.shift(1))
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
             'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']

    num_ports = num_port.get()
    all_weights = np.zeros((num_ports, len(df.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):
        # Create Random Weights
        weights = np.array(np.random.random(len(df.columns)))

        # Rebalance Weights
        weights = weights / np.sum(weights)

        # Save Weights
        all_weights[ind, :] = weights

        # Expected Return
        ret_arr[ind] = np.sum((log_ret.mean() * weights) * 252)

        # Expected Variance
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

    global opt_weight
    opt_weight = all_weights[sharpe_arr.argmax()]
    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]


    figure1, ax = plt.subplots(figsize=(5.5, 4), dpi=100)
    figure1.patch.set_facecolor('#F0F0F0')
    ax.set_facecolor('#F0F0F0')
    m = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
    figure1.colorbar(m, label='Sharpe Ratio')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.close(figure1)

    # Add red dot for max SR
    ax.scatter(max_sr_vol, max_sr_ret, c='red', s=100, edgecolors='black')
    line1 = FigureCanvasTkAgg(figure1, fmport3)
    line1.get_tk_widget().grid(row=0, column=1)




    labels = df.columns
    sizes = opt_weight
    explode = [0.1]* len(df.columns) # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots(figsize=(5.5, 4))
    fig1.patch.set_facecolor('#F0F0F0')
    ax1.set_facecolor('#F0F0F0')
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=color[:len(df.columns)])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    line1 = FigureCanvasTkAgg(fig1, fmport3)
    line1.get_tk_widget().grid(row=0, column=2, sticky=W)
    plt.tight_layout()
    plt.close(fig1)


def compare(df, df1):
    df['portfolio_value'] = df.sum(axis=1)
    df['return'] = df['portfolio_value']/ df['portfolio_value'].iloc[0]
    df1['benchmark'] = df1['benchmark']/df1['benchmark'].iloc[0]

    with sns.axes_style('whitegrid'):

        fig = plt.figure()

        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")  # rotate the x-axis values
        plt.subplots_adjust(bottom=0.2, top=0.9)  # ensuring the dates (on the x-axis) fit in the screen
        plt.ylabel('Cumulative return')
        plt.xlabel('Days')
        plt.title('Compare with benchmark')

        def buildmebarchart(i=int):
            plt.legend(['Benchmark', 'Backtest'] ,loc=3)
            plt.plot(df1['benchmark'].values[:i], lw=3, c='#bab8b8', label='Benchmark')
            plt.plot(df['return'].values[:i], lw=3, c='#069c24', label='Backtest')
            plt.axhline(1, ls='--', c='black', lw=2)


        animator = ani.FuncAnimation(fig, buildmebarchart, interval=20, frames=len(df1))
        plt.show()


def testbutton(df, df1):

    for widget in fmport7.winfo_children():
        widget.destroy()

    fmport7.pack_forget()

    bt_ret = df['portfolio_value'].pct_change()
    bm_ret = df1['benchmark'].pct_change()


    with sns.axes_style('whitegrid'):
        figure3 = plt.figure(figsize=(13, 4), dpi=100)
        figure3.patch.set_facecolor('#F0F0F0')
        pf.plotting.plot_rolling_returns(bt_ret, bm_ret)
        plt.title('Compare with benchmark')
        plt.close(figure3)
        line3 = FigureCanvasTkAgg(figure3, fmport7)
        line3.get_tk_widget().pack()


    for widget in fmport8.winfo_children():
        widget.destroy()

    fmport8.pack_forget()



    with sns.axes_style('whitegrid'):
        figure4 = plt.figure(1)
        figure4.patch.set_facecolor('#F0F0F0')
        plt.subplot(1, 3, 1)
        pf.plot_annual_returns(bt_ret)
        plt.subplot(1, 3, 2)
        pf.plot_monthly_returns_dist(bt_ret)
        plt.subplot(1, 3, 3)
        pf.plot_monthly_returns_heatmap(bt_ret)
        plt.tight_layout()
        figure4.set_size_inches(12, 4)
        plt.close(figure4)

        line4 = FigureCanvasTkAgg(figure4, fmport8)
        line4.get_tk_widget().pack()



    for widget in fmport9.winfo_children():
        widget.destroy()

    fmport9.pack_forget()


    with sns.axes_style('whitegrid'):
        figure5 = plt.figure(figsize=(6, 4), dpi=100)
        figure5.patch.set_facecolor('#F0F0F0')
        pf.plot_return_quantiles(bt_ret)
        plt.close(figure5)

        line5 = FigureCanvasTkAgg(figure5, fmport9)
        line5.get_tk_widget().pack()



    for widget in fmport10.winfo_children():
        widget.destroy()

    fmport10.pack_forget()


    with sns.axes_style('whitegrid'):
        figure6 = plt.figure(figsize=(6, 4.5), dpi=100)
        figure6.patch.set_facecolor('#F0F0F0')
        pf.plot_rolling_beta(bt_ret, bm_ret)
        plt.close(figure6)

        line6 = FigureCanvasTkAgg(figure6, fmport10)
        line6.get_tk_widget().pack()




    for widget in fmport11.winfo_children():
        widget.destroy()

    fmport11.pack_forget()


    with sns.axes_style('whitegrid'):
        figure7 = plt.figure(figsize=(6, 4.5), dpi=100)
        figure7.patch.set_facecolor('#F0F0F0')
        pf.plot_rolling_sharpe(bt_ret)
        plt.close(figure7)

        line7 = FigureCanvasTkAgg(figure7, fmport11)
        line7.get_tk_widget().pack()



    for widget in fmport12.winfo_children():
        widget.destroy()

    fmport12.pack_forget()


    with sns.axes_style('whitegrid'):
        figure8 = plt.figure(figsize=(6, 4.5), dpi=100)
        figure8.patch.set_facecolor('#F0F0F0')
        pf.plot_drawdown_underwater(bt_ret)
        plt.close(figure8)

        line8 = FigureCanvasTkAgg(figure8, fmport12)
        line8.get_tk_widget().pack()




main_frame5 = Frame(master=tab_predict)
main_frame5.pack(fill=BOTH, expand=1)

sf = ScrolledFrame(main_frame5)
sf.pack(side="top", expand=1, fill="both")
sf.bind_arrow_keys(main_frame5)
sf.bind_scroll_wheel(main_frame5)

frame5 = sf.display_widget(Frame)



prefm1 = Frame(master=frame5)
prefm1.grid(row=0, column=0, pady=15, padx=10, sticky=W, columnspan=2)

lb1 = Label(master=prefm1, text='Sector:')
lb1.grid(row=0, column=0, padx=10)

sector = []

for sec in _data.keys():
    sector.append(sec)

combo_sector_pe = ttk.Combobox(master=prefm1, values=sector)
combo_sector_pe.grid(row=0, column=1, pady=5)
combo_sector_pe.bind('<<ComboboxSelected>>', lambda e: combo_sector_selected_pe())
combo_sector_pe.bind('<Key>', 'break')

lb2 = Label(master=prefm1, text='Symbol:')
lb2.grid(row=0, column=2, padx=10)

combo_symbol_pe = ttk.Combobox(master=prefm1)
combo_symbol_pe.grid(row=0, column=3)
combo_sector_pe.bind('<<ComboboxSelected>>')
combo_symbol_pe.bind('<Key>', 'break')




bt1 = Button(master=prefm1, text='Select', width=10, command=lambda: selected(), bg='#28b5b5',
             font='Calibri 12 bold', fg='black')
bt1.grid(row=0, column=4, padx=20)


bt2 = Button(master=prefm1, text='Predict', width=10, command=lambda: predict(), bg='#8fd9a8',
             font='Calibri 12 bold', fg='black')
bt2.grid(row=0, column=5, padx=20)


bt3 = Button(master=prefm1, text='Performance', width=12, command=lambda: perf(), bg='#d2e69c',
             font='Calibri 12 bold', fg='black')
bt3.grid(row=0, column=6, padx=20)


prefm2 = Frame(master=frame5)
prefm2.grid(row=1, column=0, pady=15, padx=10)

prefm3 = Frame(master=frame5)
prefm3.grid(row=2, column=0, pady=15, padx=10)

prefm4 = Frame(master=frame5)
prefm4.grid(row=3, column=0, pady=15, padx=10)


#======================================================================================

def combo_sector_selected_pe():
    sel_sector_pe = combo_sector_pe.get()
    symbols = _data[sel_sector_pe]
    combo_symbol_pe.delete(0, END)
    combo_symbol_pe.config(values=symbols)
    combo_symbol_pe.current(0)




def selected():

    for widget in prefm2.winfo_children():
        widget.destroy()

    prefm2.pack_forget()

    sel_symbol = combo_symbol_pe.get()


    df5 = pd.read_csv(f'project_2/predict/plot/{sel_symbol}.csv')
    df5.index = pd.to_datetime(['2021-01-22', '2021-01-25', '2021-01-26', '2021-01-27',
               '2021-01-28', '2021-01-29', '2021-02-01'])


    with sns.axes_style('whitegrid'):
        figure1 = plt.Figure(figsize=(13, 4), dpi=100)
        figure1.patch.set_facecolor('#F0F0F0')
        ax1 = figure1.add_subplot(111)
        ax1.set_facecolor('white')
        line1 = FigureCanvasTkAgg(figure1, prefm2)
        line1.get_tk_widget().grid(row=0, column=0, columnspan=2)
        df5['true_value'].plot(ax=ax1, alpha=0.7, lw=3, marker='o', ms=7)
        df5['prediction'].plot(ax=ax1, alpha=0.7, lw=3, ls=' ')
        plt.tight_layout()
        ax1.set_title(f'Stock price {sel_symbol}')
        plt.close(figure1)



def predict():

    for widget in prefm2.winfo_children():
        widget.destroy()

    prefm2.pack_forget()


    sel_symbol = combo_symbol_pe.get()


    df5 = pd.read_csv(f'project_2/predict/plot/{sel_symbol}.csv')
    df5.index = pd.to_datetime(['2021-01-22', '2021-01-25', '2021-01-26', '2021-01-27',
               '2021-01-28', '2021-01-29', '2021-02-01'])


    with sns.axes_style('whitegrid'):
        figure1 = plt.Figure(figsize=(13, 4), dpi=100)
        figure1.patch.set_facecolor('#F0F0F0')
        ax1 = figure1.add_subplot(111)
        ax1.set_facecolor('white')
        line1 = FigureCanvasTkAgg(figure1, prefm2)
        line1.get_tk_widget().grid(row=0, column=0, columnspan=2)
        df5[['true_value', 'prediction']].plot(ax=ax1, alpha=0.7, lw=3, marker='o', ms=7)
        plt.tight_layout()
        ax1.set_title(f'Stock price {sel_symbol}')
        plt.close(figure1)

    with open(f'project_2/predict/result/{sel_symbol}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        amse, error = pickle.load(f)
    lb1 = Label(prefm2, text='Absolute Mean square error', font='Calibri 15', fg='grey').grid(row=1, column=0, pady=50)
    lb2 = Label(prefm2, text='Error', font='Calibri 15', fg='grey').grid(row=1, column=1)
    lb3 = Label(prefm2, text=f'{amse:.2f}', font='Calibri 20 bold', fg='#81b214').grid(row=2, column=0)
    lb4 = Label(prefm2, text=f'{error:.2f}%', font='Calibri 20 bold', fg='#f58634').grid(row=2, column=1)




def perf():

    for widget in prefm3.winfo_children():
        widget.destroy()

    prefm3.pack_forget()


    sector = {'Basic materials': ['VMC', 'VALE', 'SCCO', 'RIO', 'PKX', 'IFF', 'MT', 'LIN', 'BHP', 'BBL'],
              'Communication service': ['WPP', 'VZ', 'TU', 'RELX', 'OMC', 'NFLX', 'GOOG', 'EA', 'DISCA', 'DIS'],
              'Consumer cyclical': ['TM', 'SBUX', 'NKE', 'MCD', 'LVS', 'HD', 'GM', 'FORD', 'EBAY', "AMZN"],
              'Consumer defensive': ['UL', 'NWL', 'KR', 'K', 'GIS', 'DEO', 'CPB', 'CCEP', 'BTI', "BG"],
              'Energy': ['XOM', 'TOT', 'SNP', 'RDS-B', 'PTR', 'EQNR', 'CVX', 'COP', "BP", 'ENB'],
              'Financial service': ['WFC-PL', 'V', 'TD', 'MS', 'MA', 'JPM', 'GS', 'C', 'BRK-A', 'BAC-PL'],
              'Industrial': ['PCAR', 'LUV', 'JCI', 'GE', 'GD', 'DE', 'DAL', 'CMI', 'CAT', 'ABB'],
              'Healthcare': ['ZBH', 'SNY', 'PFE', 'JNJ', 'GSK', 'FMS', 'CNC', 'CI', 'CAH', 'ANTM'],
              'Real estate': ['WY', 'WPC', 'VTR', 'SPG', 'O', 'NLY', 'MPW', 'HST', 'AVB', "ARE"],
              'Technology': ['TXN', 'TSM', 'STM', 'SAP', 'MU', 'INTC', 'IBM', 'HPQ', 'CSCO', 'AAPL']}

    temp_dir = {'SYMBOL': [],
                'INDUSTRIES': [],
                'AMSE': [],
                'ERROR': []}


    for sec in sector.keys():
        for ticker in sector[sec]:
            with open(f'project_2/predict/result/{ticker}.pkl', 'rb') as f:
                rank = pickle.load(f)
                temp_dir['SYMBOL'].append(ticker)
                temp_dir['INDUSTRIES'].append(sec)
                temp_dir['AMSE'].append(rank[0])
                temp_dir['ERROR'].append(rank[1])



    ranking = pd.DataFrame(temp_dir)
    ranking = ranking.sort_values('ERROR', ascending=True)
    ranking['RANK'] = [i for i in range(1, len(ranking) + 1)]
    ranking.set_index('RANK', inplace=True)


    with sns.axes_style('white'):
        perfig6 = plt.figure(figsize=(7.5, 6), dpi=100)
        ax6 = perfig6.add_subplot(111)
        ax6.set_facecolor('#F0F0F0')
        perfig6.patch.set_facecolor('#F0F0F0')
        ranking.groupby('INDUSTRIES').mean().sort_values(by='ERROR', ascending=True)['ERROR'].plot.bar(
            color=['#206a5d', '#81b214', '#ffcc29', '#f58634', '#d2e69c', '#8fd9a8', '#28b5b5', '#4b778d', '#ff8882', '#ffc2b4'], ax=ax6)
        ax6.set_ylabel('PERCENT')
        ax6.spines['bottom'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        plt.close(perfig6)

        line6 = FigureCanvasTkAgg(perfig6, prefm3)
        line6.get_tk_widget().grid(row=0, column=0, rowspan=2)




    fig7, ax7 = plt.subplots(figsize=(2.95, 2.95), dpi=100)
    ax7.axis('equal')
    width = 0.35
    kwargs = dict(colors=['#81b214', 'grey'], startangle=180)
    outside, _ = ax7.pie([100 - ranking['ERROR'].mean(), ranking['ERROR'].mean()], radius=1, pctdistance=1 - width / 2, labels=['Precision', 'Error'], **kwargs)
    plt.setp(outside, width=width, edgecolor='white')

    kwargs = dict(size=20, fontweight='bold', va='center')
    ax7.text(0, 0, f'{100 - ranking["ERROR"].mean():.2f}%', ha='center', **kwargs)
    ax7.set_facecolor('#F0F0F0')
    fig7.patch.set_facecolor('#F0F0F0')
    ax7.set_xlabel('Precision', fontsize=12)
    plt.tight_layout()
    plt.close(fig7)
    line7 = FigureCanvasTkAgg(fig7, prefm3)
    line7.get_tk_widget().grid(row=0, column=1)



    with sns.axes_style('white'):
        figure8, ax8 = plt.subplots(figsize=(5, 3), dpi=100)
        figure8.patch.set_facecolor('#F0F0F0')
        sns.kdeplot(
            x=ranking["ERROR"],
            fill=True, common_norm=False, color="#81b214",
            alpha=.8, linewidth=0, label='RETURN'
        )

        plt.tight_layout()
        plt.close(figure8)

        ax8.set_facecolor('#F0F0F0')
        ax8.spines['bottom'].set_visible(False)
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['left'].set_visible(False)

        line8 = FigureCanvasTkAgg(figure8, prefm3)
        line8.get_tk_widget().grid(row=1, column=1)



    def show5():

        for i, (name, ind, a, e) in enumerate(ranking.values, start=1):
            listBox5.insert("", "end", values=(i, name, ind, f'{a:.2f}',f'{e:.2f}'))


    label = Label(prefm4, text="Ranking", font=("Calibri",14)).grid(row=0, columnspan=6)
    # create Treeview with 3 columns
    cols = ['RANK', 'SYMBOL', 'INDUSTRIES', 'AMSE', 'ERROR']
    listBox5 = ttk.Treeview(prefm4,  show='headings', selectmode='none')
    listBox5['columns'] = ['RANK', 'SYMBOL', 'INDUSTRIES', 'AMSE', 'ERROR']

    for col in cols:
        listBox5.column(col, width=160, anchor=E)


    # set column headings
    for col in cols:
        listBox5.heading(col, text=col)
    listBox5.grid(row=1, column=0, sticky='e')

    showScores5 = Button(prefm4, text="Show ranking", width=15, command=show5, bg="#81b214", fg='white').grid(row=4, column=0, pady=10)




mainloop()