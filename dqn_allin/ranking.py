import pickle
import pandas as pd
from sqlalchemy import create_engine, types


engine = create_engine('mysql://root:@localhost/dqn_ranking')  # enter your password and database names here


sector = {'basic materials': ['VMC', 'VALE', 'SCCO', 'RIO', 'PKX', 'IFF', 'MT', 'LIN', 'BHP', 'BBL'],
          'communication service': ['WPP', 'VZ', 'TU', 'RELX', 'OMC', 'NFLX', 'GOOG', 'EA', 'DISCA', 'DIS'],
          'consumer cyclical': ['TM', 'SBUX', 'NKE', 'MCD', 'LVS', 'HD', 'GM', 'FORD', 'EBAY', "AMZN"],
          'consumer defensive': ['UL', 'NWL', 'KR', 'K', 'GIS', 'DEO', 'CPB', 'CCEP', 'BTI', "BG"],
          'energy': ['XOM', 'TOT', 'SNP', 'RDS-B', 'PTR', 'EQNR', 'CVX', 'COP', "BP", 'ENB'],
          'financial service': ['WFC-PL', 'V', 'TD', 'MS', 'MA', 'JPM', 'GS', 'C', 'BRK-A', 'BAC-PL'],
          'industrial': ['PCAR', 'LUV', 'JCI', 'GE', 'GD', 'DE', 'DAL', 'CMI', 'CAT', 'ABB'],
          'health care': ['ZBH', 'SNY', 'PFE', 'JNJ', 'GSK', 'FMS', 'CNC', 'CI', 'CAH', 'ANTM'],
          'real estate': ['WY', 'WPC', 'VTR', 'SPG', 'O', 'NLY', 'MPW', 'HST', 'AVB', "ARE"],
          'technology': ['TXN', 'TSM', 'STM', 'SAP', 'MU', 'INTC', 'IBM', 'HPQ', 'CSCO', 'AAPL']}

temp_dir = {'SYMBOL': [],
            'INDUSTRIES': [],
            'RETURN': [],
            'SCORE': []}

for sec in sector.keys():
    for ticker in sector[sec]:
        with open(f'{sec}/score/{ticker}.pkl', 'rb') as f:
            rank = pickle.load(f)
            print(rank)
            temp_dir['SYMBOL'].append(rank[0])
            temp_dir['INDUSTRIES'].append(rank[1])
            temp_dir['RETURN'].append(rank[3])
            temp_dir['SCORE'].append(rank[2])

ranking = pd.DataFrame(temp_dir)
ranking = ranking.sort_values('SCORE', ascending=False)
ranking['RANK'] = [i for i in range(1, len(ranking) + 1)]
ranking.set_index('RANK', inplace=True)

# ranking.to_sql('ranking', con=engine, if_exists='replace')

