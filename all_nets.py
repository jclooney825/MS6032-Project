import networkx as nx 
import pandas as pd 
import numpy as np
from distro import Distribution
import matplotlib.pyplot  as plt 



df = pd.read_csv('bitcoinotc.csv', sep=',')

net = nx.from_pandas_edgelist(  df,
                                    source='SOURCE',
                                    target='TARGET',
                                    edge_attr='RATING',
                                    create_using=nx.DiGraph)



pos_ratings_df = df[df['RATING'] > 0]
pos_net = nx.from_pandas_edgelist(  pos_ratings_df,
                                        source='SOURCE',
                                        target='TARGET',
                                        edge_attr='RATING',
                                        create_using=nx.DiGraph)

neg_ratings_df = df[df['RATING'] < 0]
neg_net = nx.from_pandas_edgelist(  neg_ratings_df,
                                        source='SOURCE',
                                        target='TARGET',
                                        edge_attr='RATING',
                                        create_using=nx.DiGraph)

def plot_pref_attach(net1, net2, net3):

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    k_i1, pi_i1 = Distribution.pref_attach(net1)
    k_i2, pi_i2 = Distribution.pref_attach(net2)
    k_i3, pi_i3 = Distribution.pref_attach(net3)

    plt.scatter(k_i1, pi_i1, s = 10, label = '$G$')
    plt.scatter(k_i2, pi_i2, s = 10, label = '$L_{+}$', color = 'green')
    plt.scatter(k_i3, pi_i3, s = 10, label = '$L_{-}$', color = 'red')

    plt.xlabel('$k$', fontsize=13)
    plt.ylabel('$\pi(k)$', fontsize=13)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.title('Cumulative Preferential Attachment', fontsize=16)
    plt.show()

#plot_pref_attach(net, pos_net, neg_net)
