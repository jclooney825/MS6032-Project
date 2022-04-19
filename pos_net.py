import networkx as nx 
import pandas as pd 
from distro import Distribution
import numpy as np
import matplotlib.pyplot  as plt 


'''
James Clooney 
MS6021 
Networks and Complex Systems

Analysis of the bitcoin-otc web of trust.

Data provided by:
Stanford Large Network Dataset Collection
http://snap.stanford.edu/data/index.html

'''

def main(): 
    
    # Load data into a dataframe
    df = pd.read_csv('bitcoinotc.csv', sep=',')

    # Convert time from epoch to readable dates
    df['TIME'] = pd.to_datetime(df['TIME'],unit='s')

    # Load data into graph 
    pos_ratings_df = df[df['RATING'] > 0]
    pos_net = nx.from_pandas_edgelist(  pos_ratings_df,
                                        source='SOURCE',
                                        target='TARGET',
                                        edge_attr='RATING',
                                        create_using=nx.DiGraph)


    #############################   POSITIVE RATINGS    ##############################

    pos_in_degree_df = pd.DataFrame(pos_net.in_degree, columns=['node', 'degree']).sort_values(by='node')
    pos_out_degree_df = pd.DataFrame(pos_net.out_degree, columns=['node', 'degree']).sort_values(by='node')
    pos_mean_in_degree = pos_in_degree_df['degree'].mean()
    pos_mean_out_degree = pos_out_degree_df['degree'].mean()


    pos_rate_CC = pd.DataFrame(nx.clustering(pos_net).items(), columns=['node', 'CC']).sort_values(by='node')
    avg_pos_CC = pos_rate_CC['CC'].mean()

    # Distributions of in and out degrees 
    in_distro = Distribution.distributions(pos_in_degree_df)
    out_distro = Distribution.distributions(pos_out_degree_df)

    #plot_pdf(in_distro, out_distro)
    #plot_ccdf(in_distro, out_distro)
    #plot_top10(df)

    #CC_vs_degree(pos_net)

def CC_vs_degree(net):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

    x,y = Distribution.cc_by_degree(net)

    plt.scatter(x,y,  s = 12, color = 'green')
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'$\left< C(k) \right>$', fontsize=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.title(r'Average Clustering Coefficient vs Degree for $L_{+}$',  fontsize=16)
    plt.show()


def plot_top10(df):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

    data = Distribution.overall_ratings(df)
    data = data.sort_values(by='overall_rating', ascending=False)

    data['node'] = data['node'].astype(str)
    users = data[0:9]

    plt.bar(users['node'], users['overall_rating'], color='springgreen')
    plt.xlabel(r'ID', fontsize=16), plt.ylabel(r'Overall Rating', fontsize=16)
    plt.title(r'Top 10 Rated Users', fontsize=18)
    plt.show()






 # Plots probability density function (PDF)
def plot_pdf(in_degree_dist, out_degree_dist):
    plt.rcParams['text.usetex'] = True
    # Plotting data 
    fig1 = plt.figure(1, figsize=(8, 6), dpi=100)

    # Scatter plot 
    plt.scatter(in_degree_dist.index, 
                in_degree_dist['p'], 
                s = 12, 
                label='In-degree',
                color='green') 
    plt.scatter(out_degree_dist.index, 
                out_degree_dist['p'], 
                s = 12, 
                label='Out-degree', 
                color='purple') 
    
    # Plot settings
    plt.xlabel(r'$k$', fontsize=16), plt.ylabel(r'$p_k$', fontsize=16)
    plt.yscale('log'), plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title("$L_{-}$ Degree Distribution", fontsize=18)
    plt.show() 

 # Plots probability density function (PDF)
def plot_ccdf(in_degree_dist, out_degree_dist):
    plt.rcParams['text.usetex'] = True
    # Plotting data 
    fig2 = plt.figure(2, figsize=(8, 6), dpi=100)
    
    # Drop last data entry 
    in_degree_dist = in_degree_dist[:-1]
    out_degree_dist = out_degree_dist[:-1]

    # Plot in and oit degree CCDFs
    plt.scatter(in_degree_dist.index, 
                in_degree_dist['ccdf'], 
                s = 12, 
                label='In-degree', 
                color='green') 
    plt.plot(in_degree_dist.index, in_degree_dist['ccdf'], color= 'green') 

    plt.scatter(out_degree_dist.index,
                out_degree_dist['ccdf'], 
                s = 12, 
                label='Out-degree', 
                color= 'purple') 
    plt.plot(out_degree_dist.index, out_degree_dist['ccdf'], color= 'purple') 

    # Plot settings
    plt.xlabel(r'$k$', fontsize=16), plt.ylabel(r'$CCDF$', fontsize=16)
    plt.yscale('log'), plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title("CCDF of $L_{+}$", fontsize=18)
    plt.show() 



if __name__ == '__main__':
    main()
