import networkx as nx 
import pandas as pd 
import numpy as np
from distro import Distribution
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
    net = nx.from_pandas_edgelist(  df,
                                    source='SOURCE',
                                    target='TARGET',
                                    edge_attr='RATING',
                                    create_using=nx.DiGraph)


    #############################   ALL RATINGS    ##############################

    # In/Out degree dataframes (all ratings)
    in_degree_df = pd.DataFrame(net.in_degree, columns=['node', 'degree']).sort_values(by='node')
    out_degree_df = pd.DataFrame(net.out_degree, columns=['node', 'degree']).sort_values(by='node')

    # Mean on in/out degree (they should be the same)
    mean_in_degree = in_degree_df['degree'].mean()
    mean_out_degree = out_degree_df['degree'].mean()
    
    # Distributions of in and out degrees 
    in_distro = Distribution.distributions(in_degree_df)
    out_distro = Distribution.distributions(out_degree_df)

    #plot_pdf(in_distro, out_distro)
    #plot_ccdf(in_distro, out_distro)
    #plot_cc(df)    

    #CC_vs_degree(net)
    #cc_distribution(net)
    #plot_top10(df)

    plot_in_k_users(df)
    plot_out_k_users(df)

def plot_in_k_users(df):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    top_10 = Distribution.top_5_users_k(df)

    for i in top_10:
        t1,t2, y1,y2 = Distribution.k_vs_t(i, df)
        plt.plot(t1, y1, label=f'User {i}')
    
    plt.ylabel('Normalized Network Growth')
    plt.title('Top 5 User In-Degree over Time', fontsize=16)
    plt.legend()
    plt.show()

def plot_out_k_users(df):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    top_10 = Distribution.top_5_users_k(df)

    for i in top_10:
        t1,t2, y1,y2 = Distribution.k_vs_t(i, df)
        plt.plot(t2, y2, label=f'User {i}', linestyle='dashed')
    
    plt.ylabel('Normalized Network Growth')
    plt.title('Top 5 User Out-Degree over Time', fontsize=16)
    plt.legend()
    plt.show()
    
def network_vs_time(df):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    time, num_edges, num_ratings = Distribution.network_growth(df)
    plt.plot(time, num_edges, label = 'Edges')
    plt.plot(time, num_ratings, label = 'Ratings')
    plt.ylabel('Normalized Network Growth')
    plt.title('Normalized Edge and Ratings Over Time', fontsize=16)
    plt.legend()
    plt.show()

def cc_distribution(net):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

    cc = nx.clustering(net)
    df_cc = pd.DataFrame(cc.items(), columns = ['node', 'cc'])
    plt.hist(df_cc['cc'], bins=10, color='c', edgecolor='k',  alpha=0.5) 
    plt.xlabel('CC')
    plt.title('Clustering Coefficient (CC) Distribution', fontsize = 16)
    plt.show()



# Plot top 10 rated users
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


def CC_vs_degree(net):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

    x,y = Distribution.cc_by_degree(net)

    plt.scatter(x,y,  s = 14)
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'$\left< C(k) \right>$', fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.title(r'Average Clustering Coefficient vs Degree for $G$',  fontsize=16)
    plt.show()



 # Plots probability density function (PDF)
def plot_pdf(in_degree_dist, out_degree_dist):

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    # Plotting data 
    fig1 = plt.figure(1, figsize=(8, 6), dpi=100)

    # Scatter plot 
    plt.scatter(in_degree_dist.index, in_degree_dist['p'], s = 12, label='In-degree') 
    plt.scatter(out_degree_dist.index, out_degree_dist['p'], s = 12, label='Out-degree') 
    
    # Plot settings
    plt.xlabel(r'$k$', fontsize=16), plt.ylabel(r'$p_k$', fontsize=16)
    plt.yscale('log'), plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title(r"G Degree Distribution", fontsize=18)
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
    plt.scatter(in_degree_dist.index, in_degree_dist['ccdf'], s = 12, label='In-degree') 
    plt.plot(in_degree_dist.index, in_degree_dist['ccdf']) 

    plt.scatter(out_degree_dist.index, out_degree_dist['ccdf'], s = 12, label='Out-degree') 
    plt.plot(in_degree_dist.index, in_degree_dist['ccdf']) 

    # Plot settings
    plt.xlabel(r'$k$', fontsize=16), plt.ylabel(r'$CCDF$', fontsize=16)
    plt.yscale('log'), plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title("CCDF Degree Distribution", fontsize=18)
    plt.show() 



if __name__ == '__main__':
    main()
