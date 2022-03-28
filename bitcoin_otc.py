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

    CC_vs_degree(net)
   
    #plot_top10(df)

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



def CC_over_time(df):
    arr = []
    for i in range(10,len(df['TIME']), 500):
        data = df[:i]
        net = nx.from_pandas_edgelist(  data,
                                        source='SOURCE',
                                        target='TARGET',
                                        edge_attr='RATING',
                                        create_using=nx.DiGraph)
        cc = nx.average_clustering(net)
        arr.append(cc)
    return arr 

def plot_cc(df):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

    indicies = np.arange(10,len(df['TIME']), 500)
    time = df['TIME'][indicies]
    arr = CC_over_time(df)

    plt.plot(time,arr)
    plt.ylabel(r'$\left< C \right>$', fontsize=16)
    plt.grid(True)
    plt.title(r'Average Clustering Coefficient over Time')
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
