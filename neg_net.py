import networkx as nx 
import pandas as pd 
from distro import Distribution
import numpy as np
import matplotlib.pyplot  as plt 


def main(): 
    
    # Load data into a dataframe
    df = pd.read_csv('bitcoinotc.csv', sep=',')

    # Convert time from epoch to readable dates
    df['TIME'] = pd.to_datetime(df['TIME'],unit='s')

    # Load data into graph 
    neg_ratings_df = df[df['RATING'] < 0]
    neg_net = nx.from_pandas_edgelist(  neg_ratings_df,
                                        source='SOURCE',
                                        target='TARGET',
                                        edge_attr='RATING',
                                        create_using=nx.DiGraph)


#############################   NEGATIVE RATINGS    ##############################

    neg_in_degree_df = pd.DataFrame(neg_net.in_degree, columns=['node', 'degree']).sort_values(by='node')
    neg_out_degree_df = pd.DataFrame(neg_net.out_degree, columns=['node', 'degree']).sort_values(by='node')
    neg_mean_in_degree = neg_in_degree_df['degree'].mean()
    neg_mean_out_degree = neg_out_degree_df['degree'].mean()


    neg_rate_CC = pd.DataFrame(nx.clustering(neg_net).items(), columns=['node', 'CC']).sort_values(by='node')
    avg_neg_CC = neg_rate_CC['CC'].mean()

    # Distributions of in and out degrees 
    in_distro = Distribution.distributions(neg_in_degree_df)
    out_distro = Distribution.distributions(neg_out_degree_df)

    #plot_pdf(in_distro, out_distro)
    #plot_ccdf(in_distro, out_distro)


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
                color= 'red') 
    plt.scatter(out_degree_dist.index, out_degree_dist['p'], 
                s = 12, 
                label='Out-degree',
                color='blue') 
    
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
                color='red') 
    plt.plot(in_degree_dist.index, in_degree_dist['ccdf'], color ='red') 

    plt.scatter(out_degree_dist.index, 
                out_degree_dist['ccdf'], 
                s = 12, 
                label='Out-degree',
                color='blue') 
    plt.plot(out_degree_dist.index, out_degree_dist['ccdf'], color='blue') 

    # Plot settings
    plt.xlabel(r'$k$', fontsize=16), plt.ylabel(r'$CCDF$', fontsize=16)
    plt.yscale('log'), plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.title("CCDF of $L_{-}$", fontsize=18)
    plt.show() 



if __name__ == '__main__':
    main()
