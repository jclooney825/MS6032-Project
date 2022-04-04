from audioop import reverse
from re import I
from matplotlib.pyplot import axis
import pandas as pd 
import numpy as np
import networkx as nx 

'''
James Clooney 
MS6021 
Networks and Complex Systems


            Distributions Class
-------------------------------------------
Used to easily compute the distributions and 
other quantities given a dataframe object. 
'''

class Distribution:

    def distributions(df):

        degree_count = df.groupby('degree')['degree'].count()

        # Number of nodes with k degree 
        degree_dist = pd.DataFrame({'n':degree_count})

        # Normalization
        degree_dist['p'] = degree_dist['n']/degree_dist['n'].sum()
  
        # CDF of the data
        degree_dist['cdf'] = np.cumsum(degree_dist['p'])

        # CCDF of the data
        degree_dist['ccdf'] = 1-degree_dist['cdf']

        return degree_dist
    

    # Return the overall ratings for each node in the network
    def overall_ratings(df):

        # Sort by Target
        df = df.sort_values(by='TARGET')

        # All node values 
        node_vals = df['TARGET'].unique()
        
        # Finds total rating for a node i 
        f = lambda i: np.sum(df[df['TARGET'] == i]['RATING'])

        # List of all ratings in order of node number 
        tot_ratings = np.array([f(i) for i in node_vals])

        #Store data in dataframe 
        data = zip(node_vals, tot_ratings)
        tot_ratings_df =pd.DataFrame(data, columns=['node', 'overall_rating'])
        
        return tot_ratings_df


    # Returns the degree values and average cc for the degree 
    def cc_by_degree(graph):
        # Convert graph to undirected 
        graph = graph.to_undirected()

        # Calculate clustering coeff. for all nodes 
        cc = nx.clustering(graph)
        
        # Create dataframe with node number, clustering coeffcient, and degree 
        df_cc = pd.DataFrame(cc.items(), columns = ['node', 'cc'])
        df_cc['degree'] =  [i[1] for i in graph.degree()]

        # Find all unique values of degrees
        degrees = pd.unique(df_cc['degree'])

        # Average cc for all unique degrees 
        f = lambda i: df_cc[df_cc['degree'] == i]
        y = [np.mean(f(i)['cc']) for i in degrees]

        return degrees,y 


    def network_growth(df):

        time = df['TIME']

        sum_edges = np.cumsum(df['SOURCE']/df['SOURCE'])
        norm_edges = (sum_edges - sum_edges.min()) / (sum_edges.max() - sum_edges.min())

        sum_ratings = np.cumsum(df['RATING'])
        norm_ratings = (sum_ratings  -  sum_ratings.min()) / (sum_ratings.max() -  sum_ratings.min())

        return time, norm_edges, norm_ratings


    # Return top 10 users with highest number of interactions on network 
    def top_5_users_k(df):

        users = df['SOURCE'].unique() 
        
        tot_inter = [] 
        for i in users:
            num_ratings_out = len((df[df['SOURCE'] == i])['TARGET']) 
            num_ratings_in = len((df[df['TARGET'] == i])['SOURCE']) 
            
            tot = num_ratings_out + num_ratings_in 
            tot_inter.append(tot)

        data = list(zip(users, tot_inter))
        info = pd.DataFrame(data, columns=['User', 'Num Interactions'])
        sorted_users = info.sort_values('Num Interactions', ascending=False)    
        return (sorted_users.head(5))['User']

        
    def k_vs_t(user, df):

        in_user_data = df[df['SOURCE'] == user]
        out_user_data = df[df['TARGET'] == user]

        k_in = np.cumsum(in_user_data['SOURCE'] / in_user_data['SOURCE'])
        k_out = np.cumsum(out_user_data['TARGET'] / out_user_data['TARGET'])

        t_in = in_user_data['TIME']
        t_out = out_user_data['TIME']

        return t_in, t_out, k_in ,k_out