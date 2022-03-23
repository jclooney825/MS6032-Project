from matplotlib.pyplot import axis
import pandas as pd 
import numpy as np

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