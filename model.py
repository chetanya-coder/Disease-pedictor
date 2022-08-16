import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
sns.set_style("darkgrid")
data = pd.read_csv("Testing.csv")
data.head(10)
def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior
def calculate_likelihood_multinomial(df, feat_name, feat_val, Y, label):
    #Alpha value for smoothing
  a = 0.001

  #Calculate probability of each word based on class
  pb_ij = df.groupby(['symptoms','disease'])
  pb_j = df.groupby(['disease'])
  Pr =  (pb_ij['count'].sum() + a) / (pb_j['count'].sum() + 142)    

#Unstack series
  Pr = Pr.unstack()


#Convert to dictionary for greater speed
  p_x_given_y= Pr.to_dict() 
  return p_x_given_y
def naive_bayes_multinomial(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_multinomial(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=.2, random_state=41)

X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_multinomial(train, X=X_test, Y="diagnosis")
gnb=naive_bayes_multinomial()
joblib.dump(gnb,'model.pkl')


