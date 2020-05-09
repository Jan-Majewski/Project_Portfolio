#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as random

# plotly standard imports
import plotly.graph_objs as go
import chart_studio.plotly as py
from datetime import datetime
from datetime import timedelta  
# Cufflinks wrapper on plotly
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot
cufflinks.go_offline()

# Set global theme
cufflinks.set_config_file(world_readable=True, theme='pearl')
import plotly.figure_factory as ff


# In[2]:


def binarize_labels(y, threshold):
    y=np.where(y<threshold,1,0)
    return(y)




def append_split_class(df, train_size, test_size):
    
    
    loan_ID_list=df.loan_ID.unique()
    np.random.shuffle(loan_ID_list)
    
    n=len(loan_ID_list)
    n_train=int(train_size*n)
    n_test=int(test_size*n)
   
    
    train_IDs=loan_ID_list[:n_train]
    test_IDs=loan_ID_list[n_train:n_train+n_test]
    val_IDs=loan_ID_list[n_train+n_test:]
    
    
    df["train"]=df.loan_ID.apply(lambda x: x in train_IDs)
    df["test"]=df.loan_ID.apply(lambda x: x in test_IDs)
    df["val"]=df.loan_ID.apply(lambda x: x in val_IDs)
    
    return(df)

from sklearn.preprocessing import StandardScaler

def df_to_rnn_matrix(df, min_payments,RNN_features, window_size):

    df_rnn=df[df.period_max>=min_payments][RNN_features]
    ID_list=df_rnn.loan_ID.unique()
    
    for loan_id in ID_list:
        
        
        X_temp=df_rnn[df_rnn.loan_ID==loan_id][RNN_features[2:]].values
        X_temp=X_temp.reshape((1,X_temp.shape[0],X_temp.shape[1]))
        
        vector_len=X_temp.shape[1]

        for i in range(0, vector_len-window_size+1):
            if i==0:
                X_windowed=X_temp[:,i:window_size+i,:]
            else:
                X_temp2=X_temp[:,i:window_size+i,:]
                X_windowed=np.concatenate([X_windowed, X_temp2], 0)
        
    
        
        if loan_id==ID_list[0]:
            X=X_windowed
        else:
            X=np.concatenate([X, X_windowed], 0)
        
    return(X)


def shuffle_arrays(X,y):
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    
    X=X[indices,:,:]
    y=y[indices]
    return(X,y)

def sequences_train_test_split(X,minimum_payments):

    y=X[:,minimum_payments-1,-1]
    y=y.reshape(y.shape[0],1)
    X=X[:,:-1,:]
    

    X, y=shuffle_arrays(X, y)


    

    
    return(X,y)



def smooth_curve(points, factor=0.8): #this function will make our plots more smooth
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous*factor+point*(1-factor))
		else:
			smoothed_points.append(point)
	return smoothed_points





from sklearn.metrics import precision_score, recall_score, accuracy_score


def precision_recall_threshold(y_true, y_pred,threshold):
    y_true_cl=np.where(y_true<threshold,1,0)
    y_pred_cl=np.where(y_pred<threshold,1,0)
    predicted_default= y_pred_cl.sum()/ len(y_pred_cl)
    actual_default= y_true_cl.sum()/ len(y_true_cl)
    acc=accuracy_score(y_true_cl, y_pred_cl)
    recall=recall_score(y_true_cl, y_pred_cl, average="binary")
    precision=precision_score(y_true_cl, y_pred_cl, average="binary")
    print("Actual defaults ratio: {}\n".format(actual_default))
    print("Predicted defaults ratio: {}\n".format(predicted_default))
    print("accuracy: {}\n".format(acc))
    print("precision: {}\n".format(precision))
    print("recall: {}\n".format(recall))
    
    
    
def precision_recall_curve(df):
    recall=[]
    precission=[]
    idx=[]
    false_positive=[]
    true_positive=[]
    positives_share=[]
    y_true=df.y_true
    y_pred=df.y_pred
    
    for i in range(0,99,1):
        y_true_cl=y_true
        y_pred_cl=np.where(y_pred>=i/100,1,0)
 
        r=recall_score(y_true_cl, y_pred_cl, average="binary")
        p=precision_score(y_true_cl, y_pred_cl, average="binary")
        
        fp=np.logical_and(y_true_cl==0,y_pred_cl==1).sum()
        tp=np.logical_and(y_true_cl==1,y_pred_cl==1).sum()
        

        
        false_positive.append(fp)
        true_positive.append(tp)
        share_of_positives=(tp+fp)/len(y_true)
      
     
        
        
        
        
        
        recall.append(r)
        precission.append(p)
        idx.append(i/100)
        positives_share.append(share_of_positives)
        
    df_pr=pd.DataFrame(precission, columns={"precission"})
    df_pr["recall"]=recall
    df_pr["threshold"]=idx
    df_pr["false_positive"]=false_positive
    df_pr["true_positive"]=true_positive
    
    df_pr["false_accusation_ratio"]=df_pr.false_positive/ df_pr.true_positive
    df_pr["positives_share"]=positives_share
    
    return(df_pr)




n_steps = 10-1
def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, 0, 2])
    
def prediction_df(X,y,model):
    y_pred = model.predict(X)
    df_pred=pd.DataFrame(y_pred,columns=["y_pred"])
    df_pred["y_true"]=pd.DataFrame(y) 
    df_pred["ones"]=1
    df_pred.sort_values(by="y_true", inplace=True)
    df_pred["true_rank"]=df_pred.ones.cumsum()
    df_pred.sort_values(by="y_pred", inplace=True)
    df_pred["pred_rank"]=df_pred.ones.cumsum()

    df_pred.drop(columns="ones",inplace=True)
    
    return(df_pred)

    
def prediction_proba_df(X,y,model):
    y_pred = model.predict_proba(X)[:,1]
    df_pred=pd.DataFrame(y_pred,columns=["y_pred"])
    df_pred["y_true"]=pd.DataFrame(y) 
    df_pred["ones"]=1
    df_pred.sort_values(by="y_true", inplace=True)
    df_pred["true_rank"]=df_pred.ones.cumsum()
    df_pred.sort_values(by="y_pred", inplace=True)
    df_pred["pred_rank"]=df_pred.ones.cumsum()

    df_pred.drop(columns="ones",inplace=True)
    
    return(df_pred)

def prediction_IF_df(X,y,model):
    y_pred = -model.score_samples(X)
    df_pred=pd.DataFrame(y_pred,columns=["y_pred"])
    df_pred["y_true"]=pd.DataFrame(y) 
    df_pred["ones"]=1
    df_pred.sort_values(by="y_true", inplace=True)
    df_pred["true_rank"]=df_pred.ones.cumsum()
    df_pred.sort_values(by="y_pred", inplace=True)
    df_pred["pred_rank"]=df_pred.ones.cumsum()

    df_pred.drop(columns="ones",inplace=True)
    
    return(df_pred)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def prediction_Autoencoder_df(X,y,model,bins_input,labels):
    X_pred=model.predict(X)
    mse=np.mean(np.power(X - X_pred, 2), axis=1).reshape(-1, 1)
    
    if bins_input==False:
    
        bins=[]
        labels=[]
        for i in range(0,100):
            bins.append(np.quantile(mse[:,0],i/100))
            labels.append(i/100)
        labels=labels[:-1]
    else:
        bins=bins_input
        labels=labels
    
    
    mse_scaled=np.asarray(pd.cut(mse[:,0], bins,labels=labels))
    mse_scaled.reshape(mse.shape[0],1)
    
    y_pred = mse_scaled
    df_pred=pd.DataFrame(y_pred,columns=["y_pred"])
    df_pred["y_true"]=pd.DataFrame(y) 
    df_pred["ones"]=1
    df_pred.sort_values(by="y_true", inplace=True)
    df_pred["true_rank"]=df_pred.ones.cumsum()
    df_pred.sort_values(by="y_pred", inplace=True)
    df_pred["pred_rank"]=df_pred.ones.cumsum()

    df_pred.drop(columns="ones",inplace=True)
    if bins_input==False:
        
        return df_pred,bins,labels
    else:
      
        return df_pred


# In[3]:


def precision_recall_curve(df):
    recall=[]
    precission=[]
    idx=[]
    false_positive=[]
    true_positive=[]
    false_negative=[]
    positives_share=[]
    y_true=df.y_true
    y_pred=df.y_pred
    
    for i in range(0,100,1):
        y_true_cl=y_true
        y_pred_cl=np.where(y_pred>=i/100,1,0)
 
        
        fp=np.logical_and(y_true_cl==0,y_pred_cl==1).sum()
        tp=np.logical_and(y_true_cl==1,y_pred_cl==1).sum()
        
        fn=np.logical_and(y_true_cl==1,y_pred_cl==0).sum()
        
        if (tp+fn)==0:
            r=0
        else:
            r=tp/(tp+fn)
            
        if (tp+fp)==0:
            p=0
        else:
            p=tp/(tp+fp)
        

        
        false_positive.append(fp)
        true_positive.append(tp)
        share_of_positives=(tp+fp)/len(y_true)
      
     
        
        
        
        
        
        recall.append(r)
        precission.append(p)
        idx.append(i/100)
        positives_share.append(share_of_positives)
        
    df_pr=pd.DataFrame(precission, columns={"precission"})
    df_pr["recall"]=recall
    df_pr["threshold"]=idx
    df_pr["false_positive"]=false_positive
    df_pr["true_positive"]=true_positive

    
    df_pr["false_accusation_ratio"]=df_pr.false_positive/ df_pr.true_positive
    df_pr["positives_share"]=positives_share
    df_pr["search_multiplier"]=df_pr["recall"]/df_pr["positives_share"]
    
    return(df_pr)


# In[4]:


def plot_recall_vs_positives(df_pr):
    trace1=go.Scatter(
            y=df_pr.recall,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="red",
            size=5,
            opacity=0.5
            ),
            name="Model"
        )


    trace2=go.Scatter(
            y=df_pr.positives_share,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="blue",
            size=5,
            opacity=0.5
            ),
            name="Random search"
        )






    data=[trace1,trace2]
    figure=go.Figure(
        data=data,
        layout=go.Layout(
            title="Recall vs Positives share",
            yaxis=dict(title="Recall"),
            xaxis=dict(title="Positives share",range=[0,0.4]),
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
            bgcolor=None


        )))
    iplot(figure)


# In[5]:


def plot_recall_surplus(df_pr):
    trace1=go.Scatter(
            y=df_pr.recall-df_pr.positives_share,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="red",
            size=5,
            opacity=0.5
            ),
            name="Model"
        )






    data=[trace1]
    figure=go.Figure(
        data=data,
        layout=go.Layout(
            title="Recall surplus vs Positives share",
            yaxis=dict(title="Recall surplus"),
            xaxis=dict(title="Positives share",range=[0,0.4]),
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
            bgcolor=None


        )))
    iplot(figure)


# In[6]:


def plot_recall_surplus_train_test(df_pr_train,df_pr_test):
    trace0=go.Scatter(
            y=df_pr_train.recall-df_pr_train.positives_share,
            x=df_pr_train.positives_share,
            mode='lines',
            marker=dict(
            color="red",
            size=5,
            opacity=0.5
            ),
            name="Training data"
        )

    trace1=go.Scatter(
            y=df_pr_test.recall-df_pr_test.positives_share,
            x=df_pr_test.positives_share,
            mode='lines',
            marker=dict(
            color="blue",
            size=5,
            opacity=0.5
            ),
            name="Test data"
        )






    data=[trace0,trace1]
    figure=go.Figure(
        data=data,
        layout=go.Layout(
            title="Recall surplus for comparison",
            yaxis=dict(title="Recall surplus"),
            xaxis=dict(title="Positives share",range=[0,0.4]),
            legend=dict(
                x=0.8,
                y=0,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
            bgcolor=None


        )))
    iplot(figure)


# In[7]:


def plot_recall_surplus_regu_base(df_pr_train,df_pr_test):
    trace0=go.Scatter(
            y=df_pr_train.recall-df_pr_train.positives_share,
            x=df_pr_train.positives_share,
            mode='lines',
            marker=dict(
            color="red",
            size=5,
            opacity=0.5
            ),
            name="Base model"
        )

    trace1=go.Scatter(
            y=df_pr_test.recall-df_pr_test.positives_share,
            x=df_pr_test.positives_share,
            mode='lines',
            marker=dict(
            color="blue",
            size=5,
            opacity=0.5
            ),
            name="Regularized model"
        )






    data=[trace0,trace1]
    figure=go.Figure(
        data=data,
        layout=go.Layout(
            title="Recall surplus comparison",
            yaxis=dict(title="Recall surplus"),
            xaxis=dict(title="Positives share",range=[0,0.4]),
            legend=dict(
                x=0.8,
                y=0,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
            bgcolor=None


        )))
    iplot(figure)


# In[8]:


def model_comparison_outputs(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    df_pred_train=prediction_proba_df(X_train,y_train,model)
    df_pred_test=prediction_proba_df(X_test,y_test,model)
    
    df_train_pr=precision_recall_curve(df_pred_train)
    df_test_pr=precision_recall_curve(df_pred_test)
    
    return df_pred_train,df_pred_test, df_train_pr, df_test_pr
    


# In[9]:


def model_comparison_outputs_IF(X_train, y_train, X_test, y_test, model):
    model.fit(X_train)
    df_pred_train=prediction_IF_df(X_train,y_train,model)
    df_pred_test=prediction_IF_df(X_test,y_test,model)
    
    df_train_pr=precision_recall_curve(df_pred_train)
    df_test_pr=precision_recall_curve(df_pred_test)
    
    return df_pred_train,df_pred_test, df_train_pr, df_test_pr


# In[10]:


def model_comparison_outputs_Autoencoder(X_train, y_train, X_test, y_test, model):
    
    df_pred_train,bins,labels=prediction_Autoencoder_df(X_train,y_train,model,False,False)
    df_pred_test=prediction_Autoencoder_df(X_test,y_test,model,bins,labels)
    
    df_train_pr=precision_recall_curve(df_pred_train)
    df_test_pr=precision_recall_curve(df_pred_test)
    
    return df_pred_train,df_pred_test, df_train_pr, df_test_pr


# In[11]:


from plotly.subplots import make_subplots
def plot_tp_vs_fp(df_pr):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
            y=df_pr.recall-df_pr.positives_share,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="red",
            size=5,
            opacity=0.5
            ),
            name="Recall Surplus"
        ),secondary_y=False)

    fig.add_trace(go.Scatter(
            y=df_pr.false_accusation_ratio,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="blue",
            size=5,
            opacity=0.5
            ),
            name="False Accusation ratio"
        ),secondary_y=True)
    
    fig.update_xaxes(title_text="xaxis title")
    fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)




    fig.show()


# In[12]:


from plotly.subplots import make_subplots
def plot_surplus(df_pr):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
            y=df_pr.recall-df_pr.positives_share,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="red",
            size=5,
            opacity=0.5
            ),
            name="Recall Surplus"
        ),secondary_y=False)

    fig.add_trace(go.Scatter(
            y=df_pr.recall,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="blue",
            size=5,
            opacity=0.5
            ),
            name="Recall"
        ),secondary_y=True)
    
    fig.update_xaxes(title_text="xaxis title")
    fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)




    fig.show()


# In[13]:


from plotly.subplots import make_subplots
def plot_search_multiplier(df_pr):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
            y=df_pr.search_multiplier,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="red",
            size=5,
            opacity=0.5
            ),
            name="Search Multiplier"
        ),secondary_y=False)

    fig.add_trace(go.Scatter(
            y=df_pr.recall,
            x=df_pr.positives_share,
            mode='lines',
            marker=dict(
            color="blue",
            size=5,
            opacity=0.5
            ),
            name="Recall"
        ),secondary_y=True)
    
    fig.update_xaxes(title_text="xaxis title")
    fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)




    fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




