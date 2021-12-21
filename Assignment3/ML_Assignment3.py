#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from matplotlib import pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


# In[2]:


description=pd.read_csv('D:/study/Sem Wise/sem5/ML/Assignments/Assignment3/Dataset Description.csv')


# In[3]:


more=pd.read_csv('D:/study/Sem Wise/sem5/ML/Assignments/Assignment3/more_than_50k.csv')


# In[4]:


population=pd.read_csv('D:/study/Sem Wise/sem5/ML/Assignments/Assignment3/population.csv')


# In[5]:


description


# In[6]:


more.head(10)


# # 1.1

# In[7]:


print(population.isna().sum())


# In[8]:


for i in population:
    vari1=population[i]
    vari1[vari1==' ?']=np.nan


# In[9]:


print(population.isna().sum())


# # 1.2

# In[10]:


percentages=[]
for i in population:
    count=population[i].isna().sum()
    percentages.append((i,100*count/len(population)))
    
    if(percentages[-1][1]>40):
        population=population.drop(i,axis=1)
        
print("Percentages")
print(percentages)
print()

print("Removed columns are: ")
for i in percentages:
    if(i[1]>40):
        print(i[0])


# In[11]:


print(len(population))


# In[12]:


count=0
for i in population:
    count+=1

print(count)


# # 2.1

# In[13]:


#bar graphs for categorical data
for i in population:
    if(type(population[i][0])==type('abc')):
        population.groupby(i).size().plot(kind='bar')
    plt.show()


# In[14]:


#histogram for numberic data types
for i in population:
    
    if(type(population[i][0])!=type('a')):
        hist,bins=np.histogram(population[i])
        bins=np.round(bins)
        plt.bar(bins[:-1], hist)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(i)
        plt.show()


# In[15]:


to_drop=[]

for i in population:
    
    if(type(population[i][0])!=type('a')):
        continue
    
    vals=population[i].value_counts()
    vals=vals.to_dict().values()
    vals=list(vals)
    max_percentage=100*vals[0]/sum(vals)
    
    if(max_percentage>80):
        to_drop.append((i,max_percentage))
        population=population.drop(i,axis=1)
    
print("Deleted columns are: ")
print(to_drop)


# In[16]:


print(len(population))


# # 3.1

# In[17]:


modes={}

for i in population:
    
    mode=population[i].mode()
    modes[i]=mode[0]
    population[i].fillna(mode[0],inplace=True)


# In[18]:


population.isna().sum()


# In[19]:


print(len(population))


# # 3.2

# In[20]:


for i in population:
    if(type(population[i][0])!=type('a')):
        quantiles=[]
        for j in range(1,6):
            quantiles.append(population[i].quantile(0.2*j))
            
        population['binned_'+i]=population[i]
        vari=population['binned_'+i]
        
        names=['very low','low','neutral','high','very high']
        for j in range(len(quantiles)):
            if(j==0):
                vari[population[i]<=quantiles[0]]=names[j]

            else:
                vari[(population[i]>quantiles[j-1]) & (population[i]<=quantiles[j])]=names[j]
        
        population=population.drop(i,axis=1)


# In[21]:


#bar graph for bucketized data
for i in population:
    
    if('binned_'in i):
        population.groupby(i).size().plot(kind='bar')
        plt.show()


# # 3.3

# In[22]:


population.head(10)


# In[23]:


columns=[]
for i in population:
    columns.append(i)

final_df=[]
for i in columns:
    num_features=len(list(set(population[i])))
    vari_df=pd.get_dummies(population[i])
    
    for k in vari_df:
        vari_df.rename(columns = {k:i+" "+k}, inplace = True)
        
    if(num_features==2):
        for j in vari_df:
            vari_df=vari_df.drop(j,axis=1)
            break
            
    final_df.append(vari_df)

population_one_hot = pd.concat(final_df,axis = 1)    
population_one_hot.head(10)


# # PCA

# In[24]:


from sklearn.decomposition import PCA


# In[25]:


variance_sum=[]
dimensions=[]
for i in range(20,51):
    pca = PCA(n_components=i)
    pca.fit(population_one_hot)
    variance_sum.append(np.sum(pca.explained_variance_ratio_))
    dimensions.append(i)
    
    print("Sum of variance is "+str(variance_sum[-1])+" on "+str(i)+" dimensions. ")

    if(100*variance_sum[-1]>90):
        break


# In[26]:


plt.xlabel("Dimenions")
plt.ylabel("Variance")
plt.title("Variance vs Dimensions")
plt.plot(dimensions,variance_sum)
plt.show()


# In[27]:


#taking component with greater 85 percent total variance as later the the total has slowed down
pca_pop = PCA(n_components=29)
population_transformed=pca_pop.fit_transform(population_one_hot)

print("Sum of variance : "+str(np.sum(pca_pop.explained_variance_ratio_)))
print(pca.explained_variance_ratio_)


# In[28]:


population_transformed.shape


# # 4 Clustering

# ## 4.1 

# In[29]:


class KMedians():
    
    def __init__(self,n_clusters=3,n_iters=10,n_dimensions=29):
        self.n_clusters=n_clusters
        self.n_iters=n_iters
        self.n_dimensions=n_dimensions
    
    def fit(self,data,plot=False):
        self.data=data
        self.centers=np.zeros((self.n_clusters,self.n_dimensions))
        self.randoms=random.sample(range(len(data)),self.n_clusters)
        
        for i in range(len(self.randoms)):
            self.centers[i]=data[self.randoms[i]]
            
        self.loss_store=[]
        self.iterations=[]
        
        for iterr in range(self.n_iters):
            self.distance={}
            self.iterations.append(iterr)
            
            #using manhattan distance, dist(v1,v2)=sum over all components |v1-v2|
            for k in range(self.n_clusters):
                self.distance[k]=self.data-self.centers[k]
                self.distance[k]=np.absolute(self.distance[k])
                self.distance[k]=np.sum(self.distance[k],axis=1)
                
            #minimum distance of all points from their cluster centroid is the global minimum dist
            self.min_distance=self.distance[0]
            
            for k in range(self.n_clusters):
                self.min_distance=np.minimum(self.min_distance,self.distance[k])
            
            self.loss_store.append(np.sum(self.min_distance))
            self.clusters={}
            
            for k in range(self.n_clusters):
                self.clusters[k]=(self.distance[k]==self.min_distance)
                self.clusters[k]=self.data[self.clusters[k]]
            
            #re centering the initial points according to median
            
            for k in range(self.n_clusters):
                self.centers[k]=np.median(self.clusters[k],axis=0)
                
        if(plot):
            plt.xlabel("Loss")
            plt.ylabel("Iterations")
            plt.title("Loss vs Iterations")
            plt.plot(self.iterations,self.loss_store)
            plt.show()
        
        loss=self.loss_store[-1]
        return loss,self.clusters


# In[30]:


k_values=[]
loss_values=[]
for i in range(10,25):
    k_values.append(i)
    kmed=KMedians(n_clusters=i,n_iters=10)
    loss,_=kmed.fit(population_transformed)
    loss_values.append(loss/i)
    #avg within cluster distance is avg of sum of all points from their centroid which is the minimum dist


# # 4.2

# In[31]:


plt.xlabel("Different values of k")
plt.ylabel("Avg cluster distance after training 10 epochs")
plt.title("Avg within cluster distance vs K")
plt.plot(k_values,loss_values)


# # 4.3

# In[32]:


kmed=KMedians(n_clusters=17,n_iters=10)
loss,population_clusters=kmed.fit(population_transformed,plot=True)


# In[33]:


print(loss/17)


# # 5

# In[34]:


more=pd.read_csv('D:/study/Sem Wise/sem5/ML/Assignments/Assignment3/more_than_50k.csv')


# In[35]:


for i in more:
    vari1=more[i]
    vari1[vari1==' ?']=np.nan


# In[36]:


percentages=[]
for i in more:
    count=more[i].isna().sum()
    percentages.append((i,100*count/len(more)))
    
    if(percentages[-1][1]>40):
        more=more.drop(i,axis=1)
        
print("Percentages")
print(percentages)
print()

print("Removed columns are: ")
for i in percentages:
    if(i[1]>40):
        print(i[0])


# In[37]:


#histogram for numberic data types
for i in more:
    
    if(type(more[i][0])!=type('a')):
        hist,bins=np.histogram(more[i])
        bins=np.round(bins)
        plt.bar(bins[:-1], hist)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(i)
        plt.show()


# In[38]:


#bar graphs for categorical data
for i in more:
    if(type(more[i][0])==type('abc')):
        more.groupby(i).size().plot(kind='bar')
    plt.show()


# In[39]:


to_drop=[]

for i in more:
    
    if(type(more[i][0])!=type('a')):
        continue
    
    vals=more[i].value_counts()
    vals=vals.to_dict().values()
    vals=list(vals)
    max_percentage=100*vals[0]/sum(vals)
    
    if(max_percentage>80):
        to_drop.append((i,max_percentage))
        more=more.drop(i,axis=1)
    
print("Deleted columns are: ")
print(to_drop)


# In[40]:


for i in more:
    if(i in more):
        more[i].fillna(modes[i],inplace=True)
    else:
        more[i].fillna(more[i].mode()[0],inplace=True)


# In[41]:


for i in more:
    if(type(more[i][0])!=type('a')):
        quantiles=[]
        for j in range(1,6):
            quantiles.append(more[i].quantile(0.2*j))
            
        more['binned_'+i]=more[i]
        vari=more['binned_'+i]
        
        names=['very low','low','neutral','high','very high']
        for j in range(len(quantiles)):
            if(j==0):
                vari[more[i]<=quantiles[0]]=names[j]

            else:
                vari[(more[i]>quantiles[j-1]) & (more[i]<=quantiles[j])]=names[j]
        
        more=more.drop(i,axis=1)


# In[42]:


#bar graph for bucketized data
for i in more:
    
    if('binned_'in i):
        more.groupby(i).size().plot(kind='bar')
        plt.show()


# In[43]:


columns=[]
for i in more:
    columns.append(i)

final_df=[]
for i in columns:
    num_features=len(list(more[i].value_counts().to_dict().keys()))
    vari_df=pd.get_dummies(more[i])
    
    for k in vari_df:
        vari_df.rename(columns = {k:i+" "+k}, inplace = True)
        
    if(num_features==2):
        for j in vari_df:
            vari_df=vari_df.drop(j,axis=1)
            break
        
    final_df.append(vari_df)
    
more_one_hot=final_df[0]
final_df.remove(final_df[0])

for i in final_df:
    more_one_hot=pd.concat([i,more_one_hot],axis=1)
    
more_one_hot.head(10)


# In[44]:


variance_sum=[]
dimensions=[]
for i in range(20,51):
    pca = PCA(n_components=i)
    pca.fit(more_one_hot)
    variance_sum.append(np.sum(pca.explained_variance_ratio_))
    dimensions.append(i)
    
    print("Sum of variance is "+str(variance_sum[-1])+" on "+str(i)+" dimensions. ")

    if(100*variance_sum[-1]>90):
        break


# In[45]:


plt.xlabel("Dimenions")
plt.ylabel("Variance")
plt.title("Variance vs Dimensions")
plt.plot(dimensions,variance_sum)
plt.show()


# In[46]:


#taking same number of dimensions as in population
pca_more = PCA(n_components=29)
more_transformed=pca_more.fit_transform(more_one_hot)

print("Sum of variance : "+str(np.sum(pca_more.explained_variance_ratio_)))
print(pca_more.explained_variance_ratio_)


# In[47]:


more_transformed.shape


# In[48]:


k_values=[]
loss_values=[]
for i in range(10,25):
    k_values.append(i)
    kmed1=KMedians(n_clusters=i,n_iters=10)
    loss,_=kmed1.fit(more_transformed)
    loss_values.append(loss/i)


# In[49]:


plt.xlabel("Different values of k")
plt.ylabel("Avg cluster distance after training 10 epochs")
plt.title("Avg within cluster distance vs K")
plt.plot(k_values,loss_values)


# In[50]:


#taking same value of the cluster param as the graph is smooth and no elbow is detected
kmed_more=KMedians(n_clusters=17,n_iters=10)
loss,more_clusters=kmed_more.fit(more_transformed,plot=True)


# In[51]:


loss/17


# In[52]:


population_clusters


# In[53]:


pop_clust=[]

for i in population_clusters:
    vari_df=pd.DataFrame(population_clusters[i])
    vari_df['label']=i
    pop_clust.append(vari_df)
    
population_all=pd.concat(pop_clust)

population_labels=population_all['label']
population_points=population_all.drop('label',axis=1)


# In[54]:


more_clust=[]

for i in more_clusters:
    vari_df=pd.DataFrame(more_clusters[i])
    vari_df['label']=i
    more_clust.append(vari_df)
    
more_all=pd.concat(more_clust)

more_labels=more_all['label']
more_points=more_all.drop('label',axis=1)


# In[55]:


pca1 = PCA(2) 
population_projected = pca1.fit_transform(population_points)

pca2=PCA(2)
more_projected = pca2.fit_transform(more_points)


# In[56]:


plt.scatter(population_projected[:, 0], population_projected[:, 1],
            c=population_labels, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 18))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# In[57]:


plt.scatter(more_projected[:, 0], more_projected[:, 1],
            c=more_labels, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 18))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# In[58]:


diff=[]
k_vals=[]

for i in range(17):
    k_vals.append(i+1)
    x=100*len(population_clusters[i])/len(population_labels)-100*len(more_clusters[i])/len(more_labels)
    diff.append(x)

plt.plot(k_vals,diff)
plt.title("Percentage of population cluster - percentage of more clusters vs respective cluster number")
plt.show()


# In[76]:


med1=np.median(population_clusters[2],axis=0)
orig=pd.Series(med1)
orig.sort_values(ascending=False,inplace=True)
orig.head()


# In[78]:


feature_select=pd.DataFrame(pca_pop.components_, columns=population_one_hot.columns).T
vari=feature_select[0].sort_values()

print(vari)


# In[79]:


feature_select=pd.DataFrame(pca_pop.components_, columns=population_one_hot.columns).T
vari=feature_select[2].sort_values()

print(vari)


# In[80]:


feature_select=pd.DataFrame(pca_pop.components_, columns=population_one_hot.columns).T
vari=feature_select[3].sort_values()

print(vari)


# In[83]:


med2=np.median(more_clusters[6],axis=0)
orig=pd.Series(med2)
orig.sort_values(ascending=False,inplace=True)
orig.head()


# In[84]:


feature_select=pd.DataFrame(pca_more.components_, columns=more_one_hot.columns).T
vari=feature_select[2].sort_values()

print(vari)


# In[85]:


feature_select=pd.DataFrame(pca_more.components_, columns=more_one_hot.columns).T
vari=feature_select[3].sort_values()

print(vari)


# In[86]:


feature_select=pd.DataFrame(pca_more.components_, columns=more_one_hot.columns).T
vari=feature_select[5].sort_values()

print(vari)

