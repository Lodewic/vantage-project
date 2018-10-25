from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelBinarizer, Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

###################################################################
#Every class defined will have a variable called column_names.
#This will help us to get the column_names of the transformation.
#It can be accessed like the way shown below:
#Example-1:
#freq_bin = FreqBasedCategoricalBinning()
#freq_bin.fit(test_df)
#freq_bin.transform(test_df)
#freq_bin.column_names will 
#contain the column names of the final transformation
#
#Example-2 (using pipeline):
#freq_pipeline = \
#Pipeline([('FreqBasedCategoricalBinning',FreqBasedCategoricalBinning())])
#freq_pipeline.fit(test_df)
#freq_pipeline.transform(test_df)
#freq_pipeline.named_steps['FreqBasedCategoricalBinning'].column_names 
#will contain the list of column names
#used in the transformation
#This will help us to select the desired columns of a data frame
#This will help us to select the desired columns of a data frame
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self,X, y=None):
        self.column_names = self.attribute_names
        return self
    def transform(self,X):
        return X[self.attribute_names]

#We will create another variable: age, 
#based on the year of construction and recorded date
class YearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,method = 'custom'): 
        self.column_names = [] 
        #self.init_radius = init_radius
        #self.increment_radius = increment_radius
        self.method = method
        pass ##Nothing else to do
        
    def fit(self, X, y=None):
        X['construction_year'] = X['construction_year'].astype(float)
        if self.method == 'custom':
            year_recorded = X[X['construction_year'] > 0]\
                            ['date_recorded'].\
                            apply(lambda x: int(x.split("-")[0]))
            year_constructed = X[X['construction_year'] > 0]['construction_year']
            self.median_age = np.median(year_recorded - year_constructed)
            self.column_names = ['age']
            return self
        if self.method == 'median':
               X['construction_year'] = X['construction_year'].astype(float)
               #X['gps_height'] = X['gps_height'].fillna(0)
               self.median = \
                          np.median(list(X[X['construction_year'] != 0]['construction_year']))
               if math.isnan(self.median):
                  self.median = 0
               self.column_names = ['construction_year']
               return self
        if self.method == 'mean':
               X['construction_year'] = X['construction_year'].astype(float)
               #X['gps_height'] = X['gps_height'].fillna(0)
               self.mean = np.mean(list(X[X['construction_year'] != 0]['construction_year']))
               if math.isnan(self.mean):
                  self.mean = 0
               self.column_names = ['construction_year']
               return self

        if self.method == 'ignore':
               self.column_names = ['construction_year']
               return self
          
    def transform(self,X):
        if self.method == 'custom':
            year_recorded = list(X['date_recorded'].apply(lambda x: int(x.split("-")[0])))
            year_constructed = list(X['construction_year'])
            age = []
            for i,j in enumerate(year_constructed):
                if j == 0:
                   age.append(self.median_age)
                else:
                   temp_age = year_recorded[i] - year_constructed[i]
                   if temp_age < 0:
                      temp_age = self.median_age
                   age.append(temp_age)   
            X['age'] = age
            self.column_names = ['age']
            #self.column_names = X.columns
            return X[['age']]
        if self.method == 'median':      
                X['construction_year'] = X['construction_year'].astype(float)
                X['construction_year'] = X['construction_year'].fillna(0)
                construction_year = np.array(list(X['construction_year']))
                construction_year[construction_year == 0] = self.median
                self.column_names = ['construction_year']
                X['construction_year'] = construction_year
                return X[['construction_year']]

        if self.method == 'mean':
                X['construction_year'] = X['construction_year'].astype(float)
                X['construction_year'] = X['construction_year'].fillna(0)
                construction_year = np.array(list(X['construction_year']))
                construction_year[construction_year == 0] = self.mean
                self.column_names = ['construction_year']
                X['construction_year'] = construction_year
                return X[['construction_year']]
        
        if self.method == 'ignore':      
                X['construction_year'] = X['construction_year'].astype(float)
                X['construction_year'] = X['construction_year'].fillna(0)
                self.column_names = ['construction_year']
                return X[['construction_year']]        


#This will help us to convert all text columns to small case and
#also help us to substitute a user defined text in the place of NA values        
class HandleCategoricalNulls(BaseEstimator, TransformerMixin):
    def __init__(self,substitute='unknown',convert_all_to_small=True):
        '''
           cols must be a list
        '''
        self.substitute = substitute
        self.convert_all_to_small = convert_all_to_small
        self.column_names = []
    def get_alpha_numeric(self,x):
        temp_l = list()
        for i in list(x):
            temp = re.sub(r'[^a-zA-Z0-9]',' ', i)
            temp = re.sub(' +'," ",temp)
            temp_l.append(temp)
        return temp_l
    def fit(self,X, y=None):
        HandleCategorical_columns = list(X.columns) 
        self.column_names = list(X.columns) 
        return self
    def transform(self,X):
        X=X.fillna(self.substitute)
        if self.convert_all_to_small:
            X = X.apply(lambda x: x.astype(str).str.lower())
            #Include only alpha-numeric chars
            #Remove any contiguous spaces    
            for i in X.columns:
                X[i] = self.get_alpha_numeric(X[i])
        return X    
        
#helps to contol whether to group text data bsed on common grouping        
#but do not use this, as it runs for a while
class ApplyFuzzyTextProcess( BaseEstimator, TransformerMixin):
    def __init__(self,map_dict,apply=True):
        from fuzzywuzzy import fuzz
        from fuzzywuzzy import process
        self.map_dict = map_dict
        self.apply = apply
        self.column_names = []
    def find_and_replace(self,l):
        temp_l = []
        for j in list(l):
            try:
                temp_l.append(self.map_dict[str(j).lower()])
            except:
                word,score = process.extract(j, self.map_dict.keys(), limit=1)[0]
                if score > 85:
                    temp_l.append(self.map_dict[word])
                else:
                    temp_l.append(j)
                continue    
                
        return temp_l       
    def fit(self,X, y=None):
        self.column_names = list(X.columns)  
        return self
        
    def transform(self,X):
      if self.apply: 
        for i in X.columns:
                X[i] = self.find_and_replace(X[i])
        return X            

##This class will help us to bin the categorical variables with 
##many levels based on the frequency distribution
class FreqBasedCategoricalBinning( BaseEstimator, TransformerMixin):
    def __init__(self,buckets=20,apply=True):
        '''
        buckets - Desired number of classes. 
        you can get less than or equal to the desired buckets,
        if the data is heavily skewed or if we have missing ranges
        '''
        #Calculate the number of bins
        try:
            self.bin_size=np.floor(100.0/(buckets-1))
        except:
            self.bin_size=5.0
        self.freq_dict={}
        self.column_names = []
        self.apply = apply
    def get_freq(self,df):
        #Get the frequency of each level in col 
        for col in df.columns:
            total_count=df.groupby([col]).size().reset_index()
            total_count.columns=[col,col+'_freq_bin']
            total_count.index = list(total_count[col])
            total_count = total_count.drop([col],axis=1)        
            #Save the result to a dictionary
            self.freq_dict[col] = \
            np.ceil(total_count.iloc[:,[0]]/np.max(total_count.iloc[:,0])*100)
    
    def join(self,X,key,value):
        #add a column called 'sorted' to save the order.
        X['sorted'] = np.arange(len(X))
        X.index = list(X[key])
        X.drop([key],axis=1,inplace=True)
        X=X.join(value,how='left')
        #Left join may result in NaN, 
        #if we have unseen levels in the variable
        #We will pad such values with median
        temp_val = np.median(value.iloc[:,[0]])
        X[[value.columns[0]]] = X[[value.columns[0]]].fillna(temp_val)
        
        #Bin the data
        X[[value.columns[0]]] = np.ceil(X[[value.columns[0]]]/self.bin_size)
        
        #Convert the bins to int and later to str
        X[[value.columns[0]]] = X[[value.columns[0]]].astype('int') 
        #X[[value.columns[0]]] = 'class_'+X[[value.columns[0]]].astype('str')
        X[[value.columns[0]]] = X[[value.columns[0]]].astype('str')
        #Reset and drop the first (new) column resulted in reset_index()
        X=X.reset_index()
        X.drop(X.columns[0],axis=1,inplace=True)
        #Restore the order:
        X.sort_values(['sorted'],inplace=True)
        X.drop(['sorted'],axis=1,inplace=True)
        return X
        
    def fit(self,X, y=None):
      if self.apply:
        self.get_freq(X)
        #FreqBasedCategoricalBinning = []
        #Do not set the column names here, to make the logic simple.
        self.column_names = []
        return self      
      else:
        self.column_names = []
        return self      
       
    def transform(self,X,y=None):
      if self.apply:
            X = X.copy()
            for key, value in self.freq_dict.items():
                X = self.join(X,key,value)
            global FreqBasedCategoricalBinning_cols
            #Set the column names here
            self.column_names = list(X.columns)
            return X    
      else:
            self.column_names = list(X.columns)
            return X

##This class will help us to bin the categorical variables with 
##many levels based on the target classes.
class RespBasedCategoricalBinning( BaseEstimator, TransformerMixin):
    def __init__(self,buckets=20,apply=True):
        '''
        buckets - Desired number of classes. 
        you can get less than or equal to the desired buckets,
        if the data is heavily skewed or if we have missing ranges
        '''
        ##Determine the bin size
        try:
           self.bin_size=np.floor(100.0/(buckets-1))
        except:
           self.bin_size= 5.0
        #Declare a dictionary, which will save the details
        #in the fit() function
        self.freq_dict={}
        self.column_names = []
        self.apply = apply
        
    ##This function will be called by fit() function.    
    def get_probs(self,X,y):
        #Get the columns of the X data frame
        X_columns = list(X.columns)
        #Combine X and y into a single data frame
        df = X
        
        #_target_variable is the target column, which is a categorical column
        df['_target_variable'] = list(y)
        
        #For each column in the X data frame, perform the following:
        for col in X_columns:
        
            #Create a data frame with counts, group by the values of the column col
            total_count=df.groupby([col]).size().reset_index()
            
            #Assign the column names to the data frame with group counts
            total_count.columns=[col,'total']
            
            #Make the col as the index.
            #Pandas join is not behaving well if we are joining on a column.
            #So I am assigning the index using the to be joined col values
            total_count.index = list(total_count[col])
            
            #Drop the col as its values are in the index already.
            total_count = total_count.drop([col],axis=1)        
            
            #Now create another data frame, that contains the counts of 
            #values on col, group by target variable
            total_grp_count = df.groupby([col,'_target_variable']).size().reset_index()
            
            total_grp_count.columns=[col,'_target_variable','status_total']
            total_grp_count.index = total_grp_count[col]
            total_grp_count = total_grp_count.drop([col],axis=1)
            
            #Join the two data frames: total_count and total_grp_count
            #We have to use inner join, as we are certain that there will be NO mis-matches
            joined_df = total_count.join(total_grp_count,how='inner')
            joined_df = joined_df.reset_index()
            
            #Make sure that we have proper column names, after reset_index
            columns = list(joined_df.columns)
            columns[0] = col
            joined_df.columns = columns
            #print(joined_df)
            
            #Calculating the proportions
            joined_df['proportion'] = joined_df['status_total']/joined_df['total']
            
            #Pivot the table
            joined_df = pd.pivot_table(joined_df,values='proportion',\
                                       columns='_target_variable',index=col)
            #DO NOT reset the index on the pivot table.
            #As the pivoted table will have col values as the index, 
            #and this will be useful later in transform()
            
            #Fill the NaN values with 0
            joined_df = joined_df.fillna(0)
            for i in joined_df.columns:
                joined_df[i] = joined_df[i]/np.max(joined_df[i])
            #Rename the column names, by appending with col name. 
            #This will make sure that we do not have any duplicate columns            
            joined_df.columns = [col + '-'+str(i).replace(" ", "-") for i in joined_df.columns]
            #Save the col and pivoted table in a dictionary.
            #This dictionary will be used in transform() logic.
            self.freq_dict[col] = joined_df    
        #You have to drop the _target_variable, so that we revert back the changes to X
        df.drop(["_target_variable"],axis=1,inplace=True)
    #Define a function to help with the custom join            
    def join(self,X,key,value):
        '''
           X will be a data frame, on which we have to apply the transform()
           key will be a key in the self.freq_dict
           value will be the pivoted table corresponding to the key
        '''
        ##Create a column called 'sorted'  X. This will help
        ##us to restore the order of X later (or else there might be 
        ##change that the order of X might be disturbed)
        X['sorted'] = np.arange(len(X))
        #Make sure that X is indexed on the key column
        try:
             X.index = list(X[key])
             X.drop([key],axis=1,inplace=True)
             X=X.join(value,how='left')
        except:
             print("EXCEPTION/EROR: The input data frame does not have "+key+" column")
             print("Terminating the program")
         
        #Make sure that you fill the  NaN values with median
        #This is needed, to gracefully add unseen values in the categorical variables
        
        
        for i in value.columns:
            temp_val = np.median(value[i])
            X[i] = X[i].fillna(temp_val)
            X[i] = np.ceil(X[i]*100/self.bin_size)
            X[i] = X[i].astype('int') 
            X[i] = X[i].astype('str')
            
        X=X.reset_index()
        #Drop the first column, which is obtained by reset_index, as it is not needed
        X.drop(X.columns[0],axis=1,inplace=True)
        X.sort_values(['sorted'],inplace=True)
        X.drop(['sorted'],axis=1,inplace=True)
        return X
    #fit() will just build the self.freq_dict         
    #In the following func def, the y parameter is NOT optional
    def fit(self,X, y):
      if self.apply:
        self.get_probs(X,y)
        #To make the logic simple, we will set the column names in transform()
        self.column_names = []
        return self 
      else:
        self.column_names = []
        return self 
                 
    #transform() will iterate over the dictionary keys,
    #and build the transformation.
    #As we are using the self.freq_dict items,
    #even though the input data frame supplied to transform has 
    #extra values, we will not get any errors, and such columns will remain 
    #undisturbed. 
    def transform(self,X,y=None):
      if self.apply:
        X = X.copy()
        for key, value in self.freq_dict.items():
            X = self.join(X,key,value)
        self.column_names = list(X.columns) 
        return X    
      else:
        self.column_names = list(X.columns) 
        return X    

class CatMultiLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,apply=True): 
        self.column_names = [] 
        self.apply = apply
        self.binarizers={}
    
    def check_input_obj(self,X,location):
        ##Check if input object is a pandas df, else raise exception
        try:
            if not isinstance(X,pd.DataFrame):
               raise ValueError
        except:
            print("**EXCEPTION/ERROR**: In "+ location + \
                  " function of "+self.__name__+ ". Input must be a Pandas dataframe")
            exit(10)
        
    
    def fit(self, X, y=None):
      self.column_names = []
      if self.apply:         
        ##Check if input object is a pandas df, else raise exception
        self.check_input_obj(X,'fit()')
        ##Create an empty dict, 
        ##which will be updated with LabelBinarizer for each column        
        self.binarizers={}
        
        for col in X.columns:
            uniq_elements = list(set(X[col]))
            #print(uniq_elements)
            if len(uniq_elements) == 2:
               ##Add a dummy class
               #We have to name this class in a 
               #weird fashion,so that no data has this class
               uniq_elements.append('d#u/m*m-y+class_991-+xya')
            lb = LabelBinarizer()
            self.binarizers[col] = lb.fit(uniq_elements)
            #print(X)
            #self.column_names.append([str(col) + "_" + str(j) \ 
            #for j in list(lb.classes_) if j != 'd#u/m*m-y+class_991-+xya'])
            self.column_names = self.column_names + \
                 [str(col) + "_" + str(j) \
                   for j in list(lb.classes_) \
                     if j != 'd#u/m*m-y+class_991-+xya']
        #print("in transform")
        #print("len of self.binarizers",len(self.binarizers))
        #print("len of self.column_names",len(self.column_names))
        return self
      else:
         return self
    
    def transform(self, X, y=None):
         #print("in transform")
         #print("len of self.binarizers",len(self.binarizers))
         #print(self.apply)
         if self.apply:
            self.check_input_obj(X,'transform()')
            #X_transform = np.empty()
            temp_transformed_data = []
            transformed_column_names = []
            for key, value in self.binarizers.items():
                #print("key=",key)
                #print("value",value)
                #print("len of temp_transformed_data",len(temp_transformed_data))
                try:
                   temp_transformed_data.append(value.transform(X[key]))
                   transformed_column_names = \
                   transformed_column_names + \
                   [str(key) + "_" + str(j) \
                    for j in list(value.classes_)]
                except:
                   continue                
            return pd.DataFrame(np.concatenate(temp_transformed_data, axis=1),\
                                columns=transformed_column_names)[self.column_names]
            #transformed_column_names = self.column_names + [str(col) + "_" + \
            #str(j) for j in list(lb.classes_) if j != 'd#u/m*m-y+class_991-+xya']
            #transformed_X.columns = self.columns        
         else:
            self.column_names = list(X.columns)
            return X  

class FunderInstTransformer( BaseEstimator, TransformerMixin):
    def __init__(self,initial_chars=3,groups=15,apply=True):
        self.initial_chars = initial_chars
        self.groups = groups
        self.apply = apply
        self.group_dict = dict()
    def fetch_first_n_chars(self,l):
        temp_l = [str(j).lower()[0:self.initial_chars] for j in list(l)]
        return pd.Series(temp_l)
                
    def fit(self,X, y=None):   
      if self.apply:
        self.column_names = ['funder','installer'] 
        self.group_dict = dict()
        for i in X.columns:
                temp_series = self.fetch_first_n_chars(X[i]).value_counts()
                temp_series.sort_values(ascending=False,inplace=True)
                top_groups = list(temp_series[0:self.groups].index)
                self.group_dict[i] = top_groups
                 
        return self
      else:
        return self
        
    def transform(self,X):
      X = X.copy()
      if self.apply:
        for i in X.columns:
                temp_series = self.fetch_first_n_chars(X[i])
                temp_l = []
                for j in temp_series.values:
                    try:
                        #print(self.group_dict[i])
                        if j in self.group_dict[i]:
                            #print(j)
                            temp_l.append(j)  
                        else: 
                            temp_l.append('other')  
                    except:
                        continue
                X[i] = temp_l
        #X['funder_installer_same'] = X['funder']==X['installer']
        #self.column_names = ['funder','installer','funder_installer_same']
        #return X[['funder','installer','funder_installer_same']]         
        self.column_names = ['funder','installer']
        return X[['funder','installer']]         
      else:
        #self.column_names = ['funder','installer','funder_installer_same']
        #return X[['funder','installer','funder_installer_same']]
        self.column_names = ['funder','installer']
        return X[['funder','installer']]         
         
class Numpy2DFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,columns): 
        self.column_names = columns
        
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):

        return pd.DataFrame(X,columns=self.column_names)
    ##############
    ##IMPORTANT###
    ##############
    #https://stackoverflow.com/questions/41837261/data-not-persistent-in-scikit-learn-transformers
    #get_params() is very important to get the data persistent between
    #the CV evaluation. If NOT defined, then the init params are NOT set and results in 
    #CV (GridSearch) failure  
    def get_params(self, deep=False):
        return {'columns': self.column_names}

#Scale numeric data        
class ScaleData(BaseEstimator, TransformerMixin): 
     def __init__(self,std_scaler=True,apply=True): 
         self.std_scaler = std_scaler 
         self.apply = apply
 
     def fit(self,X, y=None): 
       if self.apply: 
         if self.std_scaler == True: 
            self.scaler = StandardScaler() 
         else: 
            self.scaler = MinMaxScaler() 
         return self.scaler.fit(X) 
       else:
         return X        
 
     def transform(self,X): 
       if self.apply:
         return self.scaler.transform(X) 
       else:
         return X

##Sekhar: This can be written as a generic class...
##but leave it as it is as of now.
##I will look at this later...
class LatitudeLongitudeProcess( BaseEstimator, TransformerMixin):
    def __init__(self,strategy='median'):
        '''
           type = 'median' is the default.
           other values of type can be 'custom' 
        ''' 
        self.strategy = strategy
        self.median_longitude = 0
        self.custom_longitude = 0
        self.median_latitude = 0
        self.custom_latitude = 0
        self.avg_lat_ward_dict = {}
        self.avg_long_ward_dict = {}
        self.avg_lat_lga_dict = {}
        self.avg_long_lga_dict = {}
        self.avg_lat_region_dict = {}
        self.avg_long_region_dict = {}
        self.avg_lat_country_dict = {}
        self.avg_long_country_dict = {}
        self.column_names = [] 
        
    def get_level_means(self,X):
          if 'ward' in X.columns:   
                #Get average of lats and longs at the ward level
                #First delete rows that have unknown ward values:
                df = X[~((X['ward'].isnull()) | (X['ward'] == 'unknown'))]
                avg_lat_long_by_ward_df = df[df['longitude'] != 0]. \
                groupby(['ward'])['latitude','longitude'].mean().reset_index()
                if len(avg_lat_long_by_ward_df) > 0:
                    avg_lat_long_by_ward_df.columns=['ward','avg_latitude','avg_longitude']
                    self.avg_lat_ward_dict = dict(zip(list(avg_lat_long_by_ward_df['ward']),\
                                                      list(avg_lat_long_by_ward_df['avg_latitude'])))
                    self.avg_long_ward_dict = dict(zip(list(avg_lat_long_by_ward_df['ward']),\
                                                       list(avg_lat_long_by_ward_df['avg_longitude'])))
          if 'lga' in X.columns:        
                #Get average of lats and longs at the lga level
                #First delete rows that have unknown region values:
                df = X[~((X['lga'].isnull()) | (X['lga'] == 'unknown'))]
                avg_lat_long_by_lga_df = df[df['longitude'] != 0]. \
                groupby(['lga'])['latitude','longitude'].mean().reset_index()
                if len(avg_lat_long_by_lga_df) > 0:
                    avg_lat_long_by_lga_df.columns=['lga','avg_latitude','avg_longitude']
                    self.avg_lat_lga_dict = dict(zip(list(avg_lat_long_by_lga_df['lga']),
                                                     list(avg_lat_long_by_ward_df['avg_latitude'])))
                    self.avg_long_lga_dict = dict(zip(list(avg_lat_long_by_lga_df['lga']),
                                                      list(avg_lat_long_by_ward_df['avg_longitude'])))
                
          if 'region' in X.columns:                
                #Get average of lats and longs at the region level
                #First delete rows that have unknown region values:
                df = X[~((X['region'].isnull()) | (X['region'] == 'unknown'))]
                avg_lat_long_by_region_df = df[df['longitude'] != 0]. \
                groupby(['region'])['latitude','longitude'].mean().reset_index()
                if len(avg_lat_long_by_region_df) > 0:
                    avg_lat_long_by_region_df.columns=['region','avg_latitude','avg_longitude']
                    self.avg_lat_region_dict = dict(zip(list(avg_lat_long_by_region_df['region']),\
                                   list(avg_lat_long_by_region_df['avg_latitude'])))
                    self.avg_long_region_dict = dict(zip(list(avg_lat_long_by_region_df['region']),\
                                    list(avg_lat_long_by_region_df['avg_longitude'])))            
          
          #Get average of lats and longs at the country level
          avg_long = np.mean(X[X['longitude'] != 0]['longitude'])
          avg_lat = np.mean(X[X['latitude'] != 0]['latitude'])
          self.avg_lat_country_dict['country'] = avg_lat
          self.avg_long_country_dict['country'] = avg_long
        
    def fit(self,X, y=None):
        self.column_names = ['latitude','longitude']
        X['latitude'] = X['latitude'].astype(float)
        X['longitude'] = X['longitude'].astype(float)
        if self.strategy == 'custom':
           self.get_level_means(X)
           
        elif self.strategy == 'mean':
           #Impute using mean
           self.mean_longitude = np.mean(X[X['longitude'] != 0]['longitude'])
           self.mean_latitude = np.mean(X[X['latitude'] != 0]['latitude'])           
           #X.longitude = [i for i in X.longitude if np.abs(i) <= 0 self.mean_longitude else i]
           #X.latitude  = [i for i in X.latitude if np.abs(i) <= 0 self.mean_latitude else i]
        elif self.strategy == 'median':
           #Impute using median
           self.median_longitude = np.median(X[X['longitude'] != 0]['longitude'])
           self.median_latitude = np.median(X[X['latitude'] != 0]['latitude'])           
           #X.longitude = [i for i in X.longitude if np.abs(i) <= 0 self.median_longitude else i]
           #X.latitude  = [i for i in X.latitude if np.abs(i) <= 0 self.median_latitude else i]
        else:
            print("Invalid strategy supplied for LatitudeLongitudeProcess.")
            print("Valid values are 'mean', 'median' or 'custom'. Terminating the program")
            exit(10)
        return self
    def make_up_lat_long(self,X):
          #Handle the situation gracefully, if the incoming data does not have any required columns
          try:     
              latitude_list = list(X['latitude'].fillna(0))
          except:
                latitude_list = list(np.zeros(len_X))
                #continue
          try:          
              longitude_list = list(X['longitude'].fillna(0))
          except:
                longitude_list = list(np.zeros(len_X))
                #continue
          return latitude_list, longitude_list   
        
    def custom_transform(self, X):     
          len_X = len(X)
          
          #Declare lists to hold the transformed lat and long
          latitude_transformed = []
          longitude_transformed = []
          
          #Handle the situation gracefully, if the incoming data does not have any required columns
          latitude_list, longitude_list =  self.make_up_lat_long(X)
          
          try:  
              ward_list = list(X['ward'].fillna('unknown'))
          except:
              ward_list = ['unknown'] * len_X
              #continue
              
          try:    
              lga_list = list(X['lga'].fillna('unknown'))
          except:
              lga_list = ['unknown'] * len_X
              #continue
              
          try:    
              region_list = list(X['region'].fillna('unknown'))
          except:
              region_list = ['unknown'] * len_X
              #continue
              
          for (i, j, k, l, m) in zip(latitude_list,longitude_list, \
                                    ward_list,lga_list,region_list):
                if np.round(i) == 0 or np.round(j) == 0:
                    try:
                        latitude_transformed.append(self.avg_lat_ward_dict[k])
                        longitude_transformed.append(self.avg_long_ward_dict[k])
                    except:
                        try:
                            latitude_transformed.append(self.avg_lat_lga_dict[l])
                            longitude_transformed.append(self.avg_long_lga_dict[l])
                            continue
                        except:
                            try:
                                latitude_transformed.append(avg_lat_region_dict[m])
                                longitude_transformed.append(avg_long_region_dict[m])
                                continue
                            except:   
                                latitude_transformed.append(self.avg_lat_country_dict['country'])
                                longitude_transformed.append(self.avg_long_country_dict['country'])
                                continue
                else:
                    latitude_transformed.append(i)
                    longitude_transformed.append(j)     
          X['latitude'] = latitude_transformed
          X['longitude'] = longitude_transformed          
          return X
    def transform(self,X):
      X['latitude'] = X['latitude'].astype(float)
      X['longitude'] = X['longitude'].astype(float)
      self.column_names = ['latitude','longitude'] 
      if self.strategy == 'custom':
         X = self.custom_transform(X)
         return X[['latitude','longitude']]
      elif self.strategy == 'mean':
         latitude_list, longitude_list =  self.make_up_lat_long(X)
         latitude_list = np.array(latitude_list)
         longitude_list = np.array(longitude_list)
         latitude_list[latitude_list == 0] = self.mean_latitude
         longitude_list[longitude_list == 0] = self.mean_longitude
         X['latitude'] = latitude_list
         X['longitude'] = longitude_list
         return X[['latitude','longitude']]
      elif self.strategy == 'median':
         latitude_list, longitude_list =  self.make_up_lat_long(X)
         latitude_list = np.array(latitude_list)
         longitude_list = np.array(longitude_list)
         latitude_list[latitude_list == 0] = self.median_latitude
         longitude_list[longitude_list == 0] = self.median_longitude
         X['latitude'] = latitude_list
         X['longitude'] = longitude_list
         return X[['latitude','longitude']]
         #self.column_names = list(X.columns)
         #return X
      else:
         print("EXCEPTION: The supplied strategy",self.strategy," is incorrect")
         exit(10)
         
#Here will be our strategy to handle amount_tsh:
#    1. Get the median values of 
#       amount_tsh group by 'source_class', 'basin', 
#       'waterpoint_type_group'
#    2. If amount_tsh is 0, fill the value using the 
#       median computed by grouping 'source_class', 
#       'basin', 'waterpoint_type_group' 
#    3. If there is no combination of 'source_class', 
#       'basin', 'waterpoint_type_group' found, 
#        fill using 'waterpoint_type_group'
#    4. Else, let it stay 0
class AmountTSHTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,method='custom'): 
        self.column_names = [] 
        self.method = method
    
    def check_input_obj(self,X,location):
        ##Check if input object is a pandas df, else raise exception
        try:
            if not isinstance(X,pd.DataFrame):
               raise ValueError
        except:
            print("**EXCEPTION/ERROR**: In "+ location + \
                  " function of "+self.__name__+ \
                  ". Input must be a Pandas dataframe")
            exit(10)
        
    def fit(self, X, y=None):
      self.column_names = ['amount_tsh']
      if self.method == 'custom':
            X['amount_tsh'] = X['amount_tsh'].astype(float)
            self.check_input_obj(X,"fit()") 
            #Make sure that you have all the required columns:
            if len(set(X.columns) - set(['amount_tsh','source_class', \
                                         'basin', 'waterpoint_type_group'])) == 0:
                
                #Get the required dictionaries...
                amount_tsh_df = X[X['amount_tsh'] != 0].groupby(['source_class', \
                                                                 'basin', \
                                                                 'waterpoint_type_group'])\
                                                                  ['amount_tsh'].median()
                self.amount_tsh_dict_all_level = dict(amount_tsh_df)
                amount_tsh_df = X[X['amount_tsh'] != 0].\
                                groupby(['waterpoint_type_group'])\
                                ['amount_tsh'].median()
                self.amount_tsh_dict_wp = dict(amount_tsh_df)
            else:
                raise ValueError("Check the supplied columns. Must supply 'source_class', \
                                 'basin', 'waterpoint_type_group', 'amount_tsh' only")
                exit(10)
            self.column_names = ['amount_tsh']
            return self
      if self.method == 'median':
           
          X['amount_tsh'] = X['amount_tsh'].astype(float)
          self.median = np.median(list(X[X['amount_tsh'] != 0]['amount_tsh']))
          if math.isnan(self.median):
             self.median = 0
          self.column_names = ['amount_tsh']
          return self
      if self.method == 'mean':
          X['amount_tsh'] = X['amount_tsh'].astype(float)
          self.mean = np.mean(list(X[X['amount_tsh'] > 0]['amount_tsh']))
          if math.isnan(self.mean):
             self.mean = 0
          self.column_names = ['amount_tsh']
          return self
          
      if self.method == 'ignore':
          self.column_names = ['amount_tsh']
          return self
          
          
          

    def transform(self,X):
      if self.method == 'custom':
            X['amount_tsh'] = X['amount_tsh'].astype(float)
            self.check_input_obj(X,"transform()")
            transformed_amount_tsh = []
            for i, j, k, l in list(zip(X['amount_tsh'].\
                                       fillna(0),X['source_class'], \
                                       X['basin'], X['waterpoint_type_group'])):
                if i == 0:
                    try:
                        transformed_amount_tsh.append(self.amount_tsh_dict_all_level[(j,k,l)])

                    except:
                        try:
                            transformed_amount_tsh.append(self.amount_tsh_dict_wp[l])
                        except:
                                transformed_amount_tsh.append(i)
                                continue
                else:
                        transformed_amount_tsh.append(i)
            X['amount_tsh'] = transformed_amount_tsh
            return X[['amount_tsh']]
      if self.method == 'median':
          X['amount_tsh'] = X['amount_tsh'].astype(float)
          X['amount_tsh'] = X['amount_tsh'].fillna(0)
          amount_tsh = np.array(list(X['amount_tsh']))
          amount_tsh[amount_tsh == 0] = self.median
          X['amount_tsh']  = amount_tsh
          return X[['amount_tsh']]
      if self.method == 'mean':
          X['amount_tsh'] = X['amount_tsh'].astype(float)
          X['amount_tsh'] = X['amount_tsh'].fillna(0)
          amount_tsh = np.array(list(X['amount_tsh']))
          amount_tsh[amount_tsh == 0] = self.mean
          X['amount_tsh']  = amount_tsh
          return X[['amount_tsh']]
          
      if self.method == 'ignore':
          X['amount_tsh'] = X['amount_tsh'].astype(float)
          X['amount_tsh'] = X['amount_tsh'].fillna(0)          
          return X[['amount_tsh']]

class ChooseCatPipelineType(BaseEstimator, TransformerMixin): 
     def __init__(self,freq_pipeline,resp_pipeline,method='both'): 
         self.freq_pipeline = freq_pipeline
         self.resp_pipeline = resp_pipeline
         self.method = method
         self.column_names = []
     def fit(self,X, y=None): 
       if self.method == 'resp':
         self.resp_pipeline.fit(X,y) 
       if self.method == 'freq':
         self.freq_pipeline.fit(X,y) 
       if self.method == 'both':
          self.both_pipeline = FeatureUnion(transformer_list = [ \
                                    ("freq_pipeline",self.freq_pipeline), \
                                    ("resp_pipeline",self.resp_pipeline) \
                                                ])  
          self.both_pipeline.fit(X,y)                                                
       return self

     def transform(self,X): 
           if self.method == 'resp':
             self.column_names = self.resp_pipeline.named_steps['CatMultiLabelTransformer'].column_names
             return self.resp_pipeline.transform(X) 
           if self.method == 'freq':
             self.column_names = self.freq_pipeline.named_steps['CatMultiLabelTransformer'].column_names
             return self.freq_pipeline.transform(X) 
           if self.method == 'both':
              self.column_names = self.freq_pipeline.named_steps['CatMultiLabelTransformer'].column_names + \
                                  self.resp_pipeline.named_steps['CatMultiLabelTransformer'].column_names
              return self.both_pipeline.transform(X)      
