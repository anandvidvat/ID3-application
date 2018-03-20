

import pandas as pd
"""
file_loc - gives location of file, along with format {can be relative or absolute}
"""
def import_data(file_loc):
     #path to file
    df = pd.read_csv(file_loc, sep = ',')
    
    #print number of instances in df
    print(len(df))
    
    #print shape of df 
    print(df.shape)
    
    #print first 10 instances
    print(df.head(10))
    
    return df #returns the dataframe



from sklearn.cross_validation import train_test_split


def split_dataset(df): #df is dataframe

    col = len(df.columns)
    
    X = df.values[:, 1:col]  
    Y = df.values[:,0]  # class attribute is in  first column (will change based on dataset)
    
    x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.25,random_state = 9)
    
    return  x_train,x_test,y_train,y_test
    

from sklearn.tree import DecisionTreeClassifier
def train_model(x_train,y_train):
    
    model  = DecisionTreeClassifier(criterion = 'entropy', random_state = 19, max_depth = None, min_samples_leaf = 5)
    
    model.fit(x_train,y_train)
    
    return model

def predict_model(x_test, model):
    
    y_pred = model.predict(x_test)
    
    return y_pred
    

from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def calculate_sts(y_test,y_pred):
    
    class_report = classification_report(y_test,y_pred)
    
    print("class report")
    print(class_report)
    
    acc = accuracy_score(y_test,y_pred)
    
    
    print("accuracy")
    print(acc)
          
    
    conf_mat = confusion_matrix(y_test, y_pred)
    
          
    print("confusion_matrix")
    print(conf_mat)
    

import pickle
def save_model(model):
    sav_model  = pickle.dumps(model)
    return sav_model
	

def main( path ):
    
    data = import_data(path)
    
    x_train,x_test,y_train,y_test = split_dataset(data)
    print(x_train[0].shape)
    model = train_model(x_train,y_train)
    y_pred = predict_model(x_test, model)
    
    calculate_sts(y_test, y_pred)
    
    saved_model = save_model(model)
    
    return saved_model

if __name__ == '__main__':
    path = './/balancedata.csv'
    saved_model = main(path)
    clf_model = pickle.loads(saved_model) #extracting saved model
    