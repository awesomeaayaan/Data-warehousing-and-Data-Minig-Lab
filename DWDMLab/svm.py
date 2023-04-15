import pandas as pd
from IPython.display import display_html
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.datasets import load_iris

def build_model(X_train,y_train):
    clf = LinearSVC()
    clf = clf.fit(X_train,y_train)
    return clf

def prediction_using_model(clf,X_test,y_test):
    X_test = X_test.reset_index()
    del X_test['index']
    y_test = y_test.reset_index()
    del y_test['index']
    y_pred = clf.predict(X_test)
    predictions = pd.concat([X_test,pd.Series(y_pred,name='Predicted Class')],axis=1)
    print('Do you want to view the class label prediction for top five tuples of test data?')
    choice = input()
    if choice=='yes':
        display_html(predictions.head())
     #model evaluation
    print('Do you want to view the evaluation of model?')
    choice = input()
    if choice == 'yes':
        model_evaluation(y_pred,y_test)
    else:
        quit()

def model_evaluation(y_pred,y_test):
    print('Confusion Matrix:')
    report = (confusion_matrix(y_test,y_pred))
    cf = pd.DataFrame(report).transpose()
    display_html(cf)
    score = accuracy_score(y_test,y_pred)
    print('SVM accuracy:',score)
    print('Classification report:')
    report = (classification_report(y_test,y_pred,output_dict=True))
    df = pd.DataFrame(report).transpose()
    display_html(df[['precision','recall','f1-score']].head(3))

def main():
    data = pd.read_csv('IRIS.csv') 
    print('Do you want to view top five data tuples of Iris Dataset?')
    choice = input()
    if choice == 'yes':
        display_html(data.head())
    Y = data['species']
    X = data.drop(['species'],axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=1)
    clf = build_model(X_train,y_train)
    prediction_using_model(clf,X_test,y_test)
main()
    