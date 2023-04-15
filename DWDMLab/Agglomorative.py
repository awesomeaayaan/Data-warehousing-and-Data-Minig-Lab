import pandas as pd
from IPython.display import display_html
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def toy_dataset():
    animal = [['human',1,1,0,0,1,0,'mammals'],['python',0,0,0,0,0,1,'reptiles'],
              ['salmon',0,0,1,0,0,0,'fishes'],['whale',1,1,1,0,0,0,'mammals'],
              ['frog',0,0,1,0,1,1,'amphibians'],
              ['komodo',0,0,0,0,1,0,'reptiles'],['bat',1,1,0,1,1,1,'mammals'],
              ['pigeons',1,0,0,1,1,0,'birds'],['cat',1,1,0,0,1,0,'mammals'],
              ['leopard shark',0,1,1,0,0,0,'fishes'],
              ['turtle',0,0,1,0,1,0,'reptiles'],['penguin',1,0,1,0,1,0,'birds'],
              ['porcupine',1,1,0,0,1,1,'mammals'],['eel',0,0,1,0,0,0,'fishes'],
              ['salamandar',0,0,1,0,1,1,'amphibians']]

    titles = ['Name','Warm_blooded','Give_birth','Aquatic_creature','Aerial_creature','Has_legs','Hibernates','Class']
    data = pd.DataFrame(animal,columns=titles)
    data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')
    print('Do you want to view data?')
    choice = input()
    if choice =='yes':
        display_html(data)
    return data

def ward(names,X,Y):
    Z = hierarchy.linkage(X.to_numpy(),'ward')
    dn = hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')
    fig = plt.figure(figsize=(5, 4))
    plt.show()
    
def centroid(names,X,Y):
    Z = hierarchy.linkage(X.to_numpy(),'centroid')
    dn = hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')
    fig = plt.figure(figsize=(5, 4))
    plt.show()
    
def group_average(names,X,Y):
    Z = hierarchy.linkage(X.to_numpy(),'average')
    dn = hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')
    fig = plt.figure(figsize=(5, 4))
    plt.show()
    
def complete_link(names,X,Y):
    Z = hierarchy.linkage(X.to_numpy(),'complete')
    dn = hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')
    fig = plt.figure(figsize=(5, 4))
    plt.show()
    
def single_link(names,X,Y):
    Z = hierarchy.linkage(X.to_numpy(),'single')
    print('Dendogram of single link Hierachical clustering:')
    dn = hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')
    fig = plt.figure(figsize=(5, 4))
    plt.show()

def main():
    data = toy_dataset()
    names = data['Name']
    Y = data['Class']
    X = data.drop(['Name','Class'],axis=1)
    print('Your data is ready')
    print('Select your option:\ 1.Single_link\ 2.Complete_link\ 3.Group_Average\ 4.Centroid\ 5.ward ')
    choice = int(input())
    if choice == 1:
        single_link(names,X,Y)
    elif choice ==2:
        complete_link(names,X,Y)
    elif choice == 3:
        group_average(names,X,Y)
    elif choice == 4:
        centroid(names,X,Y)
    elif choice == 5:
        ward(names,X,Y)
    else:
        print('Enter correct choice next time:')
        quit()

main()



    