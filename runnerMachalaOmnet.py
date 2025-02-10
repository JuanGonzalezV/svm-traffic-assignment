# https://sumo.dlr.de/pydoc/traci.html
# https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#defining_new_tls-programs
# people: https://sumo.dlr.de/docs/Specification/Persons.html
# https://github.com/sommer/veins/blob/veins-5.0/src/veins/modules/application/traci/TraCIDemo11p.cc

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26



        # PHASES:
        # tltls_1_2_3_6_phases Phase(duration=39.0, state='GGGggrrrrGGGggrrrr', minDur=39.0, maxDur=39.0, next=())
        # tltls_1_2_3_6_phases Phase(duration=6.0, state='yyyyyrrrryyyyyrrrr', minDur=6.0, maxDur=6.0, next=())
        # tltls_1_2_3_6_phases Phase(duration=39.0, state='rrrrrGGggrrrrrGGgg', minDur=39.0, maxDur=39.0, next=())
        # tltls_1_2_3_6_phases Phase(duration=6.0, state='rrrrryyyyrrrrryyyy', minDur=6.0, maxDur=6.0, next=())

        # tls4_5_phases Phase(duration=79.0, state='GGGG', minDur=79.0, maxDur=79.0, next=())
        # tls4_5_phases Phase(duration=6.0, state='yyyy', minDur=6.0, maxDur=6.0, next=())
        # tls4_5_phases Phase(duration=5.0, state='rrrr', minDur=5.0, maxDur=5.0, next=())


#OS libraries
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random


# we need to import python modules from the $SUMO_HOME/tools directory
# that already done by the bash script

# SUMO libraries
from sumolib import checkBinary  # noqa
import traci  as tr # noqa
from traci import trafficlight as lights


# ML libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from joblib import dump, load

#function to acquari information
def get_tls_color(tls_1_2_3_6,tls4_5):
#### EDGES -> AVENUES
    #tls_1_2_3_6_Logic = lights.getAllProgramLogics(tls_1_2_3_6)
    #tls_1_2_3_6_phases = tls_1_2_3_6_Logic[0].phases

    #tls4_5_Logic = lights.getAllProgramLogics(tls4_5)
    #tls4_5_phases = tls4_5_Logic[0].phases

    # for phase in tls4_5_phases:
    #     print("tls4_5_phases",phase)

    # print("\n")

    # for phase in tls_1_2_3_6_phases:    
    #     print("tltls_1_2_3_6_phases",phase)



    #print("tls_1_2_3_6",tls_1_2_3_6,"state:",lights.getRedYellowGreenState(tls_1_2_3_6))
    #print("tls4_5",tls4_5,"state:",lights.getRedYellowGreenState(tls4_5))


    #if phase == YELLOW:
###
    if (lights.getRedYellowGreenState(tls4_5) == 'yyyy') & (lights.getRedYellowGreenState(tls_1_2_3_6)=='rrrrryyyyrrrrryyyy'):
        #x=input("we are in yellow")
        return(True)
    else:
        return(False)




def values_in_edge(avenue):
    
    #LOCAL VARIABLES
    vehicles=0.0; emissions=0.0; people=0.0

    for edgeID in avenue: 
        vehicles += tr.edge.getLastStepVehicleNumber(edgeID)
        emissions += tr.edge.getCO2Emission(edgeID) + tr.edge.getCOEmission(edgeID) + tr.edge.getHCEmission(edgeID) + tr.edge.getNOxEmission(edgeID) + tr.edge.getNoiseEmission(edgeID) + tr.edge.getPMxEmission(edgeID)
        

    people = persons(avenue)

    #print("vehicles\n",vehicles)
    #print("people\n",people)

    #x=input("continue")

    return(vehicles,emissions,people)

    

def persons(avenue):
    vehicleIDs=tr.vehicle.getIDList()
    cont_people=0.0
    for edgeID in avenue:
        for vID in vehicleIDs:
            if  tr.vehicle.getRoadID(vID) == edgeID: # si le vehiculo esta en la calle q m interesa
                cont_people+= tr.vehicle.getPersonCapacity(vID)
    #print("cont_people",cont_people)
    #x=input("pause cont people")
    
    return(cont_people)


def make_index(vehicles,emissions,people,total_vehicles,total_emission,total_people):

    Amount_vehicle_ratio=0.0
    Vehicle_capacity_ratio=0.0
    Emissions_ratio=0.0
    
    #rank amount vehicle, veihcle capacity, emissions 
    rankAVI=0.0; rankVCI=0.0; rankEI=0.0
    Final_index=0.0

    Amount_Vehicles_Ponderation = 0.2
    Capacity_Ponderation = 0.1
    Emissions_Ponderation = 0.05


    #results for avenue
    Amount_vehicle_ratio = vehicles/total_vehicles
    Vehicle_capacity_ratio = people/total_people
    Emissions_ratio = emissions/total_emission

    #rank variables! and asign weight
    rankAVI = Amount_vehicle_ratio*Amount_Vehicles_Ponderation
    rankVCI = Vehicle_capacity_ratio*Capacity_Ponderation
    rankEI  = (1.0/Emissions_ratio)*Emissions_Ponderation
    Final_index=sum([rankAVI,rankVCI,rankEI])   

    return(Final_index)     
  


def select_max(vc,machala):
    print("Index VC=",vc,". Index Machala=",machala)
    max_index=max(vc,machala)
                
    Priority=[]
    if max_index == vc:
        Priority=('Priority Vaca de Castro',max_index)
    else:
        Priority=('Priority Machala',max_index)
 
    return(Priority)    



def make_labels(vehiclesVC,emissionsVC,peopleVC,vehiclesM,emissionM,peopleM,VacaDeCastro_Index,Machala_Index,X,y):
    max_index=max(VacaDeCastro_Index,Machala_Index)
                
    Priority=[]
    if max_index == VacaDeCastro_Index:
        Priority=('Vaca de Castro',max_index)
        X.append([vehiclesVC,emissionsVC,peopleVC,vehiclesM,emissionM,peopleM])
        y.append(1) #1 means VC has priority

    else:
        Priority=('Machala',max_index)
        X.append([vehiclesVC,emissionsVC,peopleVC,vehiclesM,emissionM,peopleM])
        y.append(0) #0 means Machala has priority
    
    
    print(Priority)
    return(X,y)

def norml2(Dataset):

    #cols names
    names = ["Vehicles Vaca de Castro","Emissions Vaca de Castro","People Vaca de Castro","Vehicles Machala","emission Machala","people Machala"]
    X = pd.DataFrame(Dataset, columns=names) 
    #print("raw X\n",X)  
    
    # Create the Scaler object
    scaled_Dataset = preprocessing.normalize(Dataset, norm='l2')

    scaled_Dataset = pd.DataFrame(scaled_Dataset, columns=names) 
    #print("Scaled Dataset norm:\n",scaled_Dataset) 
    

    
    return(scaled_Dataset)

def save_Data(X,Y,name):
    #use this function to save the datasets

    X.to_csv('trainedSVM/data'+name+'.csv')
    Y = pd.DataFrame(Y, columns = ['label'])
    Y.to_csv('trainedSVM/labels.csv')

def standarize(Dataset):
    # z= (x - u) / s
    # where u=mean, s= stand deviatoin

    names = ["Vehicles Vaca de Castro","Emissions Vaca de Castro","People Vaca de Castro","Vehicles Machala","emission Machala","people Machala"]
    X = pd.DataFrame(Dataset, columns=names)

    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    
    # Fit your data on the scaler object
    scaled_Dataset = scaler.fit_transform(Dataset)
    scaled_Dataset = pd.DataFrame(scaled_Dataset, columns=names)    

    return(scaled_Dataset)

def svm_classification(X,labels):
    
    np.savetxt('trainedSVM/raw_X.csv', X, delimiter=',', fmt='%d')
    standaried_X= standarize(X)

    #save_Data(standaried_X,labels,'standaried_X')

    print("standaried_X\n",standaried_X)

    title=['StandarizedData','NormalizedData']
    

    X_train, X_test, y_train, y_test = train_test_split(standaried_X, labels, test_size = 0.30, random_state=3)

    for k in ('linear', 'poly','rbf'):
        for g in (2,3):
            print("\n\n\t  Training SVM with standaried_X, kernel",k,"and gamma=",g," n")
            #make_svm
            clf = svm.SVC(kernel=k, C = 0.1, gamma=g, degree=3, max_iter=-1,tol=0.001) 

            # Train classifier 
            clf.fit(X_train, y_train)
                
            # Make predictions on unseen test data
            y_pred = clf.predict(X_test)
            print("y_test.shape",y_test.shape)
            print("y_test",y_test)
            print("y_pred.shape",y_pred.shape)
            print("y_pred",y_pred)
                
            accu=np.round(clf.score(X_test,y_test) *100,decimals=3)
            print("Accuracy=",accu,"%")   

            if (accu-75)>1 :
                x=input(">75")

            print("classification report\n",classification_report(y_test, y_pred))

            title='svm_standaried_X_kernel_'+k+'_gamma'+str(g)
            disp = plot_confusion_matrix(clf,X_test, y_test,cmap=plt.cm.Blues) 
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            plt.savefig('trainedSVM/confusionMat/'+title+'.png')
            #plt.show()  
            filename = 'trainedSVM/svm_standaried_X_kernel_'+k+'_gamma'+str(g)+'.joblib'
            dump(clf, filename) 
    
    
    normalized_X = norml2(X)
    print("normalized_X\n",normalized_X)
    #save_Data(normalized_X,labels,'normalized_X')
    
    X_train, X_test, y_train, y_test = train_test_split(normalized_X, labels, test_size = 0.30, random_state=3)

    for k in ('linear', 'poly','rbf'):
        for g in (2,3):
            print("\n\n\t  Training SVM with normalized_X kernel",k,"and gamma=",g," n")
            #make_svm
            clf = svm.SVC(kernel=k, C = 0.1, gamma=g, degree=3, max_iter=-1,tol=0.001) # time 1:30 - acurracy =93.43%

            # Train classifier 
            clf.fit(X_train, y_train)
                
            # Make predictions on unseen test data
            y_pred = clf.predict(X_test)
            print("y_test.shape",y_test.shape)
            print("y_test",y_test)
            print("y_pred.shape",y_pred.shape)
            print("y_pred",y_pred)
                
            accu=np.round(clf.score(X_test,y_test) *100,decimals=3)
            print("Accuracy=",accu,"%")   

            if (accu-75)>1 :
                x=input(">75")

            print("classification report\n",classification_report(y_test, y_pred))

            title='svm_normalized_X_kernel_'+k+'_gamma'+str(g)
            disp = plot_confusion_matrix(clf,X_test, y_test,cmap=plt.cm.Blues) 
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            plt.savefig('trainedSVM/confusionMat/'+title+'.png')
            # plt.show()  
            filename = 'trainedSVM/svm_normalized_X_kernel_'+k+'_gamma'+str(g)+'.joblib'
            dump(clf, filename) 

def run():
    print("\t\t\t\n\n we are running")
    """execute the TraCI control loop"""

    # EDGES -> AVENUES
    vaca_de_castro=['-31954666#6','31954666#6','-31954666#4','31954666#4','-31954666#3','31954666#3']
    machala=['-431774713#0','431774713#0','431774713#1','-431774713#1','431774713#2','-431774713#2','431774713#3','-431774713#3','431774713#4','-431774713#4','431774713#5','-431774713#5']

    # important traffic lights
    tls_1_2_3_6='6245726917'
    tls4_5='267371452'

    # machala
    vehiclesM=0.0; emissionM=0.0; peopleM=0.0

    #Vaca de Castro
    vehiclesVC=0.0; emissionVC=0.0; peopleVC=0.0

    # SET VARIABLE for calculations
    total_vehicles=0.0; total_emission=0.0; total_people=0.0
    VacaDeCastro_Index=0.0;Machala_Index=0.0

    #data maktrix and labels 
    X = []
    y=[]


    
    step =0
    while tr.simulation.getMinExpectedNumber() > 0:
        x = input("\t\t\t\n\n First simulation step, click enter")
        isYellow = get_tls_color(tls_1_2_3_6,tls4_5)
        if(isYellow):
            try:
                vehiclesVC,emissionVC,peopleVC = values_in_edge(vaca_de_castro)  
                vehiclesM,emissionM,peopleM = values_in_edge(machala)  
                
                total_vehicles = sum([vehiclesVC,vehiclesM])
                total_emission = sum([emissionVC,emissionM]) 
                total_people = sum([peopleVC,peopleM])

                VacaDeCastro_Index = make_index(vehiclesVC,emissionVC,peopleVC,total_vehicles,total_emission,total_people) 
                Machala_Index = make_index(vehiclesM,emissionM,peopleM,total_vehicles,total_emission,total_people)                 

                X,y = make_labels(vehiclesVC,emissionVC,peopleVC,vehiclesM,emissionM,peopleM,VacaDeCastro_Index,Machala_Index,X,y)
                #print("X=",X,"y=",y)
                #p=input("\n\n\n pause \n\n\n")


            except:
                print("div by 0")

        
        tr.simulationStep()
        step += 1
    
    X = np.array(X)
    y= np.array(y)
    
    svm_classification(X,y)

    tr.close()
    sys.stdout.flush()



#DO NOT TOUCH THE FOLLOWING:

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    PORT_NUMBER = 54537
    print("\n\n\t\t\tWE ARE IN MAIN\n\n\n")


    tr.init(PORT_NUMBER)
    print("\n\n\t\t\t tr.init(PORT_NUMBER) \n\n\n")
    

    tr.setOrder(2)
    print("\n\n\t\t\t SETTING ORDER \n\n\n")
    run()

    