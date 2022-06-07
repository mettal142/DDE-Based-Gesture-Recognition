import tensorflow as tf
import numpy as np
import copy as cp
import random as rd
import serial
import time
import math
import matplotlib.pyplot as plt
import os
from dataglove import *

Mode =2 #0:TrainData, 1:ReadData, 2:Combine_CNNData, 3:Combine_LSTMData, 4:Show Data
tri=0
hyper =200

MotionIndex = 1
# 0:dummy, 1:a, 2:b, 3:q, 4:e, 5:r, 6:f, 7:k, 8:l, 9:m 10:z
# 11:apple ,12:banana ,13:camera ,14:dog ,15:ear ,16:grape ,17:help ,18:korea ,19:mouse ,20:lemon
MotionDataset={1:'a',2:'b',3:'q',4:'e',5:'r',6:'f',7:'k',8:'l',9:'m',10:'z',
               11:'apple',12:'banana',13:'camera',14:'dog',15:'ear',16:'grape',17:'help',18:'korea',19:'mouse',20:'lemon'}

#MotionDataset={1:'모래시계',2:'알파'} 

USER='_00' #00:나, 01:유리, 02: 혜정, 03 : 혜원, 04 : 태인

Dtype=""
#Dtype="letter"
#Dtype="word"

choice = 35
Routine=50
FileName=''
FolderName='Data_220103'
#FolderName='DemoData'
FilePath='./'
Date='_220420'

#Date='_220420_dynamic_'

st=1
ed=21
if Dtype=="letter":
    st=1
    ed=11
elif Dtype=="word":
    st=11
    ed=21

glove = Forte_CreateDataGloveIO(0)# right:0 left:1
global Time

class GloveControl:
    
    def HapticOn(wave,ampitude):
            Forte_SendHaptic(glove,0,wave,ampitude)
            Forte_SendHaptic(glove,1,wave,ampitude)
            Forte_SendHaptic(glove,2,wave,ampitude)
            Forte_SendHaptic(glove,3,wave,ampitude)
            Forte_SendHaptic(glove,4,wave,ampitude)
            Forte_SendHaptic(glove,5,wave,ampitude)

    def HapticOff():
            Forte_SendHaptic(glove,0,0,0)
            Forte_SendHaptic(glove,1,0,0)
            Forte_SendHaptic(glove,2,0,0)
            Forte_SendHaptic(glove,3,0,0)
            Forte_SendHaptic(glove,4,0,0)
            Forte_SendHaptic(glove,5,0,0)

    def HapticShot(wave,amplitude):
        Forte_SendOneShotHaptic(glove,0,wave,amplitude)
        Forte_SendOneShotHaptic(glove,1,wave,amplitude)
        Forte_SendOneShotHaptic(glove,2,wave,amplitude)
        Forte_SendOneShotHaptic(glove,3,wave,amplitude)
        Forte_SendOneShotHaptic(glove,4,wave,amplitude)
        Forte_SendOneShotHaptic(glove,5,wave,amplitude)

    def PoseTrigger(Hand,FlexSensors):
        Thumb = FlexSensors[0] + FlexSensors[1]
        Index = FlexSensors[2] + FlexSensors[3]
        Middle = FlexSensors[4] + FlexSensors[5]
        Ring = FlexSensors[6] + FlexSensors[7]
        Pinky = FlexSensors[8] + FlexSensors[9]
        #if Thumb >= 100 and Index >= 100 and Middle >= 100  and Pinky >= 100:
        if Middle >= 80:

            GloveControl.HapticOn(126,0.7)
            return 1
        else:
            GloveControl.HapticOff()
            #HapticShot(Hand,1,1)
            return 0

class DataProcess:
    
    def HyperSampling(data,time,sample):
   
        if len(data)<sample:
            while 1:
                temp1 = []
                temp2 = []
                it = len(data) * 2
                dt = time / len(data)
                for i in range(0,it - 4,2):
                    inclination1 = (np.array(data[i + 1]) - np.array(data[i])) / dt
                    inclination2 = (np.array(data[i + 2]) - np.array(data[i + 1])) / dt
                    doubleinc = (inclination2 - inclination1) / (2 * dt)
                    data = np.insert(data,i + 1,(inclination1 + doubleinc * (dt / 2)) * (dt / 2) + data[i],axis=0)
                    if len(data) == sample:
                        break
                if len(data) == sample:
                    break
        elif len(data)>sample:
            data= DataProcess.RandomSelect(data,sample)
        #elif len(data)>sample:
        #    deltaData = cp.copy(data)
        #    temp = DataProcess.Differential(deltaData)
        #    data= DataProcess.Integration(DataProcess.Constraction(temp,sample))
        return data

    def RandomSelect(data,sample):
        temp1 = []#data
        temp2 = []#lable
    
        idx = rd.sample(range(len(data)),sample)
        idx.sort()
    
        for i in idx:
            temp1.append(data[i])
    
        return temp1

    def Expansion(DeltaData,Num):
        retData = cp.copy(DeltaData)
        absData = cp.copy(list(map(abs,cp.copy(DeltaData))))
        while True:
            i= absData.index(max(absData))
            Max=retData[i]
            del absData[i]
            del retData[i]
            absData.insert(i,abs(Max/2))
            absData.insert(i,abs(Max/2))
            retData.insert(i,Max/2)
            retData.insert(i,Max/2)
            if len(retData)>=Num:
                break
        return np.array(retData)

    def Constraction(DeltaData,num=hyper):
        dataTemp = cp.copy(DeltaData)
        absData = cp.copy(list(map(abs,cp.copy(DeltaData))))
        
        while True:

            i= absData.index(min(absData[:-1]))
            dataTemp[i+1]+=dataTemp[i]
            del absData[i]
            del dataTemp[i]
            if len(dataTemp)<=num:
                break
        return dataTemp

    def DynamicSampling(DeltaData,Num):
        Data=cp.copy(DeltaData)
        Idx=0
        if len(Data)<Num:#Expansion
            Data=DataProcess.Expansion(Data,Num)

        elif len(Data)>Num:#Constraction
            Data=DataProcess.Constraction(Data,Num)
        
        return Data

    def delzero(data,f=0.001):
        temp = []
        for i in range(len(data)):
            if abs(data[i]) >= f:
                temp.append(data[i])
        return temp   
    def delzero2(data,f=0.07):
        temp = []
        for i in range(len(data)):
            if abs(data[i]) >= f:
                temp.append(data[i])
        return temp

    def delover(data,f=100):
        temp = []
        for i in range(len(data)):
            if abs(data[i]) <= f:
                temp.append(data[i])
        return temp

    def Integration(data):
        res = []
        temp = 0
        for i in range(len(data)):
            temp+=data[i]
            res.append(temp)
        return np.array(res)
        return np.array(res)

    def Normalize(data):

        return (np.array(data)/(max(data)+abs(min(data))))

    def Differential(D):
        Data = cp.copy(D)
        retData=[]
        for i in range(len(Data)-1):
            retData.append(Data[i+1]-Data[i])
        return retData

class DataIO:

    def DataRead(FileName,x,y,Trigger=False):    
        Data = np.load(FilePath+FileName+'.npy',allow_pickle=True)
        np.random.shuffle(Data)
        DD=[]
        DT=[0,0,0]
        for i in range(len(Data)):
            DT[0]=(Data[i][0][0])
            DT[1]=(Data[i][0][1])
            DT[2]=(Data[i][0][2])

            DD.append(DT)
            DT =[0,0,0]

        train_X=[]
        train_Y=[]
        test_X=[]
        test_Y=[]
        for i in range(len(DD)):
            if Trigger:
                DD[i]=np.array(DD[i]).reshape(x,y,1)

            if i <= len(DD)*0.80:
                train_X.append(DD[i])
                train_Y.append(Data[i][1])

            else:
                #if Data[i][1]!=0:
                    test_X.append(DD[i])
                    test_Y.append(Data[i][1])

        return (train_X,train_Y),(test_X,test_Y)
    
    def DataRead_LSTM(FileName,x,y,Trigger=False):    
        Data = np.load(FilePath+FileName+'.npy',allow_pickle=True)
        np.random.shuffle(Data)
        DD=[]
        DT=[[] for i in range(hyper)]
        train_X=[]
        train_Y=[]
        test_X=[]
        test_Y=[]
        length=len(Data)
        for i in range(length):
            for j in range(hyper):
                DT[j].append(Data[i][0][0][j])
                DT[j].append(Data[i][0][1][j])
                DT[j].append(Data[i][0][2][j])
            if i <= length*0.80:
                train_X.append(DT)
                train_Y.append(Data[i][1])

            else:
                #if Data[i][1]!=0:
                    test_X.append(DT)
                    test_Y.append(Data[i][1])
            DT=[[] for i in range(hyper)]
   
        #for i in range(length):
        #    if Trigger:
        #        DD[i]=np.array(DD[i]).reshape(x,y,1)
        #    if i <= length*0.80:
        #        train_X.append(DD[i])
        #        train_Y.append(Data[i][1])

        #    else:
        #        #if Data[i][1]!=0:
        #            test_X.append(DD[i])
        #            test_Y.append(Data[i][1])
        return (train_X,train_Y),(test_X,test_Y)
              
    def ShowGraph(data,index=111,form='r.'):

        #plt.title('Data')
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.subplot(index)
        plt.plot(data,form) 

    def GetTest(model):
        t=0
        try:
            while True:
                d,deltaT=GetTestData()
                t1=time.time()
                result=model.predict(np.array(d).reshape(1,3,hyper,1))
                t2=time.time()
                print(MotionDataset[np.argmax(result)])
        except(KeyboardInterrupt):
            Forte_DestroyDataGloveIO(glove)
            exit()    
    
    def GenerateMatrix(Name,model,testX,testY):
        table=[[0 for i in range(21)] for j in range(21)]
        count=[0 for i in range(21)]

        ava=[]
        np.array(table)
        t1=0
        t2=0
        for i in range(len(testX)):
            t1=time.time()
            test=model.predict(np.array(testX[i]).reshape(1,3,hyper,1))
            #print(time.time()-t1)

            res=np.argmax(test)
            count[res]+=1
            table[res]+=test

        for i in range(1,21):
            if count[i]!=0:
                ava.append(table[i][0]/count[i])

       
        np.savetxt(Name+Date+"_"+str(hyper)+".csv",np.round(ava,5),delimiter=",")
        print(Name+Date+"_"+str(hyper)+".csv"+" Saved")
        #TrainGraph()
        #plt.show()

  
def GenerateData(Mode,MotionIndex):

    FileName='/Motion'+str(MotionIndex)+USER+'.npy'

    save = []

    Data = [[[],[],[]],0,0]
    InclinationData = [[[],[],[]],0,0]
    hypersave = []
    choicesave = []
    IMU = []
    DeltaIMU = []
    InitializedData = []
    Iterator = 0
    preTrigger = 0
    AmountOfChange = [0,0,0]
    Mat1=[]
    Mat2=[]
    Mat3=[]
    TT=0
    print("start")
    try:
        if Mode == 0:
            try:
                print("Motion Index : ", MotionIndex)
                while True:
                    try:
                        if preTrigger == 0 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #CaptureStart
                            print('Capture Start\r',flush=True)
                            BeforeData = Forte_GetEulerAngles(glove)
                            AmountOfChange = [0,0,0]
                            startTime = time.time()
                            preTrigger = 1

                        elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #Is Capturing
                            DeltaIMU = np.array(Forte_GetEulerAngles(glove)) - np.array(BeforeData)
                            BeforeData = Forte_GetEulerAngles(glove)
                            InclinationData[0][0].append(DeltaIMU[0])
                            InclinationData[0][1].append(DeltaIMU[1])
                            InclinationData[0][2].append(DeltaIMU[2])

                        elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 0: #End Capturing
                            if len(DataProcess.delzero(InclinationData[0][1]))<20 or len(DataProcess.delzero(InclinationData[0][2]))<20 or len(DataProcess.delzero(InclinationData[0][0]))<20:
                                InclinationData = [[[],[],[]],0,0]
                                preTrigger = 0
                                print('re\r',flush=True)
                                continue

                            deltaTime = time.time() - startTime
                            InclinationData[1]=MotionIndex
                            InclinationData[2]=deltaTime
                            print()
                            C= input('save?')
                            if C=='':
                                save.append(InclinationData)
                                print('saved'+str(len(save))+'\r',flush=True)
                                Iterator+=1                    
                            InclinationData = [[[],[],[]],0,0]
                            preTrigger = 0
                            if Iterator>=Routine:
                                print(FileName,"Saving...")
                                np.save(FilePath+FolderName+FileName,save,True)
                                print("SaveComplete")
                                break
            
                    except(GloveDisconnectedException):
                        print("Glove is Disconnected")
                        pass

            except(KeyboardInterrupt):
                Forte_DestroyDataGloveIO(glove) #Get Inclination Data
        elif Mode == 1: 
            print("Read Data Mode")
            Motion =(np.load(FilePath+FolderName+FileName,allow_pickle=True))

            print('Motion'+str(MotionIndex)+' loaded')
            for i in range(len(Motion)):
                #Mat1.append(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][0]))),Motion[i][2],57))
                #Mat2.append(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][1]))),Motion[i][2],57))
                #Mat3.append(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][2]))),Motion[i][2],57))
                #TT=Motion[i][2]/len(Mat1[0])

                #Mat1.append([TT for j in range(len(Mat1[0]))])
                #Mat2.append([TT for j in range(len(Mat2[0]))])
                #Mat3.append([TT for j in range(len(Mat3[0]))])
                #print(Mat1[1])
                plt.figure(figsize=(16,12))           

                DataIO.ShowGraph((DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][0]))),Motion[i][2],57)),331)
                DataIO.ShowGraph((DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][1]))),Motion[i][2],57)),332)
                DataIO.ShowGraph((DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][2]))),Motion[i][2],57)),333)
                DataIO.ShowGraph(DataProcess.Normalize(DataProcess.Integration(DataProcess.Constraction(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][0]))),1,150)),100))),334)
                DataIO.ShowGraph(DataProcess.Normalize(DataProcess.Integration(DataProcess.Constraction(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][1]))),1,150)),100))),335)
                DataIO.ShowGraph(DataProcess.Normalize(DataProcess.Integration(DataProcess.Constraction(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][2]))),1,150)),100))),336)
                DataIO.ShowGraph((DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][0]))),1,100)),337)
                DataIO.ShowGraph((DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][1]))),1,100)),338)
                DataIO.ShowGraph((DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][2]))),1,100)),339)
                plt.show() #DataRead
                plt.figure(figsize=(16,12))           

                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][0])),50))),331)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][1])),50))),332)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][2])),50))),333)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][0])),100))),334)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][1])),100))),335)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][2])),100))),336)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][0])),200))),337)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][1])),200))),338)
                #DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(Motion[i][0][2])),200))),339)
                DataIO.ShowGraph(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][0]))),Motion[i][2],57)),331)
                DataIO.ShowGraph(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][1]))),Motion[i][2],57)),332)
                DataIO.ShowGraph(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(Motion[i][0][2]))),Motion[i][2],57)),333)
                plt.show() #DataRead
                #print(Motion[i][2])
        elif Mode == 2:
            print("Combine Mode")
            savetemp = []
            #savetemp.extend(np.load(FilePath+FolderName+'/Motion0'+'.npy',allow_pickle=True))
            #print('Motion0 added'+str(len(savetemp)))


            for i in range(st,ed,1):
                Motion=[]
                Motion.extend(np.load(FilePath+FolderName+'/Motion'+str(i)+'_00'+'.npy',allow_pickle=True))#나

                savetemp.extend(Motion)
                print('Motion'+str(i)+' added'+str(len(Motion)))
            print(len(savetemp))
        
            for i in range(len(savetemp)):
                if savetemp[i][1]==0:
                    savetemp[i][0][0]=DataProcess.Nomalize(DataProcess.DynamicSampling(savetemp[i][0][0],hyper))
                    savetemp[i][0][1]=DataProcess.Nomalize(DataProcess.DynamicSampling(savetemp[i][0][1],hyper))
                    savetemp[i][0][2]=DataProcess.Nomalize(DataProcess.DynamicSampling(savetemp[i][0][2],hyper))
                else:
                    if "dynamic" in Date:
                        savetemp[i][0][0]=DataProcess.Normalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(savetemp[i][0][0])),hyper)))
                        savetemp[i][0][1]=DataProcess.Normalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(savetemp[i][0][1])),hyper)))
                        savetemp[i][0][2]=DataProcess.Normalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(savetemp[i][0][2])),hyper)))
                    else:
                        savetemp[i][0][0]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(savetemp[i][0][0]))),1,hyper))
                        savetemp[i][0][1]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(savetemp[i][0][1]))),1,hyper))
                        savetemp[i][0][2]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(savetemp[i][0][2]))),1,hyper))
   
            np.save(FilePath+'IMU_Gesture_Recognition_Model_And_Data'+'/CombinedData/'+str(hyper)+Date,savetemp,True)
            print("CombinedData","Operation Complete") #CombineData
        elif Mode == 3:
            print("LSTMdata Mode")
            savetemp = []
            #savetemp.extend(np.load(FilePath+FolderName+'/Motion0'+'.npy',allow_pickle=True))
            #print('Motion0 added'+str(len(savetemp)))
            for i in range(11,21,1):
                Motion=[]
                Motion.extend(np.load(FilePath+FolderName+'/Motion'+str(i)+'_00'+'.npy',allow_pickle=True))#나

                savetemp.extend(Motion)
                print('Motion'+str(i)+' added'+str(len(Motion)))
            print(len(savetemp))
        
            for i in range(len(savetemp)):
                if savetemp[i][1]==0:
                    savetemp[i][0][0]=DataProcess.Normalize(DataProcess.DynamicSampling(savetemp[i][0][0],hyper))
                    savetemp[i][0][1]=DataProcess.Normalize(DataProcess.DynamicSampling(savetemp[i][0][1],hyper))
                    savetemp[i][0][2]=DataProcess.Normalize(DataProcess.DynamicSampling(savetemp[i][0][2],hyper))
                else:
                    savetemp[i][0][0]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(savetemp[i][0][0]))),1,hyper))
                    savetemp[i][0][1]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(savetemp[i][0][1]))),1,hyper))
                    savetemp[i][0][2]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(savetemp[i][0][2]))),1,hyper))
                    #savetemp[i][0][0]=DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(savetemp[i][0][0])),hyper)))
                    #savetemp[i][0][1]=DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(savetemp[i][0][1])),hyper)))
                    #savetemp[i][0][2]=DataProcess.Nomalize(DataProcess.Integration(DataProcess.DynamicSampling(DataProcess.delover(DataProcess.delzero(savetemp[i][0][2])),hyper)))
   
            np.save(FilePath+'IMU_Gesture_Recognition_Model_And_Data'+'/CombinedData/'+"LSTMdata",savetemp,True)
            print("CombinedData","Operation Complete") #LSTMData
        elif Mode == 4: 
            Data = [[[],[],[]],0,0]
            InclinationData = [[[],[],[]],0,0]
            try:
                while True:
                    try:
                        if preTrigger == 0 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #CaptureStart
                            print('Capture Start\r',flush=True)
                            BeforeData = Forte_GetEulerAngles(glove)
                            AmountOfChange = [0,0,0]
                            startTime = time.time()
                            preTrigger = 1

                        elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #Is Capturing
                            DeltaIMU = np.array(Forte_GetEulerAngles(glove)) - np.array(BeforeData)
                            BeforeData = Forte_GetEulerAngles(glove)
                            Data[0][0].append(Forte_GetEulerAngles(glove)[0])
                            Data[0][1].append(Forte_GetEulerAngles(glove)[1])
                            Data[0][2].append(Forte_GetEulerAngles(glove)[2])

                            InclinationData[0][0].append(DeltaIMU[0])
                            InclinationData[0][1].append(DeltaIMU[1])
                            InclinationData[0][2].append(DeltaIMU[2])

                        elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 0: #End Capturing
                            deltaTime = time.time() - startTime
                            InclinationData[1]=MotionIndex
                            InclinationData[2]=deltaTime
                            Data[1]=MotionIndex
                            Data[2]=deltaTime
                            print()
                            C= input('save?')
                            if C=='':
                                plt.figure(figsize=(15,12))           
                                save.append(InclinationData)
                                save.append(Data)
                                DataIO.ShowGraph(DataProcess.delzero(InclinationData[0][0]),331)
                                DataIO.ShowGraph(DataProcess.delzero(InclinationData[0][1]),332)
                                DataIO.ShowGraph(DataProcess.delzero(InclinationData[0][2]),333)
                           
                                DataIO.ShowGraph(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][0]))),334)
                                DataIO.ShowGraph(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][1]))),335)
                                DataIO.ShowGraph(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][2]))),336)

                                DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][0]))),1,hyper)),337)
                                DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][1]))),1,hyper)),338)
                                DataIO.ShowGraph(DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][2]))),1,hyper)),339)
                                plt.show()
                                plt.figure(figsize=(15,12))
                                DataIO.ShowGraph(DataProcess.delzero2(DataProcess.delzero2(InclinationData[0][0])),331)
                                DataIO.ShowGraph(DataProcess.delzero2(DataProcess.delzero2(InclinationData[0][1])),332)
                                DataIO.ShowGraph(DataProcess.delzero2(DataProcess.delzero2(InclinationData[0][2])),333)

                                DataIO.ShowGraph(DataProcess.Integration(DataProcess.delzero2(DataProcess.delzero2(InclinationData[0][0]))),334)
                                DataIO.ShowGraph(DataProcess.Integration(DataProcess.delzero2(DataProcess.delzero2(InclinationData[0][1]))),335)
                                DataIO.ShowGraph(DataProcess.Integration(DataProcess.delzero2(DataProcess.delzero2(InclinationData[0][2]))),336)

                                DataIO.ShowGraph(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero2(InclinationData[0][0]))),1,hyper)),337)
                                DataIO.ShowGraph(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero2(InclinationData[0][1]))),1,hyper)),338)
                                DataIO.ShowGraph(DataProcess.Differential(DataProcess.HyperSampling(DataProcess.Integration(DataProcess.delover(DataProcess.delzero2(InclinationData[0][2]))),1,hyper)),339)
                                #DataIO.ShowGraph(DataProcess.Nomalize((DataProcess.delover(DataProcess.delzero(InclinationData[0][1])))),335)
                                #DataIO.ShowGraph(DataProcess.Nomalize((DataProcess.delover(DataProcess.delzero(InclinationData[0][2])))),336)

                                #DataIO.ShowGraph((((DataProcess.delzero(InclinationData[0][0])))),337)
                                #DataIO.ShowGraph((((DataProcess.delzero(InclinationData[0][1])))),338)
                                #DataIO.ShowGraph((((DataProcess.delzero(InclinationData[0][2])))),339)
                                plt.show()
                            InclinationData = [[[],[],[]],0,0]
                            Data = [[[],[],[]],0,0]
                            preTrigger = 0
                    except(GloveDisconnectedException):
                        print("Glove is Disconnected")
                        pass

            except(KeyboardInterrupt):
                Forte_DestroyDataGloveIO(glove) #Show Data

    except(KeyboardInterrupt):
        Forte_DestroyDataGloveIO(glove)
        exit()

def GetTestData():
    InclinationData = [[],[],[]]
    OriginData=[[],[],[]]
    DeltaIMU = []
    InitializedData = []
    preTrigger = 0
    deltaT=0
    try:
        while True:
            print(((Forte_GetSensorsRaw(glove)[0]+Forte_GetSensorsRaw(glove)[1])),Forte_GetSensorsRaw(glove)[2]+Forte_GetSensorsRaw(glove)[3],
            Forte_GetSensorsRaw(glove)[4]+Forte_GetSensorsRaw(glove)[5],Forte_GetSensorsRaw(glove)[6]+Forte_GetSensorsRaw(glove)[7],
            Forte_GetSensorsRaw(glove)[8]+Forte_GetSensorsRaw(glove)[9],end='\r',flush=True)
            try:
                if preTrigger == 0 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #CaptureStart
                    print('Capture Start\r',flush=True)
                    BeforeData = Forte_GetEulerAngles(glove)
                    preTrigger = 1

                elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #Is Capturing
                    

                    DeltaIMU = np.array(Forte_GetEulerAngles(glove)) - np.array(BeforeData)
                    BeforeData = Forte_GetEulerAngles(glove)

                    InclinationData[0].append(DeltaIMU[0])
                    InclinationData[1].append(DeltaIMU[1])
                    InclinationData[2].append(DeltaIMU[2])

                    OriginData[0].append(Forte_GetEulerAngles(glove)[0])
                    OriginData[1].append(Forte_GetEulerAngles(glove)[1])
                    OriginData[2].append(Forte_GetEulerAngles(glove)[2])

                elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 0: #End Capturing
                    
                    InclinationData[0]=DataProcess.delzero(InclinationData[0])
                    InclinationData[1]=DataProcess.delzero(InclinationData[1])
                    InclinationData[2]=DataProcess.delzero(InclinationData[2])


                    if len(InclinationData[1])<20 or len(InclinationData[2])<20 or len(InclinationData[0])<20:
                        InclinationData = [[],[],[]]
                        OriginData=[[],[],[]]
                        preTrigger = 0
                        print('re\r',flush=True)
                        continue
                    deltaT=time.time()

                    if 'dynamic' in Date:
                        InclinationData[0]=DataProcess.Normalize(DataProcess.Integration(DataProcess.DynamicSampling((DataProcess.delover(InclinationData[0])),hyper)))
                        InclinationData[1]=DataProcess.Normalize(DataProcess.Integration(DataProcess.DynamicSampling((DataProcess.delover(InclinationData[1])),hyper)))
                        InclinationData[2]=DataProcess.Normalize(DataProcess.Integration(DataProcess.DynamicSampling((DataProcess.delover(InclinationData[2])),hyper)))
                    else:
                        InclinationData[0]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration((DataProcess.delover(InclinationData[0]))),1,hyper))
                        InclinationData[1]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration((DataProcess.delover(InclinationData[1]))),1,hyper))
                        InclinationData[2]=DataProcess.Normalize(DataProcess.HyperSampling(DataProcess.Integration((DataProcess.delover(InclinationData[2]))),1,hyper))
                    now=time.time()
                    C= input('save?')
                    if C=='':
                        return InclinationData, now-deltaT

                    InclinationData = [[],[],[]]
                    OriginData = [[],[],[]]
                    preTrigger = 0

            except(GloveDisconnectedException):
                print("Glove is Disconnected")
                pass

    except(KeyboardInterrupt):
        Forte_DestroyDataGloveIO(glove)
        exit()
        
if tri:
    if Mode==0:
        GenerateData(Mode,MotionIndex)
        #for i in range(11,21,1):
        #    GenerateData(Mode,i)
    else:
        GenerateData(Mode,MotionIndex)
