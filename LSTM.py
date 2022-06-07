import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import DensityController as DC 

mode=1# 0:Test 1:Train and Test

def TrainGraph():
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],'b-',label='loss')
    plt.plot(history.history['val_loss'],'r--',label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],'g-',label='accuracy')
    #plt.plot(history.history['val_accuracy'],'k--',label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0.7,1)
    plt.legend() 
    plt.show()


if mode==0:
    print('Loading LSTM Model')
    model= DC.tf.keras.models.load_model(DC.FilePath+'IMU_Gesture_Recognition_Model_And_Data'+'/TrainModel'+'/GestureRecognitionModel'+str(DC.hyper)+"_LSTM"+'.h5')
    model.summary()

    (trainX,trainY),(testX,testY)=DC.DataIO.DataRead('IMU_Gesture_Recognition_Model_and_Data/'+'CombinedData/'+str(DC.hyper)+DC.Date,3,DC.hyper,True)

    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)

    resalt=model.evaluate((testX),(testY),verbose=1)
    print('loss :', resalt[0], 'correntness:',resalt[1]*100,"%")
    DC.DataIO.GenerateMatrix("LSTM",model,testX,testY)

    #print('Test Start')
    DC.DataIO.GetTest(model) #TestMode
else:

    (trainX,trainY),(testX,testY)=DC.DataIO.DataRead('IMU_Gesture_Recognition_Model_and_Data/'+'CombinedData/'+str(DC.hyper)+DC.Date,3,DC.hyper,False)
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)

    #print(np.shape(trainX))
    #print(np.shape(testX))

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(units = DC.hyper, return_sequences=True, input_shape=(3,DC.hyper)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.LSTM(units = DC.hyper, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.LSTM(units = DC.hyper, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=21, activation = 'relu'))

    model.add(tf.keras.layers.LSTM(units = DC.hyper))
    model.add(tf.keras.layers.Dense(units=21, activation = 'softmax'))


    model.compile(optimizer= tf.keras.optimizers.Adam(),
                    loss= 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()

    history = model.fit(trainX,trainY,epochs=50,validation_split=0.01)
    model.save(DC.FilePath+'IMU_Gesture_Recognition_Model_And_Data'+'/TrainModel'+'/GestureRecognitionModel'+str(DC.hyper)+"_LSTM"+'.h5')

    result=model.evaluate((testX),(testY),verbose=1)
    print('loss :', result[0], 'correntness:',result[1]*100,"%")
    #TrainGraph()
    DC.DataIO.GenerateMatrix("LSTM",model,testX,testY)