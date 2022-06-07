import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import DensityController as DC 
import time

Mode=1# 0:Test 1:Train and Test
#TrainDate='_211122_Demo'
TrainDate='_220106'

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
def CheckResalt(model, testX, testY, iterator):
    for i in range(iterator):
       te1 = np.argmax(model.predict(np.array(testX[i]).reshape(1,3,DC.hyper,1)))
       te2 = testY[i]
       if te1==te2:
            print(te1, te2, str(i)+'/'+str(iterator))
       else:
            print( 'Wrong Answer',te1, te2, str(i)+'/'+str(iterator))

def CAM(model):

    cam_model = tf.keras.Model(model.input,outputs=(model.layers[-3].output, model.layers[-1].output))
    cam_model.summary()

    gap_weights= model.layers[-1].get_weights()[0]
    gap_weights.shape
    
    plt.plot(gap_weights[0])
    plt.show()

        
if Mode==0:

    print('Loading CNN Model')
    model= DC.tf.keras.models.load_model(DC.FilePath+'IMU_Gesture_Recognition_Model_And_Data'+'/TrainModel'+'/GestureRecognitionModel'+str(DC.hyper)+"_CNN"+'.h5')
    model.summary()

    (trainX,trainY),(testX,testY)=DC.DataIO.DataRead('IMU_Gesture_Recognition_Model_and_Data/'+'CombinedData/'+str(DC.hyper)+DC.Date,3,DC.hyper,True)
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)

    resalt=model.evaluate((testX),(testY),verbose=1)
    print('loss :', resalt[0], 'correntness:',resalt[1]*100,"%")
    print("Saving Confusion Matrix")
    #DC.DataIO.GenerateMatrix("CNN",model,testX,testY)
    print("Save Complete")


    print('Test Start')
    DC.DataIO.GetTest(model) #TestMode

else:
    (trainX,trainY),(testX,testY)=DC.DataIO.DataRead('IMU_Gesture_Recognition_Model_and_Data/'+'CombinedData/'+str(DC.hyper)+DC.Date,3,DC.hyper,True)

    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)

    model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=(3,DC.hyper,1),kernel_size=(3,3),filters=32,padding='same',activation='relu'),
                                 tf.keras.layers.Conv2D(kernel_size=(1,3),filters=64,padding='same',activation ='relu'),
                                 tf.keras.layers.MaxPooling2D(strides=(1,2)),
                                 #tf.keras.layers.Dropout(rate=0.5),
                                 #tf.keras.layers.Conv2D(kernel_size=(1,3),padding='valid',filters=128,activation='relu'),
                                 #tf.keras.layers.Conv2D(kernel_size=(1,3),padding='valid',filters=256,activation='relu'),
                                 tf.keras.layers.MaxPooling2D(strides=(1,2)),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Flatten(),
                                 #tf.keras.layers.Dense(units=256, activation='relu'),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(units=128, activation='relu'),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(units=21, activation = 'softmax')
                                 ])

    model.compile(optimizer= tf.keras.optimizers.Adam(),
                  loss= 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit((trainX),(trainY),epochs=50,validation_split=0.01)

    model.save(DC.FilePath+'IMU_Gesture_Recognition_Model_And_Data'+'/TrainModel'+'/GestureRecognitionModel'+str(DC.hyper)+"_CNN"+'.h5')

    #CheckResalt(model,testX,testY,20)
    resalt=model.evaluate((testX),(testY),verbose=1)
    print('loss :', resalt[0], 'correntness:',resalt[1]*100,"%")

    #DC.DataIO.GenerateMatrix("CNN",model,testX,testY)

    #TrainGraph() #TrainMode
    #print('Test Start')
    DC.DataIO.GetTest(model) #TestMode
    