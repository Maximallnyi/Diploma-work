# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from main import calculate_metrics

class Classifier_FCN:

    def __init__(self,input_shape, nb_classes, verbose=False,build=True):
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
            min_lr=0.0001)

        return model 

    def fit(self, x_train, y_train, x_val, y_val,y_true, nb_epochs):
#         if not tf.test.is_gpu_available:
#             print('error')
#             exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 20
        #nb_epochs = 20

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, validation_data=(x_val,y_val))

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # convert the predicted from binary to integer 
        df_metrics = calculate_metrics(y_true, y_pred)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
        model = self.model
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred