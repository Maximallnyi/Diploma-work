import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from main import calculate_metrics

class Classifier_MLP:

    def __init__(self, input_shape, nb_classes, verbose=False,build=True):
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis 
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

        return model

    def fit(self, x_train, y_train, x_val, y_val,y_true, nb_epochs):
    # 		if not tf.test.is_gpu_available:
    # 			print('error')
    # 			exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 20
        #nb_epochs = 20

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, validation_data=(x_val,y_val))

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # convert the predicted from binary to integer 
        #y_pred = np.argmax(y_pred , axis=1)
        df_metrics = calculate_metrics(y_true, y_pred)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
        model = self.model
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred