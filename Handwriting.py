import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train , y_train) , (x_test , y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


def train_mnist():
    class myCallback (tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch , log ={}):
            print ('Accuracy is ' + str(log.get('accuracy')))
            if(log.get('accuracy') > 0.99):
                print('\nreached 99 % accuracy so cancelling')
                self.model.stop_training = True



    callbacks = myCallback()

    model = tf.keras.Sequential(
    [

        tf.keras.layers.Flatten(input_shape = (28,28)),

        tf.keras.layers.Dense(512,activation = tf.nn.relu),
        tf.keras.layers.Dense(10 , activation = tf.nn.softmax)

    ]
    )

    model.compile(optimizer = tf.optimizers.Adam() , loss = 'sparse_categorical_crossentropy' , metrics = 'accuracy')

    model.fit(x_train,y_train , epochs = 2, callbacks = [callbacks])



train_mnist()
