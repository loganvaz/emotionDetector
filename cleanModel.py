def initializeModel(inputShape, outputDimension):

    X_input = Input(shape=inputShape)
    X = X_input
    
    X = Conv2D(64, kernel_size=8,strides=1,padding="same")(X)
    X = BatchNormalization(axis=3,name="zero_batch_norm")(X)
    X = Activation("relu")(X)
   
    X = Conv2D(128,kernel_size=5,strides=2,padding="valid")(X)
    X = BatchNormalization(axis=3,name="first_batch_normalization")(X)
    X = Activation("relu")(X)

    
    X = Conv2D(256, kernel_size = 4, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3, name="second_batch_norm")(X)
    X = Activation("relu")(X)
    
    X = Conv2D(300, kernel_size=3, strides=2, padding="valid")(X)
    X = BatchNormalization(axis=3,name="third_batch_norm")(X)
    X = Activation("relu")(X)
    X = Dropout(0.25)(X)
    
    X = ZeroPadding2D((1,1))(X)
   
    X = Conv2D(400, kernel_size=3, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3,name="fourth_batch_norm")(X)
    X = Activation("relu")(X)
    X = Dropout(0.35)(X)

    X = Conv2D(512, kernel_size=4, strides=1, padding="valid")(X)
    X = BatchNormalization(axis=3,name="fifth_batch_norm")(X)
    X = Activation("relu")(X)
    

    flat = Flatten(name="flatten_2")(X)
    flat = Dropout(0.485)(flat)


    flat= Dense(256, activation="relu",kernel_regularizer = regularizers.l2(.0000185))(flat)
    flat = Dropout(0.485)(flat)
    X = Dense(128, activation="relu", kernel_regularizer = regularizers.l2(.00002))(flat)
    X = Dropout(0.5)(X)
    X = Dense(256,kernel_regularizer = regularizers.l2(.0000185) )(X)#no activation function, took away regularizer so it could scale below activation freely
    X = Dropout(0.485)(X)
    X = Activation("relu")(X+flat)    
    
    X = Dense(outputDimension, activation ="softmax",kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name="myCNN")
    
    return model
