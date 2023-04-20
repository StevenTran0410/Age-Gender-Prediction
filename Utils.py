from deepface import DeepFace
from keras.layers import Lambda, Dense, Input
from keras.models import Model
from keras import backend as K
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

output_indexes = np.array([i for i in range(0, 101)])
genders = ['Male', 'Female']

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

def load_model(name):
    if name == 'ArcFace':
        # Load the Facenet model
        pretrained_model = DeepFace.build_model('ArcFace')

        # Freeze all layers in the base model
        for layer in pretrained_model.layers[:148]:
            layer.trainable = False
            
        # Remove the last two layers from the model
        new_layer_list = pretrained_model.layers[:-2]

        # Create a new model with the modified layer list
        pretrained_model = Model(inputs=pretrained_model.input, outputs=new_layer_list[-1].output)

        # Create a new input layer
        inputs = Input(shape=(112, 112, 3))
        x = Lambda(lambda x: (x - 127.5) / 128)(inputs)
        x = pretrained_model(x)

        # Add a new output layer
        predictions_age = Dense(101, activation='softmax', name='age_output')(x)
        predictions_gender = Dense(2, activation='softmax', name='gender_output')(x)

        # Create a new model with the modified architecture
        age_gender_model = Model(inputs=inputs, outputs=[predictions_age, predictions_gender])
        # Compile the model
        age_gender_model.compile(optimizer='adam', loss={'age_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'}, metrics={'age_output': age_mae, 'gender_output': 'accuracy'})
        age_gender_model.load_weights('Age_Gender_model.h5')
        
    elif name == 'FaceNet':
        # Load the Facenet model
        pretrained_model = DeepFace.build_model('Facenet')

        # Freeze all layers in the base model
        for layer in pretrained_model.layers[:341]:
            layer.trainable = False
            
        # Remove the last two layers from the model
        new_layer_list = pretrained_model.layers[:-2]

        # Create a new model with the modified layer list
        pretrained_model = Model(inputs=pretrained_model.input, outputs=new_layer_list[-1].output)

        # Create a new input layer
        inputs = Input(shape=(160, 160, 3))
        x = Lambda(lambda x: x / 127.5 - 1)(inputs)
        x = pretrained_model(x)

        # Add a new output layer
        predictions_age = Dense(101, activation='softmax', name='age_output')(x)
        predictions_gender = Dense(2, activation='softmax', name='gender_output')(x)

        # Create a new model with the modified architecture
        age_gender_model = Model(inputs=inputs, outputs=[predictions_age, predictions_gender])

        # Compile the model
        age_gender_model.compile(optimizer='adam', loss={'age_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'}, metrics={'age_output': age_mae, 'gender_output': 'accuracy'})
        age_gender_model.load_weights('Age_Gender_model_FaceNet.h5')
        
    else:
        print('Not Available Model')
    
    return age_gender_model

def result(img, model):
    img = cv2.resize(img, (112, 112))
    test_img = image.img_to_array(img)
    test_img = np.expand_dims(test_img, axis = 0)
    predictions = model.predict(test_img)
    apparent_age = np.round(np.sum(predictions[0] * output_indexes, axis = 1))
    gender = np.argmax(predictions[1])
    
    return int(apparent_age[0]), genders[gender]