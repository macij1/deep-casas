#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from tensorflow.keras.models import Sequential

def get_LSTM(input_dim, output_dim, max_length, no_activities):
    model = Sequential(name='LSTM')
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True))
    model.add(LSTM(units=output_dim))
    model.add(Dense(units=no_activities, activation='softmax'))
    return model

def get_biLSTM(input_dim, output_dim, max_length, no_activities):
    model = Sequential(name='biLSTM')
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True))
    model.add(Bidirectional(LSTM(units=output_dim)))
    model.add(Dense(units=no_activities, activation='softmax'))
    return model

def get_Ensemble2LSTM(input_dim, output_dim, max_length, no_activities):
    # In Keras 3.0, the Merge layer is replaced with functional API or concatenation
    model1 = Sequential()
    model1.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True))
    model1.add(Bidirectional(LSTM(units=output_dim)))
    
    model2 = Sequential()
    model2.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True))
    model2.add(LSTM(units=output_dim))
    
    # Use functional API for merging
    inputs1 = model1.input
    outputs1 = model1.output
    inputs2 = model2.input
    outputs2 = model2.output
    
    merged = keras.layers.Concatenate()([outputs1, outputs2])
    output = Dense(units=no_activities, activation='softmax')(merged)
    
    model = keras.Model(inputs=[inputs1, inputs2], outputs=output, name='Ensemble2LSTM')
    return model

def get_CascadeEnsembleLSTM(input_dim, output_dim, max_length, no_activities):
    # Similar to Ensemble2LSTM, use functional API
    model1 = Sequential()
    model1.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True))
    model1.add(Bidirectional(LSTM(units=output_dim, return_sequences=True)))
    
    model2 = Sequential()
    model2.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True))
    model2.add(LSTM(units=output_dim, return_sequences=True))
    
    inputs1 = model1.input
    outputs1 = model1.output
    inputs2 = model2.input
    outputs2 = model2.output
    
    merged = keras.layers.Concatenate()([outputs1, outputs2])
    x = LSTM(units=output_dim)(merged)
    output = Dense(units=no_activities, activation='softmax')(x)
    
    model = keras.Model(inputs=[inputs1, inputs2], outputs=output, name='CascadeEnsembleLSTM')
    return model

def get_CascadeLSTM(input_dim, output_dim, max_length, no_activities):
    model = Sequential(name='CascadeLSTM')
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True))
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True)))
    model.add(LSTM(units=output_dim))
    model.add(Dense(units=no_activities, activation='softmax'))
    return model

def compileModel(model):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
