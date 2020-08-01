#%%
import tensorflow as tf 
import numpy as np 
import argparse 
import importlib 
from typing import Optional
from deeper.utils.function_helpers.decorators import inits_args
import os

tf.keras.backend.set_floatx("float64")

class BaseApp():

    def __init__(
        self,
        module_nm:str,
        model_args:dict,
        model_load:Optional[str]=None,
        data_paths:Optional[dict]=None,
        train_args:Optional[dict]=None,
        model_train:Optional[bool]=True,
        evaluation_callbacks:Optional[dict]=None,
        evaluation_args:Optional[dict]=None
    ):
        self.module_nm = module_nm
        self.model_args = model_args 
        self.model_load = model_load 
        self.data_paths = data_paths 
        self.train_args = train_args 
        self.model_train = model_train 
        self.evaluation_callbacks = evaluation_callbacks 
        self.evaluation_args = evaluation_args


    def _load_data(self, data_paths):
        # Load the data
        required_data = ['X_train', 'X_test', 'y_train', 'y_test']
        data_load_method = np.load
        X_train = data_load_method(data_paths['X_train']).astype(float)
        X_test = data_load_method(data_paths['X_test']).astype(float)
        y_train = data_load_method(data_paths['y_train']).astype(float)
        y_test = data_load_method(data_paths['y_test']).astype(float)
        return X_train, X_test, y_train, y_test


    def _init_model(self, model, model_args):
        mod = importlib.import_module(model + '.wrapper').ModelWrapper(
            model_args
        )
        return mod


    def start(self):

        # Create the model
        self.model = self._init_model(self.module_nm, self.model_args)

        # Load
        if (
            self.model_load is not None 
            and os.path.isfile(os.path.abspath(self.model_load))
        ):
            self.model.load_weights(self.model_load)


    def train(self, X_train, X_test, y_train, y_test):
        if self.model_train:
            self.model.train_from_config(
                X_train, y_train, X_test, y_test, self.train_args
            )
            if self.model_load is not None:
                self.model.save_weights(self.model_load)


    def run(self):

        # Load data
        X_train, X_test, y_train, y_test = self._load_data(self.data_paths)

        # Start Model
        self.start()

        # Train the model
        self.train(X_train, X_test, y_train, y_test)
        
        # Evaluation
        #if evaluation_callbacks is not None:
        #    pass
        y_train_pred = self.model.predict(X_train, **self.evaluation_args)
        y_test_pred = self.model.predict(X_test, **self.evaluation_args)

        output = {
            'y_train_pred': y_train_pred, 
            'y_test_pred': y_test_pred
        }

        return output




if __name__=='__main__':


    tf.keras.backend.set_floatx('float64')

    # Load configs
    import json 
    FP_CONFIG_MODEL = './configs/defaultconfig.json'
    config = json.loads(open(FP_CONFIG_MODEL,'r').read())
    config_module_nm = config['module']
    config_data_path = config['data_paths']
    config_model_args = config['model_args']
    config_train_args = config['training_args']


    #
    app = BaseApp(
        module_nm=config_module_nm,
        model_args=config_model_args,
        model_load=None,
        data_paths=config_data_path,
        train_args=config_train_args,
        model_train=True,
        evaluation_callbacks={},
        evaluation_args={}
    )

    #
    app.run()

    #


# %%
