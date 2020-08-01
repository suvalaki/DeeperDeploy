import importlib
from flask.views import MethodView
from flask import request
from server.__init__ import app

import threading
import numpy as np

# Global threads
training_thread = threading.Thread()


class PredictView(MethodView):

    def get(self):
        module = importlib.import_module(app.config["module"])
        print(module)
        print(str(app.config))
        return(str(app.config))


class LoadModelView(MethodView):

    def get(self):
        module = importlib.import_module(app.config["module"])
        model = importlib.import_module(
            app.config["module"] + '.wrapper'
        )
        app.config['model'] = model.ModelWrapper(app.config["model_args"])
        print('done')
        return('1')


class TrainModelView(MethodView):

    def get(self):

        global training_thread

        X_train = np.load(app.config["data_path"] + '/X_train.np')

        train = importlib.import_module(app.config["module"] + '.train.train')
        train(
            m1,
            X_train,
            y_train,
            X_test,
            y_test,
            num=100,
            samples=1,
            epochs=1500,
            iter_train=1,
            num_inference=1000,
            save=None,#"model_w_5",
            batch=True,
            temperature_function=lambda x: exponential_multiplicative_cooling(
                x, 1.0, 0.5, 0.98
            ),
            # temperature_function = lambda x: 0.1
            save_results="./gumble_results.txt",
            beta_z_method=z_cooling,
            beta_y_method=y_cooling,
            tensorboard=None,#"./logs/" + param_string + "/samples__" + str(1),
        )


        #inputs = request.json
        model = app.config['model']



app.add_url_rule('/module/', view_func=PredictView.as_view('module'))
app.add_url_rule('/model/', view_func=LoadModelView.as_view('model'))