from flask import Flask, escape, request
import json
import os

basedir = os.path.abspath(os.path.dirname(__file__))

CONFIGPATH = './configs/defaultconfig.json'

# Load applicaiton configs
config = json.loads(open(CONFIGPATH, 'r').read())

app = Flask(__name__)
app.config.update(config)


@app.route('/')
def d_view():
    print('hellp')
    return("")

# Load the model 
from server.views import *

#app.add_url_rule('/module/', view_func=views.PredictView.as_view('module'))


if __name__=='__main__':
    app.run()