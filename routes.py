from flask import Flask, request, jsonify

import numpy as np

from tagflow.src.predict import predict
from tagflow.src.hough import hough_circle

app = Flask(__name__)


@app.route('/')
def root():
    return 'Hi there'


@app.route('/track', methods=['POST'])
def track():

    if request.method == 'POST':

        payload = request.get_json(force=True)
        imt = np.array(payload['images'])
        r0 = np.array(payload['points'])

        y1 = predict(imt, r0)

        result = {
            'prediction': y1.tolist()
        }

        return jsonify(result)


@app.route('/hough', methods=['POST'])
def hough():

    if request.method == 'POST':

        payload = request.get_json(force=True)
        payload['image'] = np.array(payload['image'])

        r0, circle = hough_circle(**payload)

        result = {
            'points': r0.tolist(),
            'roi': circle.tolist()
        }

        return jsonify(result)
    

if __name__ == '__main__':
    app.run(port=5000, debug=True)
