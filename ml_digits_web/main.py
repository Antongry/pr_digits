import pickle
import numpy
from flask import Flask, render_template, request, jsonify

model = None

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        json_data = request.json
        pixels = numpy.array(json_data['pixels']).T
        pixels_array = pixels.reshape(28 * 28)
        prediction = str(model.predict([pixels_array, ])[0])
        proba = model.predict_proba([pixels_array, ])[0]
        proba = list(zip(range(10), proba))
        proba = sorted(proba, key=lambda x:x[1], reverse=True)
        print(prediction)
        return jsonify({'predicted': prediction, 'probs': proba[:5]})


if __name__ == '__main__':
    with open('tree_model1.pickle', 'rb') as f:
        model = pickle.load(f)
    app.run()
