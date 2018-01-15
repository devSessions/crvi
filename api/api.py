from flask import Flask, jsonify, request
import predict
import socket

app = Flask(__name__)
@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return (
        "Welcome Guest!!!"
    )

#to spedicy route after url
@app.route('/api', methods=['POST'])
def get_tasks():
    #get url from form
    # url = request.form['url']
    url = request.files['url']

    #sends url for prediction
    sender = predict.predict(url)

    #get values from prediction
    rec = sender.predict_only()

    # #list of out values
    # outputlist=[rec]

    # #for multiple json apis
    # tasks = []

    # tasks1 = [
    #     {
    #         'value': outputlist[0],

    #     },

    # ]
    # tasks.append(tasks1)
    # return jsonify({'tasks': tasks})
    return jsonify({'cash': rec})



if __name__ == '__main__':
    #for remote host
    ip = socket.gethostbyname(socket.gethostname())
    app.run(port=5000,host=ip)

    #for local host
    #app.run(debug=True, port=5000)