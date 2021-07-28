from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import pickle

app=Flask(__name__)
api=Api(app)

data_arg=reqparse.RequestParser()
data_arg.add_argument("id" , type=str)
# load ML model
model=pickle.load(open('model.pkl', 'rb'))
class predict(Resource):
    def __init__(self):
        self.model1 = model
    def post(self):
        # parse data from post request
        args = data_arg.parse_args()
        # convert string into int list
        temp=args.id.strip('][').split(',')
        temp = [float(i) for i in temp]
        # predict output
        out=self.model1.predict([temp])
        # Return prediction
        return jsonify({"message":  int(out)})
api.add_resource(predict, '/')


if __name__ == '__main__':
    app.run(debug=True)