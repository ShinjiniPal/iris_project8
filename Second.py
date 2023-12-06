from flask import Flask, request, Response
from joblib import load
import numpy as np

my_lr_model=load('Model/LogReg_model.joblib')
from flask import Flask

app = Flask(__name__)


@app.route( "/get_predictions",methods=['POST','GET'])
def testing():
    data=request.json
    user_sent_this_data=data.get('mydata')

    user_number=np.array(user_sent_this_data).reshape(1,-1)
    model_predictions=my_lr_model.predict(user_number)
    return Response(str(model_predictions))

if __name__=='__main__':
    app.run(debug=False, port=5001)




