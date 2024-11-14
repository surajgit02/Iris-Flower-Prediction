from flask import Flask,request,render_template
import pickle
import numpy as np
app=Flask(__name__)
f=open("iris.pkl",'rb')
model=pickle.load(f)
@app.route("/")
def show():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    print(request.form)
    features=[float(x) for x in request.form.values()]
    #or
    '''sl=features[0]
    print(sl)'''
    final_features=[np.array(features)]
    print(final_features)
    result=model.predict(final_features)
    if result[0]==0:
        flower ='SETOSA'
    elif result[0]==1:
        flower="VERSICOLOR"
    else:
        flower="VIRGINICA"
    return render_template('index.html',predicted_flower="Flower is"+flower)
    # predicted_flower  is  a variable is a variable whisch is
    # also defined in HTML as following
    #<h1><font color="red">{{predicted_flower}}</font> </h1>

app.run(debug=True)