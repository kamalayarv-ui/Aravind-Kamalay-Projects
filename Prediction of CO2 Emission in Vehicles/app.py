from flask import Flask,render_template,request
import pickle
model=pickle.load(open('co2.pkl','rb'))
app=Flask(__name__)
#Interface between the server and Application wsgi

@app.route('/') #binds with the URL
def loadPage():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    a=request.form["make"]
    b=request.form["vehicles_class"]
    c=request.form["engine_size"]
    d=request.form["Cylinders"]
    e=request.form["transmission"]
    f=request.form["fuel_type"]
    g=request.form["fuel_consumption_city"]
    h=request.form["fuel_consumption_hwy"]
    fc_comb1=float(55*float(f)/100)
    fc_combo2=float(45*float(h)/100)
    i=float(fc_comb1+fc_combo2)
    t=[[int(a),int(b),float(c),int(d),int(e),int(f),float(g),float(h),float(i)]]
    y=model.predict(t)
    print(y)
    return render_template('next.html',j=a,k=b,l=c,m=d,n=e,o=f,p=g,q=h,r=i,x=y[0][0],resp=i)
if __name__=="__main__":
    app.run(debug=True)
