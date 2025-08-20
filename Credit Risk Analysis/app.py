from flask import Flask, render_template, request
import pickle

model = pickle.load(open('credictrisk.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def loadPage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve all form data with type conversion as needed
        b = float(request.form["Annual_Income"])
        c = int(request.form["Num_Bank_Accounts"])
        d = int(request.form["Num_Credit_Card"])
        e = float(request.form["Interest_Rate"])
        f = int(request.form["Num_of_Loan"])
        h = int(request.form["Delay_from_due_date"])
        i = int(request.form["Num_of_Delayed_Payment"])
        j = float(request.form["Changed_Credit_Limit"])
        k = int(request.form["Num_Credit_Inquiries"])
        l = float(request.form["Outstanding_Debt"])
        m = float(request.form["Credit_Utilization_Ratio"])
        o = float(request.form["Total_EMI_per_month"])
        p = float(request.form["Amount_invested_monthly"])
        q = float(request.form["Monthly_Balance"])
        
        # Prepare the input feature array for prediction
        t = [[ b, c, d, e, f, h, i, j, k, l, m, o, p, q]]
        y = model.predict(t)


        prediction_map = {0: "Good", 1: "Poor", 2: "Standard"}
        prediction_text = prediction_map.get(y[0], "Unknown") 
        
        
        # Render template with prediction and input values
        return render_template('next.html', Annual_Income=b, Num_Bank_Accounts=c, 
                               Num_Credit_Card=d, Interest_Rate=e, Num_of_Loan=f, 
                               Delay_from_due_date=h, Num_of_Delayed_Payment=i, Changed_Credit_Limit=j,
                               Num_Credit_Inquiries=k, Outstanding_Debt=l, Credit_Utilization_Ratio=m,
                             Total_EMI_per_month=o, Amount_invested_monthly=p,
                               Monthly_Balance=q, prediction=prediction_text)
    except ValueError as ve:
        print("Error converting input:", ve)
        return "Error processing the form data. Please ensure all inputs are correctly formatted.", 400


if __name__ == "__main__":
    app.run(debug=True)
