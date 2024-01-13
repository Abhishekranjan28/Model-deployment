from flask import Flask,render_template,request
import pickle

tokenizer=pickle.load(open("model/cv.pkl","rb"))
model=pickle.load(open("model/clf.pkl","rb"))

app=Flask(__name__)

@app.route("/")

def home():
    
    return render_template("index.html")

@app.route("/predict",methods=["POST"])

def predict():
    email_text=request.form.get("email-content")
    tokenize_email=tokenizer.transform([email_text])
    predictions=model.predict(tokenize_email)
    if predictions==1:
        predictions=1
    else:
        predictions=-1
    return render_template("index.html",predictions=predictions,email_text=email_text)

if __name__=="__main__":
    app.run(debug=True)


 