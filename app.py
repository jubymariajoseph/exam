from flask import Flask, render_template,request
import numpy
import pandas
import pickle


app = Flask(__name__)
with open("min.pkl",'rb') as f:
 min = pickle.load(f)
with open("model1.pkl",'rb') as f:
 model = pickle.load(f)
@app.route("/",methods=['GET','POST'])
def home():
    prediction = ''
    'Place'='Song_Quality'='energy'='danceability'='Semi_Final_Number'=	'Song_In_English'='duration'=''
    if request.method=='POST':
        Place = float(request.form.get('Place',''))
        Song_Quality = float(request.form.get('Song.Quality',''))
        energy = float(request.form.get('energy'))
        danceability= float(request.form.get('danceability',''))
        Semi_Final_Number = float(request.form.get('Semi_Final_Number',''))
        Song_In_English = float(request.form.get('Song_In_English',''))
        duration= float(request.form.get('duration',''))
        features = numpy.array([Place,Song_Quality,energy,danceability,Semi_Final_Number,Song_In_English,duration])
        feature_sca  = min.transform(features)
        prediction = model.predict(feature_sca)[0]
    return render_template("home.html",Place=Place,Song_Quality=Song_Quality,energy=energy,danceability=danceability,Semi_Final_Number=Semi_Final_Number,Song_In_English=Song_In_English,duration=duration)

if __name__ =="__main__":
    app.run(debug=True)

        
           