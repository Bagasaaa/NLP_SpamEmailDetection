from flask import Flask, flash, request, redirect, url_for, render_template, Markup, jsonify
from werkzeug.utils import secure_filename
from flask import send_from_directory
import pandas as pd
import joblib

app = Flask(__name__, template_folder='templates')
app.secret_key = 'bagas_data_science'

##### home interface as .html
@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')

##### PREPROCESSING TEXT (INPUT TEXT) #####
@app.route("/book_your_hotel_room", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        no_of_adults = request.form.get('no_of_adults')
        no_of_children = request.form.get('no_of_children')
        no_of_weekend_nights = request.form.get('no_of_weekend_nights')
        no_of_week_nights = request.form.get('no_of_week_nights')
        room_type_reserved = request.form.get('room_type_reserved')
        arrival_time = request.form.get('arrival_time')
        df = pd.DataFrame({'Adults': [int(no_of_adults)],
                           'Children': [int(no_of_children)],
                           'WeekendNights': [int(no_of_weekend_nights)],
                           'WeekNights': [int(no_of_week_nights)],
                           'RoomType': [room_type_reserved],
                           'checkInDate': [pd.to_datetime(arrival_time)]})
        df['checkInDate'] = df['checkInDate'].astype('int64') // 10**9
        from sklearn.preprocessing import LabelEncoder

        labelEncoder_roomType = LabelEncoder()

        df['RoomType'] = labelEncoder_roomType.fit_transform(df['RoomType'])
        # Memuat kembali model dari file
        loaded_model = joblib.load('logistic_regression_model.sav')
        result = loaded_model.predict(df)
        result = result.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "Full Booked", 1: "Booking Success"}
        result = [labels[result]]
            
        return redirect(url_for("cleansing", text=result))

    return render_template("input_text.html")

@app.route("/<text>", methods=['GET'])
def cleansing(text):
    return f'Booking Status: {text}'

if __name__ == '__main__':
    app.run(debug=True)