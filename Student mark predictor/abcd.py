import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

import sklearn

app=Flask(__name__,template_folder='templets')

model = joblib.load('student_mark_predictor_model.pkl')

df = pd.DataFrame()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global df

    input_feature = [int(x) for x in request.form.values()]
    feature_values = np.array(input_feature)

    # input validation
    if input_feature[0] < 0 or input_feature[0] > 24:
        return render_template('index.html',
                               prediction_text='Please enter valid hour between 1 to 24 if you are living on earth')

    output = model.predict([feature_values])[0][0].round(2)

    # input and predicted value store in df and csv table also
    df = pd.concat([df, pd.DataFrame({'Study hour': input_feature, 'predicted output': [output]})], ignore_index=True)
    print(df)
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html',
                           prediction_text='You will get [{}%] marks, when You do study [{}] hour per day '.format(
                               output, int(feature_values)))


if __name__ == '__main__':
    app.run(debug=True, port=9090)
