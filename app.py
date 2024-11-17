import os
from flask import Flask, request, render_template
from rpy2.robjects import r, pandas2ri
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Initialize Flask app
app = Flask(__name__)

# Activate R-Pandas conversion
pandas2ri.activate()

# Import required R packages
def load_models():
    try:
        # Import R packages explicitly
        e1071 = importr('e1071')
        class_pkg = importr('class')
        kernlab = importr('kernlab')

        # Define absolute paths for models
        knn_model_path = r"C:/Local volume/Programming/Machine Learning/Dashboard/models/KNN_heart.RData"
        naive_model_path = r"C:/Local volume/Programming/Machine Learning/Dashboard/models/NAIVE_BAYES_heart.RData"
        svm_model_path = r"C:/Local volume/Programming/Machine Learning/Dashboard/models/SVM_heart.RData"

        # Load models using their absolute paths
        r['load'](knn_model_path)
        r['load'](naive_model_path)
        r['load'](svm_model_path)

        print("Loaded R objects:", list(r['ls']()))
        return e1071  # Return e1071 package for Naive Bayes predictions
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Load models and get e1071 package
e1071 = load_models()

def normalize(x, min_val, max_val):
    try:
        return (x - min_val) / (max_val - min_val)
    except ZeroDivisionError:
        return 0

def prepare_input(input_data):
    try:
        train_predictors = r['train_predictors']

        # Normalize numeric fields
        input_data['age'] = normalize(input_data['age'], min(train_predictors.rx2('age')), max(train_predictors.rx2('age')))
        input_data['trestbps'] = normalize(input_data['trestbps'], min(train_predictors.rx2('trestbps')), max(train_predictors.rx2('trestbps')))
        input_data['chol'] = normalize(input_data['chol'], min(train_predictors.rx2('chol')), max(train_predictors.rx2('chol')))
        input_data['thalach'] = normalize(input_data['thalach'], min(train_predictors.rx2('thalach')), max(train_predictors.rx2('thalach')))
        input_data['oldpeak'] = normalize(input_data['oldpeak'], min(train_predictors.rx2('oldpeak')), max(train_predictors.rx2('oldpeak')))

        # Convert categorical variables to factors
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            input_data[col] = str(int(input_data[col]))

        # Ensure all columns are present
        for col in train_predictors.colnames:
            if col not in input_data:
                input_data[col] = '0'

        return pd.DataFrame([input_data])
    except Exception as e:
        raise RuntimeError(f"Error preparing input data: {e}")

def predict_from_r_model(model_name, input_data):
    try:
        r_input_data = pandas2ri.py2rpy(input_data)

        if model_name == 'knn':
            prediction = r['knn'](
                train=r['train_predictors'],
                test=r_input_data,
                cl=r['train_target'],
                k=21
            )
        elif model_name == 'naive':
            # Use e1071::predict directly for Naive Bayes
            prediction = e1071.predict(r['nb_model'], r_input_data)
        elif model_name == 'svm':
            prediction = r['predict'](r['svm_model'], r_input_data)
        else:
            raise ValueError("Invalid model name")

        # Convert prediction to Python string
        result = str(list(prediction)[0])
        
        # Map the prediction to a more meaningful output
        prediction_map = {
            '0': 'No Heart Disease',
            '1': 'Heart Disease Present'
        }
        
        return prediction_map.get(result, result)
    except Exception as e:
        raise RuntimeError(f"Prediction error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<model_name>', methods=['GET', 'POST'])
def model_predict(model_name):
    if request.method == 'POST':
        try:
            input_data = {
                "age": float(request.form['age']),
                "sex": float(request.form['sex']),
                "cp": float(request.form['cp']),
                "trestbps": float(request.form['trestbps']),
                "chol": float(request.form['chol']),
                "fbs": float(request.form['fbs']),
                "restecg": float(request.form['restecg']),
                "thalach": float(request.form['thalach']),
                "exang": float(request.form['exang']),
                "oldpeak": float(request.form['oldpeak']),
                "slope": float(request.form['slope']),
                "ca": float(request.form['ca']),
                "thal": float(request.form['thal'])
            }

            df_input = prepare_input(input_data)
            prediction = predict_from_r_model(model_name, df_input)
            
            return render_template(f'{model_name}.html', result=prediction)
        except Exception as e:
            return render_template(f'{model_name}.html', error=str(e))
    return render_template(f'{model_name}.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=False, use_reloader=False)