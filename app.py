from flask import Flask, render_template, request
import os
import numpy as np
from src.creditrisk.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/train', methods=['GET'])
def trainpage():
    os.system("python main.py")
    return render_template('index.html', train_success=True)


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html')

    try:
        person_age                 = float(request.form['person_age'])
        person_income              = float(request.form['person_income'])
        person_emp_length          = float(request.form['person_emp_length'])
        loan_amnt                  = float(request.form['loan_amnt'])
        loan_int_rate              = float(request.form['loan_int_rate'])
        loan_percent_income        = float(request.form['loan_percent_income'])
        cb_person_cred_hist_length = float(request.form['cb_person_cred_hist_length'])

        person_home_ownership      = request.form['person_home_ownership']   # RENT | OWN | MORTGAGE | OTHER
        loan_intent                = request.form['loan_intent']             # PERSONAL | EDUCATION | ...
        loan_grade                 = request.form['loan_grade']              # A | B | C | D | E | F | G
        cb_person_default_on_file  = request.form['cb_person_default_on_file']  # Y | N

        # ── Encode categoricals (must match your training label encoding) ────
        home_ownership_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
        loan_intent_map    = {
            'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2,
            'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5
        }
        loan_grade_map     = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        default_map        = {'N': 0, 'Y': 1}

        # ── Build feature vector — order must match X_train.csv columns ──────
        data = np.array([[
            person_age,
            person_income,
            home_ownership_map.get(person_home_ownership, 0),
            person_emp_length,
            loan_intent_map.get(loan_intent, 0),
            loan_grade_map.get(loan_grade, 0),
            loan_amnt,
            loan_int_rate,
            loan_percent_income,
            default_map.get(cb_person_default_on_file, 0),
            cb_person_cred_hist_length,
        ]])

        # ── Run pipeline ─────────────────────────────────────────────────────
        pipeline   = PredictionPipeline()
        prediction = pipeline.predict(data)        # returns array e.g. [0] or [1]
        pred_value = int(prediction[0])            # 0 = no default, 1 = default
        risk_label = 'HIGH' if pred_value == 1 else 'LOW'

        return render_template(
            'result.html',
            # ── prediction outputs ──
            prediction  = pred_value,
            risk_label  = risk_label,
            # ── numeric inputs (passed back for display) ──
            person_age                 = person_age,
            person_income              = person_income,
            person_emp_length          = person_emp_length,
            loan_amnt                  = loan_amnt,
            loan_int_rate              = loan_int_rate,
            loan_percent_income        = loan_percent_income,
            cb_person_cred_hist_length = cb_person_cred_hist_length,
            # ── categorical inputs ──
            person_home_ownership      = person_home_ownership,
            loan_intent                = loan_intent,
            loan_grade                 = loan_grade,
            cb_person_default_on_file  = cb_person_default_on_file,
        )

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return render_template('index.html', error=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081, debug=True)