from flask import Flask, request, render_template, redirect
import joblib
import pandas as pd

app = Flask(__name__)
pipeline = joblib.load('software_success_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Stakeholder_Engagement_Index': float(request.form['stakeholder_engagement']),
            'Developer_Skills_Index': float(request.form['dev_skills']),
            'Financial_Investment_Index': float(request.form['financial_investment']),
            'Risk_Management_Index': float(request.form['risk_management']),
            'Communication_Effectiveness_Index': float(request.form['communication']),
            'Stakeholder_Collaboration_Index': float(request.form['collaboration']),
            'User_Feedback_Index': float(request.form['user_feedback']),
            'Challenge_Scalability': float(request.form['challenge_scalability']),
            'Challenge_Financial': float(request.form['challenge_funding']),
            'Challenge_Integration': float(request.form['challenge_integration']),
            'Challenge_Feedback': float(request.form['challenge_feedback']),
            'Challenge_TeamDynamics': float(request.form['challenge_team']),
            'Age': request.form['age'],
            'Gender': request.form['gender'],
            'Role': request.form['role'],
            'Experience_Years': request.form['experience_years']
        }

        df_input = pd.DataFrame([input_data])
        predicted_score = round(float(pipeline.predict(df_input)[0]), 2)

        return render_template('result.html', prediction=predicted_score)

    except Exception:
        return redirect('/result?status=none')


@app.route('/result')
def result():
    status = request.args.get('status')
    if status == 'none':
        return render_template('result.html', prediction=None)
    return redirect('/')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)