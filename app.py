import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained Naive Bayes model
with open("naive.pkl", "rb") as f:
  model = pickle.load(f)

# HTML template with modern CSS styling embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Dashboard</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
        }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 35px 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            width: 100%;
            max-width: 650px;
        }
        h2 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 25px;
            font-weight: 700;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            margin-bottom: 5px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 13px;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input, select {
            width: 100%;
            padding: 11px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
            transition: all 0.2s ease;
            background-color: #f8fafc;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            outline: none;
            background-color: #fff;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        }
        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 25px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 25px;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            font-size: 16px;
            font-weight: 600;
        }
        .success {
            background-color: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .info {
            background-color: #eff6ff;
            color: #1e40af;
            border: 1px solid #bfdbfe;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Employee Status Prediction</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="education">Education</label>
                    <select name="education" id="education" required>
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">PHD</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="joining_year">Joining Year</label>
                    <input type="number" name="joining_year" id="joining_year" placeholder="e.g. 2018" required>
                </div>
                <div class="form-group">
                    <label for="city">City</label>
                    <select name="city" id="city" required>
                        <option value="0">Bangalore</option>
                        <option value="1">Pune</option>
                        <option value="2">New Delhi</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="payment_tier">Payment Tier</label>
                    <select name="payment_tier" id="payment_tier" required>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" name="age" id="age" placeholder="e.g. 26" required>
                </div>
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select name="gender" id="gender" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="ever_benched">Ever Benched</label>
                    <select name="ever_benched" id="ever_benched" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="experience">Experience in Domain</label>
                    <input type="number" name="experience" id="experience" placeholder="e.g. 3" required>
                </div>
            </div>
            <button type="submit">Run Prediction</button>
        </form>

        {% if prediction is not none %}
            <div class="result-card {% if prediction == 1 %}success{% else %}info{% endif %}">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  prediction_text = ""
  if request.method == "POST":
    try:
      # Map inputs precisely to the model's feature order[cite: 1]
      education = float(request.form["education"])
      joining_year = float(request.form["joining_year"])
      city = float(request.form["city"])
      payment_tier = float(request.form["payment_tier"])
      age = float(request.form["age"])
      gender = float(request.form["gender"])
      ever_benched = float(request.form["ever_benched"])
      experience = float(request.form["experience"])

      features = np.array([[
          education,
          joining_year,
          city,
          payment_tier,
          age,
          gender,
          ever_benched,
          experience,
      ]])

      pred = model.predict(features)[0]
      prediction = int(pred)
      prediction_text = (
          f"Prediction Result: Employee is likely to Leave (Class: {prediction})"
          if prediction == 1
          else (
              f"Prediction Result: Employee is likely to Stay (Class:"
              f" {prediction})"
          )
      )
    except Exception as e:
      prediction_text = f"Error processing input: {str(e)}"
      prediction = -1

  return render_template_string(
      HTML_TEMPLATE, prediction=prediction, prediction_text=prediction_text
  )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
