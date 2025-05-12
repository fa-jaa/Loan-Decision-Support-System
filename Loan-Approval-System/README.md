# Loan Approval System with Explainable AI (XAI)

A Streamlit-based web application that helps account managers interpret loan approval model outputs using Explainable AI (XAI). The system provides risk assessment and actionable recommendations for loan applications.

## Features

- **Interactive Risk Assessment**: Enter applicant details using the provided input options to receive predictions and recommendations on loan approval or rejection, based on historical data.
- **Explainable AI Integration**: Uses SHAP (SHapley Additive exPlanations) to provide transparent risk factor analysis
- **Modern UI**: Clean, intuitive interface with real-time feedback
- **Decision Support**: Clear visualization of risk factors and their impact on the final decision

## Prerequisites

- Python 3.9.13 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fa-jaa/Loan-Approval-System.git
cd Loan-Approval-System
```

2. Create and activate a virtual environment:
```bash
python -m venv loanSystem
source loanSystem/bin/activate  # On Windows, use: loanSystem\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. The application provides two ways to analyze loan applications:
   - **Manual Entry**: Enter applicant details directly through the interface
   - **File Upload**: Upload a CSV file containing applicant data (not implemented)

4. The system will automatically:
   - Calculate risk scores
   - Generate SHAP explanations
   - Provide recommendations
   - Show decision thresholds

## Default Model and Dataset

The application comes with a pre-trained XGBoost model (`xgboost_no_protected.pkl`) and a processed dataset (`UnProtected_Processed.csv`) from the HMDA 2017 New York Data set. These are automatically loaded if no custom files are uploaded.

## Features Analyzed

The system analyzes various loan application factors including:
- Loan amount
- Applicant income
- Loan type
- Property type
- Lien status
- Owner occupancy
- Co-applicant status
- Loan-to-income ratio

## Outputs

For each analysis, the system provides:
1. **Risk Score**: A numerical score between 0-100
2. **Decision**: Approved, Needs Review, or Rejected
3. **Feature Importance**: Visual representation of factors affecting the decision
4. **Recommendations**: Actionable steps for account managers


## Technical Details

- Built with Streamlit for the web interface
- Uses SHAP for model interpretability
- Implements XGBoost for risk prediction
- Includes comprehensive error handling and data validation

## Acknowledgments

- SHAP library for model interpretability
- Streamlit for the web interface framework
- XGBoost for the machine learning model 
