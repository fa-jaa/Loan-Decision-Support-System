import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from shap.plots._style import style_context  # Import for SHAP styling
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import re
import io
import base64
from io import BytesIO
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
import random
import math
from PIL import Image
from streamlit.components.v1 import html  # Import for HTML component

# Set random seed for consistency
np.random.seed(42)
random.seed(42)

# Configure page
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
with open('static/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header with modern styling
st.markdown('<p class="main-header">Loan Approval System</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
This application helps account managers interpret loan approval model outputs using
Explainable AI (XAI). Upload model and dataset files, then enter applicant details to see risk assessment and 
actionable recommendations.
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'categorical_features' not in st.session_state:
    st.session_state.categorical_features = []
if 'numerical_features' not in st.session_state:
    st.session_state.numerical_features = []
if 'feature_groups' not in st.session_state:
    st.session_state.feature_groups = {}
if 'standalone_features' not in st.session_state:
    st.session_state.standalone_features = []
if 'expected_model_features' not in st.session_state:
    st.session_state.expected_model_features = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None
if 'threshold_min' not in st.session_state:
    st.session_state.threshold_min = 50  # Default min threshold
if 'threshold_max' not in st.session_state:
    st.session_state.threshold_max = 65  # Default max threshold

# Feature metadata with descriptions and impact direction
feature_metadata = {
    'loan_amount_k': {
        'description': 'Loan amount in thousands of dollars',
        'impact': 'Higher loan amounts may increase default risk',
        'direction': 'negative'
    },
    'applicant_income_k': {
        'description': 'Applicant income in thousands of dollars per year',
        'impact': 'Higher income generally reduces default risk',
        'direction': 'positive'
    },
    'loan_type_Conventional': {
        'description': 'Loan is a conventional mortgage',
        'impact': 'Impact varies based on other factors',
        'direction': 'varies'
    },
    'loan_type_FHA-insured': {
        'description': 'Loan is insured by the Federal Housing Administration',
        'impact': 'May have different risk profile than conventional loans',
        'direction': 'varies'
    },
    'loan_type_FSA/RHS-guaranteed': {
        'description': 'Loan is guaranteed by Farm Service Agency or Rural Housing Service',
        'impact': 'May have different risk profile than conventional loans',
        'direction': 'varies'
    },
    'loan_type_VA-guaranteed': {
        'description': 'Loan is guaranteed by Department of Veterans Affairs',
        'impact': 'May have different risk profile than conventional loans',
        'direction': 'varies'
    },
    'purpose_Home improvement': {
        'description': 'Loan is for home improvement',
        'impact': 'Impact depends on property value changes',
        'direction': 'varies'
    },
    'purpose_Home purchase': {
        'description': 'Loan is for home purchase',
        'impact': 'Generally lower risk than other purposes',
        'direction': 'positive'
    },
    'purpose_Refinancing': {
        'description': 'Loan is to refinance existing debt',
        'impact': 'Risk depends on terms of refinancing',
        'direction': 'varies'
    },
    'property_type_Manufactured housing': {
        'description': 'Property is manufactured housing',
        'impact': 'May have different depreciation characteristics',
        'direction': 'varies'
    },
    'property_type_Multifamily dwelling': {
        'description': 'Property is multifamily dwelling',
        'impact': 'Risk profile depends on rental income potential',
        'direction': 'varies'
    },
    'property_type_One-to-four family dwelling (other than manufactured housing)': {
        'description': 'Property is a standard residential dwelling',
        'impact': 'Generally more stable in value than other property types',
        'direction': 'positive'
    },
    'lien_status_Not secured by a lien': {
        'description': 'Loan is not secured by a lien on a dwelling',
        'impact': 'Higher risk due to lack of collateral',
        'direction': 'negative'
    },
    'lien_status_Secured by a first lien': {
        'description': 'Loan is secured by a first lien on a dwelling',
        'impact': 'Lower risk due to primary claim on collateral',
        'direction': 'positive'
    },
    'lien_status_Secured by a subordinate lien': {
        'description': 'Loan is secured by a subordinate lien on a dwelling',
        'impact': 'Higher risk than first liens but lower than unsecured',
        'direction': 'varies'
    },
    'owner_occupancy_Not applicable': {
        'description': 'Owner occupancy status is not applicable',
        'impact': 'Impact varies based on specifics',
        'direction': 'varies'
    },
    'owner_occupancy_Not owner-occupied as a principal dwelling': {
        'description': 'Property is not owner-occupied',
        'impact': 'Higher risk than owner-occupied properties',
        'direction': 'negative'
    },
    'owner_occupancy_Owner-occupied as a principal dwelling': {
        'description': 'Property is owner-occupied',
        'impact': 'Lower risk than non-owner-occupied properties',
        'direction': 'positive'
    },
    'co_applicant_status': {
        'description': 'Whether there is a co-applicant on the loan',
        'impact': 'Having a co-applicant generally reduces risk',
        'direction': 'positive'
    },
    'loan_to_income_ratio': {
        'description': 'Ratio of loan amount to annual income',
        'impact': 'Higher ratios increase default risk',
        'direction': 'negative'
    }
}

# Add recommendation metadata for account managers
recommendation_metadata = {
    'loan_amount_k': {
        'category': 'modifiable',
        'recommendation': 'The loan amount significantly impacts the risk assessment. Suggest a lower amount to improve debt-to-income ratio.',
        'regulation': 'Per Regulation B (ECOA), ensure consistency in recommendations across similar applicant profiles.'
    },
    'loan_to_income_ratio': {
        'category': 'modifiable',
        'recommendation': 'This DTI ratio exceeds preferred parameters. Account managers should explore options for improving this metric with the client.',
        'regulation': 'Aligns with Regulation Z (TILA) ability-to-repay requirements.'
    },
    'purpose_Home improvement': {
        'category': 'modifiable',
        'recommendation': 'Home improvement loans show higher risk profiles than purchase loans in our model. Consider alternative structuring if appropriate.',
        'regulation': 'Document purpose clearly for Regulation Z (TILA) and HMDA reporting requirements.'
    },
    'purpose_Refinancing': {
        'category': 'modifiable',
        'recommendation': 'Refinancing purpose creates additional risk factors. Verify the benefit to borrower meets internal guidelines.',
        'regulation': 'Ensure compliance with Regulation Z (TILA) requirements for refinance transactions.'
    },
    'lien_status_Not secured by a lien': {
        'category': 'potentially_modifiable',
        'recommendation': 'Unsecured status significantly increases risk. Explore collateral options with the applicant if possible.',
        'regulation': 'Update HMDA reporting fields appropriately if lien status changes.'
    },
    'lien_status_Secured by a subordinate lien': {
        'category': 'potentially_modifiable',
        'recommendation': 'Subordinate lien position elevates risk profile. Consider restructuring to achieve first-lien position.',
        'regulation': 'Verify appropriate disclosures for subordinate financing under Regulation Z.'
    },
    'loan_type_FHA-insured': {
        'category': 'potentially_modifiable',
        'recommendation': 'Risk model indicates conventional product may be more appropriate. Analyze qualification for conventional financing.',
        'regulation': 'Ensure compliance with FHA guidelines and fair lending considerations before recommending product changes.'
    },
    'loan_type_FSA/RHS-guaranteed': {
        'category': 'potentially_modifiable',
        'recommendation': 'Risk assessment suggests reviewing conventional loan options. Evaluate if borrower qualifies for alternative products.',
        'regulation': 'Follow RHS/FSA specific guidelines and disclosure requirements.'
    },
    'loan_type_VA-guaranteed': {
        'category': 'potentially_modifiable',
        'recommendation': 'Consider evaluating conventional options alongside VA product. Compare total cost scenarios for borrower.',
        'regulation': 'Remember VA loans have specific requirements for qualified veterans that must be verified.'
    },
    'owner_occupancy_Not applicable': {
        'category': 'potentially_modifiable',
        'recommendation': 'Owner-occupancy status is unclear. Verify intended occupancy and update application accordingly.',
        'regulation': 'Accurate occupancy status is required for HMDA reporting and affects product eligibility.'
    },
    'owner_occupancy_Not owner-occupied as a principal dwelling': {
        'category': 'potentially_modifiable',
        'recommendation': 'Non-owner occupied status increases risk profile. Review investment property guidelines and pricing.',
        'regulation': 'Non-owner occupied properties have different regulatory requirements and LTV limits.'
    },
    'co_applicant_status': {
        'category': 'fixed',
        'recommendation': 'No co-applicant is present. If appropriate, discuss potential addition of creditworthy co-applicant with the borrower.',
        'regulation': 'Any co-applicant must be properly disclosed on all loan documents per Regulation B.'
    },
    'property_type_Manufactured housing': {
        'category': 'fixed',
        'recommendation': 'Manufactured housing presents elevated risk factors. Ensure property meets all agency guidelines for this property type.',
        'regulation': 'Manufactured housing has specific HMDA reporting requirements and compliance considerations.'
    },
    'property_type_Multifamily dwelling': {
        'category': 'fixed',
        'recommendation': 'Multifamily property has specific risk characteristics. Review rental income documentation and property condition.',
        'regulation': 'Multiple-unit properties have different regulatory treatment under various lending regulations.'
    },
    'applicant_income_k': {
        'category': 'modifiable',
        'recommendation': 'Validate all income sources are properly documented and calculated. Explore additional eligible income if appropriate.',
        'regulation': 'Income verification must follow Regulation B and fair lending guidelines consistently.'
    }
}

# Sidebar for file uploads with modern styling
with st.sidebar:
    st.header("📁 Upload Files")
    
    # Model upload with better user feedback
    with st.container():
        model_file = st.file_uploader("Upload Model (pickle file)", type=["pkl", "pickle"])
        
        if model_file is not None:
            try:
                st.session_state.model = pickle.load(model_file)
                st.success("✅ Model loaded successfully!")
                
                # Try to extract expected feature names from model
                try:
                    if hasattr(st.session_state.model, 'feature_names_in_'):
                        st.session_state.expected_model_features = list(st.session_state.model.feature_names_in_)
                        st.write(f"Model expects {len(st.session_state.expected_model_features)} features")
                    elif hasattr(st.session_state.model, 'feature_names'):
                        st.session_state.expected_model_features = list(st.session_state.model.feature_names)
                        st.write(f"Model expects {len(st.session_state.expected_model_features)} features")
                    elif hasattr(st.session_state.model, 'get_booster') and hasattr(st.session_state.model.get_booster(), 'feature_names'):
                        st.session_state.expected_model_features = list(st.session_state.model.get_booster().feature_names)
                        st.write(f"Model expects {len(st.session_state.expected_model_features)} features")
                except Exception as feat_err:
                    st.warning(f"Could not determine expected features from model: {str(feat_err)}")
                    st.warning("You'll need to make sure your input features match exactly what the model expects")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
        else:
            # Use default model if none uploaded
            try:
                with open('xgboost_no_protected.pkl', 'rb') as f:
                    st.session_state.model = pickle.load(f)
                st.success("✅ Default model loaded successfully!")
                
                # Try to extract expected feature names from model
                try:
                    if hasattr(st.session_state.model, 'feature_names_in_'):
                        st.session_state.expected_model_features = list(st.session_state.model.feature_names_in_)
                        st.write(f"Model expects {len(st.session_state.expected_model_features)} features")
                    elif hasattr(st.session_state.model, 'feature_names'):
                        st.session_state.expected_model_features = list(st.session_state.model.feature_names)
                        st.write(f"Model expects {len(st.session_state.expected_model_features)} features")
                    elif hasattr(st.session_state.model, 'get_booster') and hasattr(st.session_state.model.get_booster(), 'feature_names'):
                        st.session_state.expected_model_features = list(st.session_state.model.get_booster().feature_names)
                        st.write(f"Model expects {len(st.session_state.expected_model_features)} features")
                except Exception as feat_err:
                    st.warning(f"Could not determine expected features from model: {str(feat_err)}")
                    st.warning("You'll need to make sure your input features match exactly what the model expects")
            except Exception as e:
                st.error(f"❌ Error loading default model: {str(e)}")
    
    # Dataset upload with modern styling
    with st.container():
        dataset_file = st.file_uploader("Upload Dataset (CSV file)", type=["csv"])
        
        if dataset_file is not None:
            try:
                st.session_state.dataset = pd.read_csv(dataset_file)
                st.success(f"✅ Dataset loaded: {st.session_state.dataset.shape[0]} rows, {st.session_state.dataset.shape[1]} columns")
                
                # Target column selection with better default detection
                target_options = st.session_state.dataset.columns.tolist()
                default_target_idx = target_options.index('loan_status') if 'loan_status' in target_options else len(target_options)-1
                
                target_col = st.selectbox(
                    "Select target column",
                    options=target_options,
                    index=default_target_idx,
                    help="This column will be excluded from features"
                )
                
                # Set features excluding target
                st.session_state.features = [col for col in st.session_state.dataset.columns if col != target_col]
                
                # Create SHAP explainer if model is also loaded
                if st.session_state.model is not None:
                    with st.spinner("Creating SHAP explainer..."):
                        # Sample background data for SHAP
                        background_data = st.session_state.dataset[st.session_state.features].sample(
                            min(100, len(st.session_state.dataset))
                        )
                        
                        try:
                            # Ensure background data only includes expected model features
                            if st.session_state.expected_model_features:
                                missing_cols = [col for col in st.session_state.expected_model_features if col not in background_data.columns]
                                extra_cols = [col for col in background_data.columns if col not in st.session_state.expected_model_features]
                            
                                # Add missing columns with default value 0
                                for col in missing_cols:
                                    background_data[col] = 0
                            
                                # Remove extra columns
                                background_data = background_data.drop(columns=extra_cols, errors='ignore')
                            
                                # Ensure column order matches model expectations
                                background_data = background_data[st.session_state.expected_model_features]
                            
                            # Try tree explainer first (works for tree-based models)
                            if hasattr(st.session_state.model, 'feature_importances_') or type(st.session_state.model).__name__ in [
                                'XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor', 
                                'RandomForestClassifier', 'RandomForestRegressor', 'GradientBoostingClassifier'
                            ]:
                                st.session_state.explainer = shap.TreeExplainer(st.session_state.model, background_data)
                            else:
                                # For non-tree models, use Kernel explainer
                                if hasattr(st.session_state.model, 'predict_proba'):
                                    predict_fn = lambda x: st.session_state.model.predict_proba(x)[:, 1]
                                else:
                                    predict_fn = lambda x: st.session_state.model.predict(x)
                                st.session_state.explainer = shap.KernelExplainer(predict_fn, background_data)
                            
                            st.success("✅ SHAP explainer created successfully!")
                        except Exception as e:
                            st.error(f"❌ Error creating SHAP explainer: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
        else:
            # Use default dataset if none uploaded
            try:
                st.session_state.dataset = pd.read_csv('UnProtected_Processed.csv')
                st.success(f"✅ Default dataset loaded: {st.session_state.dataset.shape[0]} rows, {st.session_state.dataset.shape[1]} columns")
                
                # Target column selection with better default detection
                target_options = st.session_state.dataset.columns.tolist()
                default_target_idx = target_options.index('loan_status') if 'loan_status' in target_options else len(target_options)-1
                
                target_col = st.selectbox(
                    "Select target column",
                    options=target_options,
                    index=default_target_idx,
                    help="This column will be excluded from features"
                )
                
                # Set features excluding target
                st.session_state.features = [col for col in st.session_state.dataset.columns if col != target_col]
                
                # Create SHAP explainer if model is also loaded
                if st.session_state.model is not None:
                    with st.spinner("Creating SHAP explainer..."):
                        # Sample background data for SHAP
                        background_data = st.session_state.dataset[st.session_state.features].sample(
                            min(100, len(st.session_state.dataset))
                        )
                        
                        try:
                            # Ensure background data only includes expected model features
                            if st.session_state.expected_model_features:
                                missing_cols = [col for col in st.session_state.expected_model_features if col not in background_data.columns]
                                extra_cols = [col for col in background_data.columns if col not in st.session_state.expected_model_features]
                            
                                # Add missing columns with default value 0
                                for col in missing_cols:
                                    background_data[col] = 0
                            
                                # Remove extra columns
                                background_data = background_data.drop(columns=extra_cols, errors='ignore')
                            
                                # Ensure column order matches model expectations
                                background_data = background_data[st.session_state.expected_model_features]
                            
                            # Try tree explainer first (works for tree-based models)
                            if hasattr(st.session_state.model, 'feature_importances_') or type(st.session_state.model).__name__ in [
                                'XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor', 
                                'RandomForestClassifier', 'RandomForestRegressor', 'GradientBoostingClassifier'
                            ]:
                                st.session_state.explainer = shap.TreeExplainer(st.session_state.model, background_data)
                            else:
                                # For non-tree models, use Kernel explainer
                                if hasattr(st.session_state.model, 'predict_proba'):
                                    predict_fn = lambda x: st.session_state.model.predict_proba(x)[:, 1]
                                else:
                                    predict_fn = lambda x: st.session_state.model.predict(x)
                                st.session_state.explainer = shap.KernelExplainer(predict_fn, background_data)
                            
                            st.success("✅ SHAP explainer created successfully!")
                        except Exception as e:
                            st.error(f"❌ Error creating SHAP explainer: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading default dataset: {str(e)}")
    
    # Add threshold configuration sliders
    st.markdown("---")
    st.subheader("Decision Thresholds")
    
    st.markdown("Configure score thresholds for decisions:")
    
    threshold_min = st.slider(
        "Minimum score for 'Needs Review'", 
        min_value=0, 
        max_value=100, 
        value=st.session_state.threshold_min,
        help="Applications below this score will be rejected"
    )
    
    threshold_max = st.slider(
        "Minimum score for 'Approved'", 
        min_value=0, 
        max_value=100, 
        value=st.session_state.threshold_max,
        help="Applications above this score will be automatically approved"
    )
    
    # Update session state with new threshold values
    st.session_state.threshold_min = threshold_min
    st.session_state.threshold_max = threshold_max
    
    # Display threshold logic
    st.markdown("""
    <div style="font-size: 0.85rem; margin-top: 10px;">
        <p><span style="color: #EF4444;">⚫</span> Score < {0}: <strong>Rejected</strong></p>
        <p><span style="color: #F59E0B;">⚫</span> Score {0}–{1}: <strong>Needs Review</strong></p>
        <p><span style="color: #10B981;">⚫</span> Score > {1}: <strong>Approved</strong></p>
    </div>
    """.format(threshold_min, threshold_max), unsafe_allow_html=True)
    
    # Enhanced sidebar info with better styling
    st.markdown("---")
    st.markdown("""
    ### Usage Guide:
    1. 📤 Upload trained model
    2. 📄 Upload dataset
    3. 📝 Enter client application details
    4. 🔍 Click "Analyze Application"
    5. 📊 Review risk assessment and recommendations
    """)

# Main content area with modern styling
if st.session_state.model is None or st.session_state.dataset is None:
    # Improved empty state
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background-color: #F8FAFC; border-radius: 12px; margin: 2rem 0;">
        <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c8.svg" width="80" style="margin-bottom: 1.5rem;">
        <h2 style="margin-bottom: 1rem; font-weight: 600; color: #1F2937;">Get Started</h2>
        <p style="font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem;">
            Please upload model and dataset files in the sidebar to begin analysis.
        </p>
        <p style="font-size: 0.9rem; color: #64748B;">
            👈 Use the sidebar on the left to upload your files
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Create input form based on dataset features
    st.markdown('<p class="sub-header">Enter Applicant Details</p>', unsafe_allow_html=True)
    
    # Display Global Feature Importance with Plotly for better formatting
    if st.session_state.explainer is not None:
        st.markdown('<div class="category-header">Global Feature Importance (Top 5)</div>', unsafe_allow_html=True)
        
        try:
            # Sample data from the dataset
            sample_data = st.session_state.dataset[st.session_state.features].sample(
                min(100, len(st.session_state.dataset))
            )
            
            # If we know what features the model expects, ensure sample data matches
            if st.session_state.expected_model_features:
                # Create a new sample_data with only the expected features
                missing_cols = [col for col in st.session_state.expected_model_features if col not in sample_data.columns]
                extra_cols = [col for col in sample_data.columns if col not in st.session_state.expected_model_features]
                
                # Add missing columns with default value 0
                for col in missing_cols:
                    sample_data[col] = 0
                
                # Remove extra columns
                sample_data = sample_data.drop(columns=extra_cols, errors='ignore')
                
                # Ensure column order matches exactly what model expects
                sample_data = sample_data[st.session_state.expected_model_features]
            
            # Calculate SHAP values for sample data
            background_shap_values = st.session_state.explainer.shap_values(sample_data)
            
            # Handle different formats based on model type
            if isinstance(background_shap_values, list):
                background_shap_values = background_shap_values[1] if len(background_shap_values) > 1 else background_shap_values[0]
            
            # Get feature importance values
            feature_importance = np.abs(background_shap_values).mean(0)
            feature_names = sample_data.columns
            
            # Create a DataFrame for sorted importance
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
            importance_df = importance_df.sort_values("Importance", ascending=False).head(5)
            
            # Create plotly horizontal bar chart instead of matplotlib
            fig = px.bar(
                importance_df,
                y="Feature",
                x="Importance",
                orientation='h',
                title="Top 5 Most Important Features",
                labels={"Importance": "Mean |SHAP Value|", "Feature": ""},
                color_discrete_sequence=["#1A73E8"],
                text="Importance"
            )
            
            # Customize layout
            fig.update_traces(
                texttemplate='%{text:.3f}', 
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=50, r=150, t=80, b=50),
                plot_bgcolor='rgba(250, 250, 250, 0.9)',
                xaxis=dict(
                    title=dict(
                        text="Mean |SHAP Value|",
                        font=dict(size=12)
                    )
                ),
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not generate global feature importance: {e}")
    
    # Create container for all inputs with modern styling
    with st.form("application_form"):
        # Initialize dict to store all inputs
        input_data = {}
        
        # Add tabs for input methods
        input_tabs = st.tabs(["Manual Entry", "File Upload"])
        
        with input_tabs[0]:  # Manual Entry tab
            # Create a more compact layout with all inputs together
            # Use a 3-column layout for more compact presentation
            col1, col2, col3 = st.columns(3)
            
            # Loan Amount
            loan_amount_field = 'loan_amount_k'  # Expected field name
            income_field = 'applicant_income_k'  # Expected field name
            
            # Check if fields exist in dataset
            loan_amount_exists = loan_amount_field in st.session_state.features
            income_exists = income_field in st.session_state.features
            
            # Add loan amount input with compact styling
            if loan_amount_exists:
                min_val = float(st.session_state.dataset[loan_amount_field].min())
                max_val = float(st.session_state.dataset[loan_amount_field].max())
                mean_val = float(st.session_state.dataset[loan_amount_field].mean())
                
                loan_amount = col1.number_input(
                    "Loan Amount (thousands)",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=1.0,
                    format="%.1f"
                )
                input_data[loan_amount_field] = loan_amount
            else:
                st.error(f"Expected field '{loan_amount_field}' not found in dataset!")
                loan_amount = 100.0  # Default value
            
            # Add income input with compact styling
            if income_exists:
                min_val = float(st.session_state.dataset[income_field].min())
                max_val = float(st.session_state.dataset[income_field].max())
                mean_val = float(st.session_state.dataset[income_field].mean())
                
                income = col2.number_input(
                    "Applicant Income (thousands)",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=1.0,
                    format="%.1f"
                )
                input_data[income_field] = income
            else:
                st.error(f"Expected field '{income_field}' not found in dataset!")
                income = 50.0  # Default value
                
            # Loan Type with compact select box - put in column 3
            loan_type_options = ["Conventional", "FHA-insured", "FSA/RHS-guaranteed", "VA-guaranteed"]
            loan_type = col3.selectbox(
                "Loan Type",
                options=loan_type_options,
                index=0
            )
            
            # Reset all loan type features to 0
            for opt in loan_type_options:
                feature_name = f"loan_type_{opt}"
                input_data[feature_name] = 0
            
            # Set selected loan type to 1
            input_data[f"loan_type_{loan_type}"] = 1
            
            # Calculate loan-to-income ratio with compact display
            if loan_amount_exists and income_exists and income > 0:
                ratio = loan_amount / income
                input_data['loan_to_income_ratio'] = ratio
                
                # Display the ratio with color coding
                ratio_color = "#10B981" if ratio < 2 else "#F59E0B" if ratio < 4 else "#EF4444"
                threshold_text = "Within Policy" if ratio < 2 else "Borderline" if ratio < 4 else "Exceeds Guidelines"
                
                col1.markdown(f"""
                <div style="margin-top: 5px; font-size: 0.85rem;">
                    <strong>Loan-to-Income Ratio:</strong> <span style="color: {ratio_color};">{ratio:.2f}</span>
                    <span style="font-size: 0.8rem; color: {ratio_color};">({threshold_text})</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Second row of inputs
            col4, col5, col6 = st.columns(3)
                    
            # Loan Purpose with compact styling
            purpose_options = ["Home improvement", "Home purchase", "Refinancing"]
            purpose = col4.selectbox(
                "Loan Purpose",
                options=purpose_options,
                index=0
            )
            
            # Reset all purpose features to 0
            for opt in purpose_options:
                feature_name = f"purpose_{opt}"
                input_data[feature_name] = 0
            
            # Set selected purpose to 1
            input_data[f"purpose_{purpose}"] = 1
            
            # Property Type with compact UI
            property_options = [
                "Manufactured housing", 
                "Multifamily dwelling", 
                "One-to-four family dwelling (other than manufactured housing)"
            ]
            property_type = col5.selectbox(
                "Property Type",
                options=property_options,
                index=2
            )
            
            # Reset all property type features to 0
            for opt in property_options:
                feature_name = f"property_type_{opt}"
                input_data[feature_name] = 0
            
            # Set selected property type to 1
            input_data[f"property_type_{property_type}"] = 1
            
            # Lien Status with compact styling
            lien_options = [
                "Not secured by a lien",
                "Secured by a first lien",
                "Secured by a subordinate lien"
            ]
            lien_status = col6.selectbox(
                "Lien Status",
                options=lien_options,
                index=1
            )
            
            # Reset all lien status features to 0
            for opt in lien_options:
                feature_name = f"lien_status_{opt}"
                input_data[feature_name] = 0
            
            # Set selected lien status to 1
            input_data[f"lien_status_{lien_status}"] = 1
            
            # Third row for remaining inputs
            col7, col8 = st.columns(2)
            
            # Owner Occupancy with compact styling
            occupancy_options = [
                "Not applicable",
                "Not owner-occupied as a principal dwelling",
                "Owner-occupied as a principal dwelling"
            ]
            occupancy = col7.selectbox(
                "Owner Occupancy",
                options=occupancy_options,
                index=2
            )
            
            # Reset all occupancy features to 0
            for opt in occupancy_options:
                feature_name = f"owner_occupancy_{opt}"
                input_data[feature_name] = 0
            
            # Set selected occupancy to 1
            input_data[f"owner_occupancy_{occupancy}"] = 1
            
            # Co-applicant Status with compact styling
            co_applicant_options = ["Yes", "No"]
            co_applicant = col8.selectbox(
                "Co-applicant",
                options=co_applicant_options,
                index=1
            )
            input_data['co_applicant_status'] = 1 if co_applicant == "Yes" else 0

        with input_tabs[1]:  # File Upload tab
            st.markdown("""
            <div style="padding: 15px; background-color: #F8FAFC; border-radius: 8px; margin-bottom: 15px;">
                <h3 style="font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;">Upload Applicant Data</h3>
                <p style="font-size: 0.9rem; color: #475569;">
                    Upload a CSV or Excel file with client application data. The file should contain the same columns 
                    as the training dataset used to develop the model.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "pdf"])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith(".pdf"):
                        st.warning("PDF parsing would require additional processing")
                        df = None
                    
                    if df is not None:
                        st.success(f"File uploaded successfully! Found {len(df)} records.")
                        st.dataframe(df.head(3))
                        
                        # If multiple records, ask which one to use
                        if len(df) > 1:
                            record_idx = st.selectbox(
                                "Select record to analyze:",
                                options=range(len(df)),
                                format_func=lambda x: f"Record {x+1}"
                            )
                            selected_record = df.iloc[record_idx]
                        else:
                            selected_record = df.iloc[0]
                        
                        # Convert selected record to input_data format
                        for col in selected_record.index:
                            input_data[col] = selected_record[col]
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            else:
                st.info("No file uploaded. Please upload a file or use manual entry.")
        
        # Enhanced submit button
        submitted = st.form_submit_button("Analyze Application", use_container_width=True)
        
        if submitted:
            st.session_state.input_data = input_data
    
    # Analyze submission with improved UI
    if 'input_data' in st.session_state:
        with st.spinner("Analyzing application..."):
            # Get input data from session state
            input_data = st.session_state.input_data
            
            # Calculate monthly income for display purposes
            if 'applicant_income_k' in input_data:
                monthly_income = input_data['applicant_income_k'] * 1000 / 12
                input_data['monthly_income'] = monthly_income
            
            # Create input dataframe with all required features
            input_df = pd.DataFrame([input_data])
            
            # Remove any non-model features
            model_input_df = input_df.copy()
            
            if 'monthly_income' in model_input_df:
                model_input_df = model_input_df.drop(columns=['monthly_income'])
            
            # If we know what features the model expects, make sure our input matches
            if st.session_state.expected_model_features:
                # Check for missing features
                missing_features = [f for f in st.session_state.expected_model_features if f not in model_input_df.columns]
                if missing_features:
                    for feature in missing_features:
                        model_input_df[feature] = 0  # Set missing features to 0
                
                # Ensure column order matches exactly what model expects
                model_input_df = model_input_df[st.session_state.expected_model_features]
            
            # Make prediction
            try:
                if hasattr(st.session_state.model, 'predict_proba'):
                    proba = st.session_state.model.predict_proba(model_input_df)[:, 1][0]
                    raw_prediction = proba
                else:
                    raw_prediction = st.session_state.model.predict(model_input_df)[0]
                    
                # Convert prediction to a 0-100 score for easier interpretation
                if 0 <= raw_prediction <= 1:
                    prediction_score = int(raw_prediction * 100)
                else:
                    # If prediction is not already between 0-1, apply sigmoid transform
                    prediction_score = int(100 / (1 + math.exp(-raw_prediction)))
                
                # Determine decision based on thresholds
                if prediction_score > st.session_state.threshold_max:
                    decision = "APPROVED"
                elif prediction_score >= st.session_state.threshold_min:
                    decision = "NEEDS REVIEW"
                else:
                    decision = "REJECTED"
                
                st.session_state.prediction = prediction_score
                st.session_state.decision = decision
                
            except Exception as pred_error:
                st.error(f"Prediction error: {str(pred_error)}")
                st.error("Feature mismatch. Make sure your input features match what the model expects.")
                if st.session_state.expected_model_features:
                    st.write("Model expected features:", st.session_state.expected_model_features)
                st.write("Input features:", list(model_input_df.columns))
                st.stop()
            
            # Calculate SHAP values for this instance
            try:
                shap_values = st.session_state.explainer.shap_values(model_input_df)
                
                # Handle different formats of SHAP values based on model type
                if isinstance(shap_values, list):
                    # For classification models, use class 1 (positive class)
                    instance_shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    expected_value = st.session_state.explainer.expected_value[1] if isinstance(st.session_state.explainer.expected_value, list) else st.session_state.explainer.expected_value
                else:
                    instance_shap_values = shap_values
                    expected_value = st.session_state.explainer.expected_value
            except Exception as shap_error:
                st.error(f"Error calculating SHAP values: {str(shap_error)}")
                st.error("This might be due to a feature mismatch. Check that your input features match what the model expects.")
                st.stop()
            
            # Store explanation data
            st.session_state.explanation = {
                'shap_values': instance_shap_values[0],
                'expected_value': expected_value,
                'features': model_input_df.columns.tolist(),
                'feature_values': model_input_df.iloc[0].to_dict(),
                'raw_prediction': raw_prediction
            }
        
        # Create container for results with modern styling
        st.markdown('<p class="sub-header">Loan Risk Assessment</p>', unsafe_allow_html=True)
        
        # ----- BEGIN DECISION DISPLAY COMPONENT -----
        # Modern Decision Display with unified component
        decision = st.session_state.decision
        score = st.session_state.prediction
        threshold_min = st.session_state.threshold_min
        threshold_max = st.session_state.threshold_max

        # Configure based on decision
        if decision == "APPROVED":
            header_class = "decision-approved-header"
            icon = "✓"
            message = "This application meets approval criteria."
        elif decision == "NEEDS REVIEW":
            header_class = "decision-review-header"
            icon = "⚠"
            message = "This application requires account manager review. See recommended actions below."
        else:  # REJECTED
            header_class = "decision-rejected-header"
            icon = "✕"
            message = "This application falls outside standard risk parameters. Exceptions require senior approval."

        # Calculate progress bar sections
        red_width = threshold_min
        yellow_width = threshold_max - threshold_min
        green_width = 100 - threshold_max

        # Fix the indicator position to be a percentage between 0-100
        # Ensure it's clamped within the valid range
        indicator_position = max(0, min(100, score))

        # Create the modern decision display as a Streamlit component
        st.container()  # This creates a clean break for the component

        # Use st.components.html to render pure HTML without issues
        decision_html = f"""
        <style>
            .decision-container {{
                border-radius: 0.75rem;
                overflow: hidden;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                margin-bottom: 1.5rem;
                border: 1px solid rgba(0, 0, 0, 0.05);
                font-family: 'Inter', -apple-system, sans-serif;
            }}
            
            .decision-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1.5rem;
                color: white;
            }}
            
            .decision-approved-header {{
                background: linear-gradient(135deg, #10B981, #059669);
            }}
            
            .decision-review-header {{
                background: linear-gradient(135deg, #F59E0B, #D97706);
            }}
            
            .decision-rejected-header {{
                background: linear-gradient(135deg, #EF4444, #DC2626);
            }}
            
            .decision-icon {{
                width: 3.5rem;
                height: 3.5rem;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 9999px;
                background-color: rgba(255, 255, 255, 0.2);
                font-size: 1.75rem;
                font-weight: bold;
                margin-right: 1.25rem;
            }}
            
            .decision-title {{
                font-size: 1.75rem;
                font-weight: 700;
                margin: 0;
                letter-spacing: -0.02em;
            }}
            
            .decision-subtitle {{
                font-size: 0.95rem;
                opacity: 0.9;
                margin: 0.25rem 0 0 0;
            }}
            
            .score-box {{
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 0.75rem;
                padding: 0.75rem 1.75rem;
                text-align: center;
            }}
            
            .score-value {{
                font-size: 2.5rem;
                font-weight: 700;
                line-height: 1;
                margin: 0;
            }}
            
            .score-label {{
                font-size: 0.8rem;
                opacity: 0.9;
                text-transform: uppercase;
                margin: 0.25rem 0 0 0;
                letter-spacing: 0.05em;
            }}
            
            .decision-body {{
                background-color: white;
                padding: 1.5rem;
            }}
            
            .threshold-labels {{
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                color: #4B5563;
                margin-bottom: 0.375rem;
            }}
            
            .progress-container {{
                height: 0.625rem;
                background-color: #F3F4F6;
                border-radius: 9999px;
                overflow: hidden;
                position: relative;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
            }}
            
            .progress-section {{
                height: 100%;
                float: left;
            }}
            
            .progress-red {{
                background: linear-gradient(90deg, #FCA5A5, #EF4444);
            }}
            
            .progress-yellow {{
                background: linear-gradient(90deg, #FCD34D, #F59E0B);
            }}
            
            .progress-green {{
                background: linear-gradient(90deg, #6EE7B7, #10B981);
            }}
            
            .progress-indicator {{
                position: absolute;
                height: 1.75rem;
                width: 0.5rem;
                background-color: #1F2937;
                border-radius: 9999px;
                top: -0.625rem;
                transform: translateX(-50%);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            
            .threshold-categories {{
                display: flex;
                justify-content: space-between;
                font-size: 0.85rem;
                font-weight: 500;
                margin-top: 0.5rem;
            }}
            
            .category-red {{
                color: #DC2626;
            }}
            
            .category-yellow {{
                color: #D97706;
            }}
            
            .category-green {{
                color: #059669;
            }}
            
            .decision-message {{
                margin-top: 1.25rem;
                font-size: 0.95rem;
                color: #4B5563;
                padding: 1rem;
                background-color: #F9FAFB;
                border-radius: 0.5rem;
                border-left: 4px solid #D1D5DB;
            }}
        </style>

        <div class="decision-container">
            <div class="decision-header {header_class}">
                <div style="display: flex; align-items: center;">
                    <div class="decision-icon">{icon}</div>
                    <div>
                        <p class="decision-title">{decision}</p>
                        <p class="decision-subtitle">Decision</p>
                    </div>
                </div>
                <div class="score-box">
                    <p class="score-value">{score}</p>
                    <p class="score-label">RISK SCORE</p>
                </div>
            </div>
            
            <div class="decision-body">
                <div class="threshold-labels">
                    <span>0</span>
                    <span>{threshold_min}</span>
                    <span>{threshold_max}</span>
                    <span>100</span>
                </div>
                
                <div class="progress-container">
                    <div class="progress-section progress-red" style="width: {red_width}%;"></div>
                    <div class="progress-section progress-yellow" style="width: {yellow_width}%;"></div>
                    <div class="progress-section progress-green" style="width: {green_width}%;"></div>
                    <div class="progress-indicator" style="left: {indicator_position}%;"></div>
                </div>
                
                <div class="threshold-categories">
                    <span class="category-red">Rejected</span>
                    <span class="category-yellow">Needs Review</span>
                    <span class="category-green">Approved</span>
                </div>
                
                <div class="decision-message">
                    <p>{message}</p>
                </div>
            </div>
        </div>
        """

        # Use HTML component to display the decision component
        html(decision_html, height=350)
        # ----- END DECISION DISPLAY COMPONENT -----
        
        # ----- MODERN TABS FOR RESULTS ANALYSIS -----
        # Replace the old tab structure with new tabs
        tab1, tab2 = st.tabs(["Impact Analysis", "Recommendations"])
        
        with tab1:
            # Create two-column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # ----- BEGIN SHAP WATERFALL PLOT -----
                # Interactive SHAP Waterfall Plot - Explainability core
                st.subheader("Risk Factors")

                # Get the SHAP values and feature data
                features = st.session_state.explanation['features']
                shap_values = st.session_state.explanation['shap_values']
                expected_value = st.session_state.explanation['expected_value']
                feature_values = st.session_state.explanation['feature_values']

                # Create DataFrame for sorting
                shap_df = pd.DataFrame({
                    'Feature': features,
                    'SHAP Value': shap_values,
                    'Abs Value': abs(shap_values),
                    'Feature Value': [feature_values.get(f, 'N/A') for f in features]
                })

                # Sort by absolute value and get top features
                shap_df = shap_df.sort_values('Abs Value', ascending=False)
                top_features = shap_df.head(8)  # Show top 8 features for better visualization

                # Create the data structure for the SHAP waterfall plot
                feature_names = list(top_features['Feature'])
                feature_values_list = list(top_features['SHAP Value'])

                # Create a buffer to save the plot to
                buf = io.BytesIO()

                # Generate the figure and plot using the corrected style_context syntax
                plt.figure(figsize=(12, 8))  # Match the example's figure size

                # Apply style context with keyword arguments
                with style_context(primary_color_positive="green", primary_color_negative="red"):
                    # Create the waterfall plot using the legacy function with the example's pattern
                    shap.plots._waterfall.waterfall_legacy(
                        expected_value, 
                        np.array(feature_values_list),  # Convert to numpy array
                        feature_names=feature_names,
                        show=False,
                        max_display=7
                    )
                    
                    # Add a title
                    plt.title("SHAP Feature Contribution Analysis", fontsize=16, fontweight='bold', pad=20)
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save to buffer
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    plt.close()

                # Display the waterfall plot
                buf.seek(0)
                st.image(buf)
                # ----- END SHAP WATERFALL PLOT -----
            
            with col2:
                # Feature Importance Table
                st.subheader("Risk Weight Analysis")
                
                try:
                    # Create feature weight table
                    feature_weights = pd.DataFrame({
                        'Feature': features,
                        'SHAP Value': shap_values,
                        'Abs Impact': np.abs(shap_values),
                        'Feature Value': [st.session_state.explanation['feature_values'][f] for f in features]
                    })
                    
                    # Calculate weight percentage
                    total_impact = feature_weights['Abs Impact'].sum()
                    feature_weights['Weight %'] = (feature_weights['Abs Impact'] / total_impact * 100).round(1)
                    
                    # Sort by absolute impact
                    feature_weights = feature_weights.sort_values('Abs Impact', ascending=False)
                    
                    # Format feature values for display
                    def format_feature_value(row):
                        value = row['Feature Value']
                        feature = row['Feature']
                        
                        if feature in ['loan_amount_k', 'applicant_income_k']:
                            return f"{value:,.1f}k"
                        elif isinstance(value, (int, float)):
                            if value == 0 or value == 1:
                                return "Yes" if value == 1 else "No"
                            else:
                                return f"{value:.2f}"
                        else:
                            return str(value)
                    
                    formatted_values = feature_weights.apply(format_feature_value, axis=1)
                    
                    # Format a clean DataFrame for display with Streamlit
                    display_df = pd.DataFrame({
                        'Feature': feature_weights['Feature'],
                        'Weight %': feature_weights['Weight %'].apply(lambda x: f"{x}%"),
                        'Value': formatted_values,
                        'Impact': feature_weights['SHAP Value'].apply(lambda x: "+" if x > 0 else "-")
                    })
                    
                    # Display top 10 features using Streamlit's native table
                    st.dataframe(
                        display_df.head(10),
                        column_config={
                            "Feature": st.column_config.TextColumn("Factor"),
                            "Weight %": st.column_config.TextColumn("Weight %"),
                            "Value": st.column_config.TextColumn("Value"),
                            "Impact": st.column_config.TextColumn("Direction")
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=400
                    )
                    
                except Exception as e:
                    st.error(f"Error displaying feature importance: {e}")
                    st.markdown("""
                    <div style="background-color: #FEF2F2; padding: 16px; border-radius: 8px; border-left: 4px solid #EF4444;">
                        <p style="font-weight: 600; margin-bottom: 5px; color: #B91C1C;">Could not generate risk weight analysis</p>
                        <p style="font-size: 0.9rem; color: #7F1D1D;">Please check that the model provides valid SHAP values for this analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        with tab2:
            # ----- MODERN RECOMMENDATIONS UI FOR ACCOUNT MANAGERS -----
            st.markdown("""
            <style>
            .recommendation-header {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                color: #1F2937;
            }
            
            .recommendation-card {
                background-color: white;
                border-radius: 10px;
                padding: 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border-left: 4px solid #3B82F6;
                transition: transform 0.2s ease-in-out;
            }
            
            .recommendation-card:hover {
                transform: translateY(-2px);
            }
            
            .card-modifiable {
                border-left-color: #10B981;
            }
            
            .card-potentially-modifiable {
                border-left-color: #F59E0B;
            }
            
            .card-fixed {
                border-left-color: #6B7280;
            }
            
            .card-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #111827;
            }
            
            .card-info {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.75rem;
            }
            
            .info-label {
                font-size: 0.875rem;
                color: #6B7280;
            }
            
            .card-description {
                font-size: 0.95rem;
                color: #4B5563;
                margin-bottom: 0.75rem;
                line-height: 1.5;
            }
            
            .card-action {
                font-size: 0.95rem;
                font-weight: 500;
                color: #1F2937;
                padding-top: 0.75rem;
                border-top: 1px solid #E5E7EB;
            }
            
            .card-regulation {
                font-size: 0.85rem;
                font-style: italic;
                color: #64748B;
                padding-top: 0.75rem;
                margin-top: 0.5rem;
            }
            
            .impact-positive {
                color: #10B981;
                font-weight: 500;
            }
            
            .impact-negative {
                color: #EF4444;
                font-weight: 500;
            }
            
            .highlight {
                background: rgba(16, 185, 129, 0.1);
                padding: 2px 4px;
                border-radius: 4px;
                font-weight: 500;
            }
            
            .highlight-warning {
                background: rgba(245, 158, 11, 0.1);
                padding: 2px 4px;
                border-radius: 4px;
                font-weight: 500;
            }
            
            .category-pill {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-bottom: 1rem;
            }
            
            .category-modifiable {
                background-color: rgba(16, 185, 129, 0.1);
                color: #059669;
            }
            
            .category-potentially {
                background-color: rgba(245, 158, 11, 0.1);
                color: #D97706;
            }
            
            .category-fixed {
                background-color: rgba(107, 114, 128, 0.1);
                color: #4B5563;
            }
            
            .summary-box {
                background-color: #F9FAFB;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                border: 1px solid #E5E7EB;
            }
            
            .summary-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: #111827;
            }
            
            .summary-score {
                font-size: 2rem;
                font-weight: 700;
                color: #1F2937;
            }
            
            .key-factors {
                margin-top: 1rem;
            }
            
            .key-factor-item {
                display: flex;
                align-items: center;
                margin-bottom: 0.5rem;
            }
            
            .key-factor-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 0.5rem;
            }
            
            .key-factor-text {
                font-size: 0.925rem;
                color: #4B5563;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Decision Score Summary
            decision = st.session_state.decision
            score = st.session_state.prediction
            threshold_min = st.session_state.threshold_min
            threshold_max = st.session_state.threshold_max
            
            # Define score color based on decision
            if decision == "APPROVED":
                score_color = "#10B981"
                score_message = "This application meets our credit risk guidelines for approval."
            elif decision == "NEEDS REVIEW":
                score_color = "#F59E0B"
                score_message = "This application requires manual review due to identified risk factors."
            else:  # REJECTED
                score_color = "#EF4444"
                score_message = "This application presents significant risk factors requiring exception approval."
            
            # Create a summary box at the top
            st.markdown(f"""
            <div class="summary-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="summary-title">Risk Assessment Summary</div>
                        <div style="font-size: 0.95rem; color: #6B7280; margin-bottom: 0.5rem;">Underwriting result</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="summary-score" style="color: {score_color};">{score}</div>
                        <div style="font-size: 0.85rem; color: #6B7280;">out of 100</div>
                    </div>
                </div>
                <div style="font-size: 0.95rem; color: #4B5563; margin-top: 1rem;">
                    {score_message}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get feature details for recommendations
            feature_details = pd.DataFrame({
                'Feature': features,
                'SHAP Value': shap_values,
                'Feature Value': [st.session_state.explanation['feature_values'][f] for f in features]
            })
            
            # Sort by absolute impact
            feature_details['Abs Impact'] = feature_details['SHAP Value'].abs()
            feature_details = feature_details.sort_values('Abs Impact', ascending=False)
            
            # Separate positive and negative factors
            positive_factors = feature_details[feature_details['SHAP Value'] > 0]
            negative_factors = feature_details[feature_details['SHAP Value'] < 0]
            
            # Show main recommendation header
            st.markdown('<div class="recommendation-header">Action Recommendations</div>', unsafe_allow_html=True)
            
            if st.session_state.decision == "APPROVED":
                # If approved, show a success message with account manager guidelines
                st.markdown("""
                <div style="background-color: #F0FDF4; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="background-color: #10B981; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 13L9 17L19 7" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #10B981;">Approved for Processing</div>
                    </div>
                    <p style="font-size: 0.95rem; color: #065F46; margin-left: 44px; margin-bottom: 12px;">
                        This application meets all approval criteria. Account managers may proceed with standard processing.
                    </p>
                    <div style="background-color: white; border-radius: 8px; padding: 12px; margin-left: 44px;">
                        <div style="font-weight: 600; color: #065F46; margin-bottom: 8px;">Account Manager Actions:</div>
                        <ol style="margin-left: 16px; margin-bottom: 0; padding-left: 0; color: #065F46;">
                            <li style="margin-bottom: 4px;">Verify all supporting documentation is complete</li>
                            <li style="margin-bottom: 4px;">Proceed with standard closing procedures</li>
                            <li style="margin-bottom: 4px;">Document approval in LOS with reference to score</li>
                            <li>Schedule closing within standard timeline</li>
                        </ol>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # For non-approved applications, show recommendations based on score
                # Initialize recommendation categories
                modifiable_recs = []
                potentially_modifiable_recs = []
                fixed_recs = []
                
                # Get most impactful negative factors
                key_negatives = negative_factors.head(5)
                
                # Process each negative factor and categorize recommendations
                for _, row in key_negatives.iterrows():
                    feature = row['Feature']
                    impact = row['SHAP Value']
                    value = row['Feature Value']
                    
                    # Format value based on feature type
                    if feature in ['loan_amount_k', 'applicant_income_k']:
                        formatted_value = f"${value:,.1f}k"
                    elif isinstance(value, (int, float)):
                        if value == 0 or value == 1:
                            formatted_value = "Yes" if value == 1 else "No"
                        else:
                            formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    # If we have a specific recommendation for this feature, add it to the appropriate category
                    if feature in recommendation_metadata:
                        rec_info = recommendation_metadata[feature]
                        recommendation = rec_info['recommendation']
                        category = rec_info['category']
                        regulation = rec_info.get('regulation', '')
                        
                        # Add more realistic and detailed action steps based on feature for account managers
                        action_step = ""
                        if feature == 'loan_amount_k':
                            # Create a suggested target based on the current value
                            suggested_target = max(value * 0.8, 50)  # Suggest 20% reduction or minimum 50k
                            action_step = f"Discuss restructuring the loan amount from ${value:,.1f}k to approximately ${suggested_target:,.1f}k with the client. Document the conversation and any mitigating factors if the original amount is maintained."
                        elif feature == 'loan_to_income_ratio':
                            # Calculate a healthier ratio target
                            healthier_ratio = min(value * 0.8, 3.0)  # 20% lower or 3.0 max
                            action_step = f"Current DTI of {value:.2f} exceeds guidelines. Target {healthier_ratio:.2f} through loan amount reduction or verification of additional eligible income sources. Document compensating factors if DTI cannot be lowered."
                        elif 'purpose_' in feature:
                            action_step = "Review loan purpose with applicant. If purpose is correctly stated, document additional compensating factors that may offset this risk element."
                        elif 'lien_status_' in feature and 'first lien' not in feature:
                            action_step = "Assess feasibility of restructuring to a first-lien position. If not possible, document rationale for proceeding with subordinate lien and obtain necessary approvals."
                        elif 'owner_occupancy_' in feature and 'principal dwelling' not in feature:
                            action_step = "Verify occupancy status with the applicant. If non-owner occupied status is confirmed, ensure appropriate pricing and LTV adjustments are applied per policy."
                        elif feature == 'co_applicant_status' and value == 0:
                            action_step = "Discuss potential co-applicant options with the primary borrower if appropriate. Document conversation in client notes."
                        else:
                            action_step = "Review with the client and document any mitigating factors in the loan file. Obtain underwriter input if needed."
                        
                        rec_item = {
                            'feature': feature,
                            'impact': impact,
                            'value': formatted_value,
                            'recommendation': recommendation,
                            'action': action_step,
                            'regulation': regulation
                        }
                        
                        if category == 'modifiable':
                            modifiable_recs.append(rec_item)
                        elif category == 'potentially_modifiable':
                            potentially_modifiable_recs.append(rec_item)
                        elif category == 'fixed':
                            fixed_recs.append(rec_item)
                    else:
                        # Generate generic recommendation based on feature type for account managers
                        if feature == 'loan_amount_k':
                            recommendation = "Loan amount appears to be a risk factor"
                            action_step = f"Discuss loan amount reduction with client. Document consideration of this factor in your notes."
                            regulation = "Follow Ability-to-Repay (ATR) guidelines when adjusting loan amounts."
                            modifiable_recs.append({
                                'feature': feature,
                                'impact': impact,
                                'value': formatted_value,
                                'recommendation': recommendation,
                                'action': action_step,
                                'regulation': regulation
                            })
                        elif 'loan_to_income_ratio' in feature:
                            recommendation = "DTI ratio exceeds preferred parameters"
                            action_step = f"Review income documentation for additional eligible sources or discuss loan amount reduction options with client."
                            regulation = "Document ATR considerations per Regulation Z requirements."
                            modifiable_recs.append({
                                'feature': feature,
                                'impact': impact,
                                'value': formatted_value,
                                'recommendation': recommendation,
                                'action': action_step,
                                'regulation': regulation
                            })
                        elif 'lien_status' in feature:
                            recommendation = "Lien position impacts risk assessment"
                            action_step = "Evaluate options for restructuring to achieve first-lien position. Document rationale if proceeding with current lien status."
                            regulation = "Update HMDA lien status field appropriately if changes are made."
                            potentially_modifiable_recs.append({
                                'feature': feature,
                                'impact': impact,
                                'value': formatted_value,
                                'recommendation': recommendation,
                                'action': action_step,
                                'regulation': regulation
                            })
                        elif 'loan_type' in feature:
                            recommendation = "Loan product selection impacts risk profile"
                            action_step = "Review alternative loan products that may better align with client's profile. Document product selection rationale."
                            regulation = "Ensure compliance with all product-specific disclosure requirements."
                            potentially_modifiable_recs.append({
                                'feature': feature,
                                'impact': impact,
                                'value': formatted_value,
                                'recommendation': recommendation,
                                'action': action_step,
                                'regulation': regulation
                            })
                        elif 'owner_occupancy' in feature:
                            recommendation = "Occupancy status affects risk assessment"
                            action_step = "Verify occupancy intent with client and document in file. Ensure pricing reflects appropriate occupancy designation."
                            regulation = "Accurate occupancy reporting is required for HMDA and regulatory compliance."
                            potentially_modifiable_recs.append({
                                'feature': feature,
                                'impact': impact,
                                'value': formatted_value,
                                'recommendation': recommendation,
                                'action': action_step,
                                'regulation': regulation
                            })
                        else:
                            recommendation = f"This factor significantly impacts the risk assessment"
                            action_step = "Review with underwriting team. Document consideration of this factor in your notes."
                            regulation = "Follow standard documentation procedures."
                            fixed_recs.append({
                                'feature': feature,
                                'impact': impact,
                                'value': formatted_value,
                                'recommendation': recommendation,
                                'action': action_step,
                                'regulation': regulation
                            })
                
                # Display recommendations in a modern, card-based UI for account managers
                # Only show these detailed recommendations if score is below 70%
                if st.session_state.prediction < 70:
                    # Modifiable Factors (Most actionable)
                    if modifiable_recs:
                        st.markdown("""
                        <div class="category-pill category-modifiable">
                            Primary Action Items
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for rec in modifiable_recs:
                            feature_name = rec['feature']
                            # Clean up feature name for display
                            if '_' in feature_name:
                                display_name = feature_name.replace('_', ' ').title()
                                if display_name.startswith('Loan Type ') or display_name.startswith('Purpose ') or display_name.startswith('Property Type ') or display_name.startswith('Lien Status ') or display_name.startswith('Owner Occupancy '):
                                    display_name = display_name.split(' ', 2)[-1]
                            else:
                                display_name = feature_name.replace('_', ' ').title()
                            
                            st.markdown(f"""
                            <div class="recommendation-card card-modifiable">
                                <div class="card-title">{display_name}</div>
                                <div class="card-info">
                                    <span class="info-label">Current value: <strong>{rec['value']}</strong></span>
                                    <span class="impact-negative">Impact: {rec['impact']:.4f}</span>
                                </div>
                                <div class="card-description">
                                    {rec['recommendation']}
                                </div>
                                <div class="card-action">
                                    <strong>Account Manager Action:</strong> {rec['action']}
                                </div>
                                <div class="card-regulation">
                                    <strong>Compliance Note:</strong> {rec['regulation']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Potentially Modifiable Factors (Conditional)
                    if potentially_modifiable_recs:
                        st.markdown("""
                        <div class="category-pill category-potentially">
                            Secondary Considerations
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for rec in potentially_modifiable_recs:
                            feature_name = rec['feature']
                            # Clean up feature name for display
                            if '_' in feature_name:
                                display_name = feature_name.replace('_', ' ').title()
                                if display_name.startswith('Loan Type ') or display_name.startswith('Purpose ') or display_name.startswith('Property Type ') or display_name.startswith('Lien Status ') or display_name.startswith('Owner Occupancy '):
                                    display_name = display_name.split(' ', 2)[-1]
                            else:
                                display_name = feature_name.replace('_', ' ').title()
                            
                            st.markdown(f"""
                            <div class="recommendation-card card-potentially-modifiable">
                                <div class="card-title">{display_name}</div>
                                <div class="card-info">
                                    <span class="info-label">Current value: <strong>{rec['value']}</strong></span>
                                    <span class="impact-negative">Impact: {rec['impact']:.4f}</span>
                                </div>
                                <div class="card-description">
                                    {rec['recommendation']}
                                </div>
                                <div class="card-action">
                                    <strong>Account Manager Action:</strong> {rec['action']}
                                </div>
                                <div class="card-regulation">
                                    <strong>Compliance Note:</strong> {rec['regulation']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Fixed Factors (Informational only)
                    if fixed_recs:
                        st.markdown("""
                        <div class="category-pill category-fixed">
                            Documentation Requirements
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for rec in fixed_recs:
                            feature_name = rec['feature']
                            # Clean up feature name for display
                            if '_' in feature_name:
                                display_name = feature_name.replace('_', ' ').title()
                                if display_name.startswith('Loan Type ') or display_name.startswith('Purpose ') or display_name.startswith('Property Type ') or display_name.startswith('Lien Status ') or display_name.startswith('Owner Occupancy '):
                                    display_name = display_name.split(' ', 2)[-1]
                            else:
                                display_name = feature_name.replace('_', ' ').title()
                            
                            st.markdown(f"""
                            <div class="recommendation-card card-fixed">
                                <div class="card-title">{display_name}</div>
                                <div class="card-info">
                                    <span class="info-label">Current value: <strong>{rec['value']}</strong></span>
                                    <span class="impact-negative">Impact: {rec['impact']:.4f}</span>
                                </div>
                                <div class="card-description">
                                    {rec['recommendation']}
                                </div>
                                <div class="card-action">
                                    <strong>Documentation Requirement:</strong> {rec['action']}
                                </div>
                                <div class="card-regulation">
                                    <strong>Compliance Note:</strong> {rec['regulation']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Simpler recommendations for scores above 70% but not approved
                    st.markdown("""
                    <div style="background-color: #FFF7ED; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <div style="background-color: #F59E0B; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 9V13M12 17H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                            </div>
                            <div style="font-size: 1.1rem; font-weight: 600; color: #D97706;">Limited Review Required</div>
                        </div>
                        <p style="font-size: 0.95rem; color: #92400E; margin-left: 44px; margin-bottom: 15px;">
                            This application shows moderate risk factors requiring verification. Address the following items to complete processing:
                        </p>
                        
                        <div style="margin-left: 44px;">
                    """, unsafe_allow_html=True)
                    
                    # Display simplified cards for top 3 issues
                    for i, (_, row) in enumerate(key_negatives.head(3).iterrows()):
                        feature = row['Feature']
                        impact = row['SHAP Value']
                        value = row['Feature Value']
                        
                        # Format value
                        if feature in ['loan_amount_k', 'applicant_income_k']:
                            formatted_value = f"${value:,.1f}k"
                        elif isinstance(value, (int, float)):
                            if value == 0 or value == 1:
                                formatted_value = "Yes" if value == 1 else "No"
                            else:
                                formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = str(value)
                        
                        # Clean up feature name for display
                        if '_' in feature:
                            display_name = feature.replace('_', ' ').title()
                            if display_name.startswith('Loan Type ') or display_name.startswith('Purpose ') or display_name.startswith('Property Type ') or display_name.startswith('Lien Status ') or display_name.startswith('Owner Occupancy '):
                                display_name = display_name.split(' ', 2)[-1]
                        else:
                            display_name = feature.replace('_', ' ').title()
                        
                        # Generate action-oriented recommendations for account managers
                        if feature == 'loan_amount_k':
                            suggestion = f"Verify loan amount ({formatted_value}) is properly documented with clear rationale"
                        elif 'loan_to_income_ratio' in feature:
                            suggestion = f"Document income verification and DTI exception justification for {formatted_value} ratio"
                        elif 'lien_status' in feature:
                            suggestion = "Verify and document lien position status in LOS"
                        elif 'loan_type' in feature:
                            suggestion = "Confirm loan product eligibility requirements are fully met"
                        elif 'owner_occupancy' in feature:
                            suggestion = "Verify occupancy status and include occupancy certification in file"
                        else:
                            suggestion = f"Document verification of {display_name} in file notes"
                        
                        st.markdown(f"""
                        <div style="background-color: white; border-radius: 8px; padding: 12px; margin-bottom: 10px; border-left: 3px solid #F59E0B;">
                            <div style="font-weight: 500; color: #1F2937; margin-bottom: 4px;">{display_name}</div>
                            <div style="font-size: 0.9rem; color: #4B5563;">{suggestion}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add a conclusion box with next steps for account managers
                if st.session_state.decision == "NEEDS REVIEW":
                    st.markdown("""
                    <div style="background-color: #F9FAFB; padding: 20px; border-radius: 10px; margin-top: 30px; border: 1px solid #E5E7EB;">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;">
                            Processing Instructions
                        </div>
                        <p style="font-size: 0.95rem; color: #4B5563; margin-bottom: 15px;">
                            Follow these steps to address the identified risk factors:
                        </p>
                        <ol style="margin-left: 20px; color: #4B5563; font-size: 0.95rem;">
                            <li style="margin-bottom: 8px;">Contact the applicant to discuss possible adjustments to the application</li>
                            <li style="margin-bottom: 8px;">Document all conversations and decisions in the loan origination system</li>
                            <li style="margin-bottom: 8px;">Submit application to underwriting with your notes on mitigating factors</li>
                            <li style="margin-bottom: 8px;">Schedule follow-up with underwriter regarding decision timeline</li>
                        </ol>
                        <p style="font-size: 0.9rem; color: #6B7280; margin-top: 15px; font-style: italic;">
                            Note: Applications requiring review should be processed within 48 hours to maintain service level standards.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif st.session_state.decision == "REJECTED":
                    st.markdown("""
                    <div style="background-color: #F9FAFB; padding: 20px; border-radius: 10px; margin-top: 30px; border: 1px solid #E5E7EB;">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #1F2937; margin-bottom: 12px;">
                            Exception Process Instructions
                        </div>
                        <p style="font-size: 0.95rem; color: #4B5563; margin-bottom: 15px;">
                            This application has significant risk factors. Follow the exception process if proceeding:
                        </p>
                        <ol style="margin-left: 20px; color: #4B5563; font-size: 0.95rem;">
                            <li style="margin-bottom: 8px;">Complete Exception Request Form with detailed mitigating factors</li>
                            <li style="margin-bottom: 8px;">Obtain approval from senior underwriter or credit committee</li>
                            <li style="margin-bottom: 8px;">Document all compensating factors and approvals in LOS</li>
                            <li style="margin-bottom: 8px;">Consider alternative loan products that may better suit applicant</li>
                        </ol>
                        <div style="font-size: 0.9rem; color: #EF4444; margin-top: 15px; padding: 10px; background-color: #FEF2F2; border-radius: 6px;">
                            <strong>Important:</strong> Applications with scores below {threshold_min} require Senior Credit Officer approval per policy section 4.3.2.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add a small help section at the bottom with account manager focus
            st.markdown("""
            <div style="font-size: 0.85rem; color: #6B7280; margin-top: 30px; padding-top: 15px; border-top: 1px solid #E5E7EB;">
                <strong>About this Analysis:</strong> These recommendations are generated using our explainable AI system based on risk factors identified in the application. All actions should follow credit policy guidelines and fair lending requirements. For policy questions, contact the Credit Policy team.
            </div>
            """, unsafe_allow_html=True)
        
        # Footer and action buttons
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate detailed PDF report for account managers to save
            report_html = f"""
            
            <h2>Loan Risk Assessment Report</h2>
            <p><strong>Decision:</strong> {st.session_state.decision}</p>
            <p><strong>Score:</strong> {st.session_state.prediction}/100</p>
            <h3>Key Risk Factors:</h3>
            <ul>
            """
            
            # Add top positive and negative factors
            positive_factors = feature_details[feature_details['SHAP Value'] > 0].head(3)
            negative_factors = feature_details[feature_details['SHAP Value'] < 0].head(3)
            
            for _, row in positive_factors.iterrows():
                report_html += f"<li><strong>Positive:</strong> {row['Feature']} (+{row['SHAP Value']:.4f})</li>"
            
            for _, row in negative_factors.iterrows():
                report_html += f"<li><strong>Negative:</strong> {row['Feature']} ({row['SHAP Value']:.4f})</li>"
            
            report_html += "</ul>"
            
            # Convert to PDF report (in real implementation)
            # Here we'll create a download link for a text version
            report_bytes = report_html.encode()
            b64 = base64.b64encode(report_bytes).decode()
            
            download_link = f'<a href="data:text/html;base64,{b64}" download="Risk_Assessment_Report.html" class="download-button">Download Risk Assessment</a>'
            st.markdown(download_link, unsafe_allow_html=True)
        
        with col2:
            # Show different actions based on decision with account manager focus
            if st.session_state.decision == "NEEDS REVIEW":
                st.markdown('<a href="#" class="next-level-button">Submit to Underwriting</a>', unsafe_allow_html=True)
            elif st.session_state.decision == "APPROVED":
                st.markdown("""
                <div style="font-size: 0.9rem; color: #10B981; font-weight: 500; text-align: center; margin-top: 10px;">
                    ✓ Ready for Processing - No further review needed
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-size: 0.9rem; color: #EF4444; font-weight: 500; text-align: center; margin-top: 10px;">
                    ⚠ Requires exception approval - Submit to Credit Committee
                </div>
                """, unsafe_allow_html=True)

# Modern footer with account manager focus
st.markdown("""
<div class="footer">
    <p>© 2025 Loan Risk Assessment Tool for Account Managers</p>
    <p style="margin-top: 8px; font-size: 0.8rem;">Powered by XAI Technology • For Internal Bank Use Only</p>
</div>
""", unsafe_allow_html=True)