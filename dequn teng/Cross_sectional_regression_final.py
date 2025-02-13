import pandas as pd
import numpy as np
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import seaborn as sns
import matplotlib.pyplot as plt

"""
Hypotheses:
H1: Technical specifications (flow temperature and system sizing) significantly affect heat pump SPFH4 performance?
H2: Building characteristics (floor area and house type) significantly influence SPFH4 performance beyond technical effects?
H3: Installation quality (installation time and commissioning) significantly impact SPFH4 performance after controlling for 
    technical and building characteristics?
"""

def prepare_data_for_analysis(df):
    """Prepare data for regression analysis with proper type handling"""
    
    # Filter data for SPF analysis
    df = df[df['Included_SPF_analysis'] == True].copy()
    print(f"\nFiltered for SPF analysis: {df.shape[0]} rows")
    
    # Define column groups with clear mapping to hypotheses
    # H1: Technical Variables
    tech_cols = ['HP_Size_kW_x', 'Mean_annual_SH_flow_temp_selected_window']
    # H2: Building Variables
    building_cols = ['Total_Floor_Area', 'House_Form_x']
    # H3: Quality Variables
    quality_cols = ['Install_time_HP', 'Max_quality_score_selected_window']
    dependent_col = 'SPFH4_selected_window'
    
    # Select all needed columns
    needed_cols = tech_cols + building_cols + quality_cols + [dependent_col]
    df_clean = df[needed_cols].copy()
    
    # Convert numeric columns first
    numeric_cols = [col for col in needed_cols if col != 'House_Form_x']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Create derived variables for technical hypothesis (H1)
    df_clean['High_flow_temp'] = (df_clean['Mean_annual_SH_flow_temp_selected_window'] > 45).astype(float)
    df_clean['Size_per_area'] = df_clean['HP_Size_kW_x'] * 1000 / df_clean['Total_Floor_Area']  # W/m²
    df_clean['Oversized'] = (df_clean['Size_per_area'] > 100).astype(float)  # Threshold of 100 W/m²
    
    # Create interaction term for building hypothesis (H2)
    df_clean['Flow_temp_X_area'] = (
        df_clean['Mean_annual_SH_flow_temp_selected_window'] * 
        df_clean['Total_Floor_Area']
    )
    
    # Update numeric_cols to include new derived variables
    derived_vars = ['High_flow_temp', 'Size_per_area', 'Oversized', 'Flow_temp_X_area']
    numeric_cols = numeric_cols + derived_vars
    
    # Remove infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with any missing values in numeric columns BEFORE creating dummies
    df_clean = df_clean.dropna(subset=numeric_cols)
    
    # Create dummy variables for house form AFTER dropping NA rows
    df_clean['House_Form_x'] = df_clean['House_Form_x'].fillna('Unknown')
    house_dummies = pd.get_dummies(df_clean['House_Form_x'], prefix='House', drop_first=True)
    
    # Print summary statistics before standardization
    print("\nSummary statistics before standardization:")
    print(df_clean[numeric_cols].describe())
    
    # Standardize numeric variables
    df_standardized = df_clean.copy()
    for col in numeric_cols:
        df_standardized[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    
    # Convert dummy variables to float64
    house_dummies = house_dummies.astype('float64')
    
    # Combine standardized numeric data with dummy variables
    df_final = pd.concat([
        df_standardized[numeric_cols],  # Now includes derived variables
        house_dummies
    ], axis=1)
    
    # Final verification
    print(f"\nFinal shape: {df_final.shape}")
    print("\nData types in final dataset:")
    print(df_final.dtypes)
    print("\nMissing values in final dataset:")
    print(df_final.isna().sum())
    print("\nInfinite values in final dataset:")
    print(np.isinf(df_final.select_dtypes(include=np.number)).sum())
    
    return df_final

def fit_models(df):
    """Fit hierarchical regression models to test hypotheses"""
    
    # Define dependent variable
    y = df['SPFH4_selected_window'].astype('float64')
    
    # Model 1: Testing H1 - Technical specifications
    tech_vars = [
        'Mean_annual_SH_flow_temp_selected_window',
        'High_flow_temp',
        'HP_Size_kW_x',
        'Size_per_area',
        'Oversized'
    ]
    X1 = df[tech_vars].astype('float64')
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(y, X1).fit()
    
    # Model 2: Testing H2 - Adding building characteristics
    building_vars = [
        'Total_Floor_Area',
        'Flow_temp_X_area'
    ] + [col for col in df.columns if col.startswith('House_')]
    X2 = df[tech_vars + building_vars].astype('float64')
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()
    
    # Model 3: Testing H3 - Adding installation quality
    quality_vars = ['Install_time_HP', 'Max_quality_score_selected_window']
    X3 = df[tech_vars + building_vars + quality_vars].astype('float64')
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(y, X3).fit()
    
    return [model1, model2, model3]

def create_latex_table(models):
    """Create LaTeX table with hypothesis-focused presentation"""
    
    stargazer = Stargazer(models)
    
    # Basic settings
    stargazer.title = "Heat Pump Performance Models (SPFH4)"
    stargazer.column_separators = True
    stargazer.custom_columns = ['Technical Model', 'Building Model', 'Full Model']
    
    # Variable names with clear hypothesis mapping
    var_names = {
        'const': 'Intercept',
        'Mean_annual_SH_flow_temp_selected_window': 'Flow Temperature (°C)',
        'High_flow_temp': 'High Flow Temp (>45°C)',
        'HP_Size_kW_x': 'Heat Pump Size (kW)',
        'Size_per_area': 'Size per Floor Area (W/m²)',
        'Oversized': 'Oversized System (>100 W/m²)',
        'Total_Floor_Area': 'Floor Area (m²)',
        'Flow_temp_X_area': 'Flow Temp × Floor Area',
        'Install_time_HP': 'Installation Time (hours)',
        'Max_quality_score_selected_window': 'Commissioning Quality Score'
    }
    
    # Add house type names
    house_vars = [col for col in models[-1].model.exog_names if col.startswith('House_')]
    for var in house_vars:
        house_type = var.replace("House_", "")
        var_names[var] = f'House Type: {house_type}'
    
    # Set covariate order to match hypothesis structure
    covariate_order = [
        'const',
        # H1: Technical variables
        'Mean_annual_SH_flow_temp_selected_window',
        'High_flow_temp',
        'HP_Size_kW_x',
        'Size_per_area',
        'Oversized',
        # H2: Building variables
        'Total_Floor_Area'
    ] + house_vars + [
        'Flow_temp_X_area',
        # H3: Quality variables
        'Install_time_HP',
        'Max_quality_score_selected_window'
    ]
    
    stargazer.covariate_order = covariate_order
    stargazer.rename_covariates(var_names)
    
    # Enhanced statistics and formatting
    stargazer.significant_digits = 3
    stargazer.show_degrees_of_freedom = True
    stargazer.show_f_statistics = True
    stargazer.show_adj_r2 = True
    stargazer.show_residual_std_err = True
    stargazer.show_notes = True
    
    # Comprehensive notes
    notes = [
        'All continuous variables are standardized (mean=0, std=1)',
        'House types are dummy variables with Detached as reference category',
        'High Flow Temp is a binary indicator for flow temperatures above 45°C',
        'Oversized System indicates heat pump size > 100 W/m² of floor area',
        'Standard errors in parentheses',
        '* p<0.1, ** p<0.05, *** p<0.01'
    ]
    stargazer.add_custom_notes(notes)
    
    return stargazer

try:
    # Load data
    print("Loading data...")
    df = pd.read_csv('merged_heat_pump_data.csv')
    print("Initial shape:", df.shape)
    
    # Prepare data
    print("\nPreparing data...")
    df_prepared = prepare_data_for_analysis(df)
    print("Final shape:", df_prepared.shape)
    
    # Fit models
    print("\nFitting models...")
    models = fit_models(df_prepared)
    
    # Create and save LaTeX table
    print("\nCreating LaTeX table...")
    stargazer = create_latex_table(models)
    
    # Save full document
    latex_content = f"""
\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}
\\begin{{document}}

\\section{{Heat Pump Performance Analysis}}

\\subsection{{Research Hypotheses}}
\\begin{{enumerate}}
\\item[$H_1$:] Do technical specifications (flow temperature and system sizing) significantly affect heat pump SPFH4 performance?
\\item[$H_2$:] Do building characteristics (floor area and house type) significantly influence SPFH4 performance beyond technical effects?
\\item[$H_3$:] Does installation quality (installation time and commissioning) significantly impact SPFH4 performance after controlling for 
    technical and building characteristics?
\\end{{enumerate}}

{stargazer.render_latex()}

\\end{{document}}
"""
    
    with open('regression_table.tex', 'w') as f:
        f.write(latex_content)
    
    # Print model summaries
    print("\nModel Summaries:")
    for i, model in enumerate(models, 1):
        print(f"\nModel {i} Results:")
        print("=" * 50)
        print(model.summary().tables[1])

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("\nTraceback:")
    import traceback
    traceback.print_exc()