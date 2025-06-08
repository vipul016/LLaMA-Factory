# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, r2_score, 
                            accuracy_score, precision_score, 
                            recall_score, f1_score, 
                            classification_report)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from datetime import datetime
import os
from together import Together
import traceback
import logging

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
from scipy import stats
from collections import Counter
from functools import partial

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..extras.misc import fix_proxy, is_env_enabled
from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .css import CSS
from .engine import Engine


if is_gradio_available():
    import gradio as gr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def safe_operation(func):
    """Decorator for safe operation execution with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return f"Error: {str(e)}"
    return wrapper

@safe_operation
def run_all_models(X_train, X_test, y_train, y_test, problem_type, metric_average="weighted"):
    try:
        results = []
        
        if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": xgb.XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "SVC": SVC(),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "XGBoost": xgb.XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False)
            }
        
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if problem_type == "Regression":
                    metrics = {
                        "R² Score": r2_score(y_test, y_pred),
                        "MAE": np.mean(np.abs(y_test - y_pred)),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
                    }
                else:
                    metrics = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average=metric_average),
                        "Recall": recall_score(y_test, y_pred, average=metric_average),
                        "F1 Score": f1_score(y_test, y_pred, average=metric_average)
                    }
                   
                results.append({
                    "model": model_name,
                    "parameters": model.get_params(),
                    "metrics": metrics
                })
                
            except Exception as e:
                logging.error(f"Error in model {model_name}: {str(e)}")
                results.append({
                    "model": model_name,
                    "error": str(e)
                })
        
        return results
    except Exception as e:
        logging.error(f"Error in run_all_models: {str(e)}")
        raise

@safe_operation
def format_results_table(results, problem_type, metric_average="weighted"):
    try:
        if problem_type == "Regression":
            headers = ["Model", "R² Score", "MAE", "MSE", "RMSE"]
        else:
            headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        
        table = "| " + " |  ".join(headers) + "  |\n"
        
        for result in results:
            if "error" in result:
                row = [result["model"], "Error: " + result["error"]] + ["N/A"] * (len(headers) - 2)
            else:
                metrics = result["metrics"]
                if problem_type == "Regression":
                    row = [
                        result["model"],
                        f"{metrics['R² Score']:.4f}",
                        f"{metrics['MAE']:.4f}",
                        f"{metrics['MSE']:.4f}",
                        f"{metrics['RMSE']:.4f}"
                    ]
                else:
                    row = [
                        result["model"],
                        f"{metrics['Accuracy']:.4f}",
                        f"{metrics['Precision']:.4f}",
                        f"{metrics['Recall']:.4f}",
                        f"{metrics['F1 Score']:.4f}"
                    ]
            table += "|  " + "  |  ".join(row) + "  |\n"
        
        if problem_type == "Classification":
            table += f"\nMetric Averaging Method: {metric_average}"
        
        return table
    except Exception as e:
        logging.error(f"Error in format_results_table: {str(e)}")
        raise

@safe_operation
def create_log_file():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return os.path.join(log_dir, f"preprocessing_log_{timestamp}.txt")
    except Exception as e:
        logging.error(f"Error in create_log_file: {str(e)}")
        raise

@safe_operation
def log_step(log_file, step_name, details):
    try:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Step: {step_name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"{details}\n")
    except Exception as e:
        logging.error(f"Error in log_step: {str(e)}")
        raise

@safe_operation
def create_results_dir():
    try:
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir
    except Exception as e:
        logging.error(f"Error creating results directory: {str(e)}")
        raise

@safe_operation
def run_model(csv_file,
            problem_type,
            model_name,
            input_column,
            drop_column,
            corr_threshold,
            missing_threshold,
            imputation,
            scaling_method,
            test_size,
            sampling_method,
            skew_method_right,
            skew_method_left,
            outlier_method,
            outlier_action,
            metric_average="weighted",
            flag=False):
    try:
        if csv_file is None:
            return "Please upload a CSV File", None, None
        
        log_file = create_log_file()
        results_dir = create_results_dir()
        results_file = os.path.join(results_dir, "results.csv")
        
        log_step(log_file, "Initial Data Loading", f"Loading data from {csv_file.name}")
    
        try:
            df = pd.read_csv(csv_file.name)
        except Exception as e:
            error_msg = f"Error reading CSV file: {str(e)}"
            logging.error(error_msg)
            return error_msg, None, None

        initial_columns = list(df.columns)
        log_step(log_file, "Initial Data Info", f"Initial columns: {initial_columns}\nShape: {df.shape}")

        try:
            if drop_column:
                df.drop(columns=drop_column, errors="ignore", inplace=True)
                log_step(log_file, "Manual Column Dropping", f"Dropped columns: {drop_column}")
        except Exception as e:
            logging.warning(f"Error in column dropping: {str(e)}")

        try:
            drop_nan = df.isnull().sum()
            drop_nan = drop_nan[drop_nan.values > len(df)*missing_threshold]
            if not drop_nan.empty:
                df.drop(labels=drop_nan.index, inplace=True, axis=1)
                log_step(log_file, "High Missing Value Columns Dropped", 
                        f"Columns dropped due to >{missing_threshold*100}% missing values:\n\n{drop_nan.to_string()}")
        except Exception as e:
            logging.warning(f"Error in missing value handling: {str(e)}")

        try:
            if corr_threshold > 0:
                df_numeric = df.select_dtypes(include=['number']).drop(columns=[input_column], errors='ignore')
                if not df_numeric.empty:
                    corr_matrix = df_numeric.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
                    if to_drop:
                        df.drop(to_drop, axis=1, inplace=True)
                        log_step(log_file, "Correlation-based Feature Removal", 
                                f"Threshold: {corr_threshold}\nDropped columns: {to_drop}")
        except Exception as e:
            logging.warning(f"Error in correlation analysis: {str(e)}")
    
        try:
            if imputation != "None":
                imputed_cols = []
                for col in df.columns:
                    try:
                        if df[col].dtype in [float, int]:
                            if imputation == "Mean":
                                df.loc[:, col] = df[col].fillna(df[col].mean())
                                imputed_cols.append((col, "Mean", df[col].mean()))
                            elif imputation == "Median":
                                df.loc[:, col] = df[col].fillna(df[col].median())
                                imputed_cols.append((col, "Median", df[col].median()))
                        else:
                            mode_val = df[col].mode()[0]
                            df.loc[:, col] = df[col].fillna(mode_val)
                            imputed_cols.append((col, "Mode", mode_val))
                    except Exception as e:
                        logging.warning(f"Error imputing column {col}: {str(e)}")
                        continue
                if imputed_cols:
                    log_step(log_file, "Imputation", 
                            "Imputed columns:\n" + "\n".join([f"{col}: {method} = {value}" for col, method, value in imputed_cols]))
        except Exception as e:
            logging.warning(f"Error in imputation: {str(e)}")

        try:
            if outlier_method != "None" and outlier_action != "None":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outlier_info = []
                for col in numeric_cols:
                    try:
                        if col == input_column:
                            continue
                        if outlier_method == "IQR":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            mask = (df[col] < lower) | (df[col] > upper)
                        elif outlier_method == "Z-Score":
                            z_scores = np.abs(stats.zscore(df[col].dropna()))
                            mask = z_scores > 3
                            mask = pd.Series(mask, index=df[col].dropna().index).reindex_like(df[col])
                            mask = mask.fillna(False)
                        elif outlier_method == "Percentile":
                            lower = df[col].quantile(0.01)
                            upper = df[col].quantile(0.99)
                            mask = (df[col] < lower) | (df[col] > upper)
                        else:
                            continue
                        
                        outlier_ratio = mask.sum()/len(df)
                        if outlier_ratio > .30:
                            df.drop(columns=col, inplace=True)
                            outlier_info.append(f"Column {col} dropped due to {outlier_ratio:.2%} outliers")
                        else:
                            if outlier_action == "Remove Rows":
                                df = df[~mask]
                                outlier_info.append(f"Column {col}: {mask.sum()} rows removed")
                    except Exception as e:
                        logging.warning(f"Error processing outliers for column {col}: {str(e)}")
                        continue
                
                if outlier_info:
                    log_step(log_file, "Outlier Handling", 
                            f"Method: {outlier_method}\nAction: {outlier_action}\n" + "\n".join(outlier_info))
        except Exception as e:
            logging.warning(f"Error in outlier handling: {str(e)}")
   
        try:
            if skew_method_right != "None" or skew_method_left != "None":
                skew_info = []
                for col in df.select_dtypes(include=[np.number]).columns:
                    try:
                        if col == input_column:
                            continue
                        skew = df[col].skew()
                        if skew > 0.5 and skew_method_right != "None":
                            if skew_method_right == "Square Root":
                                df[col] = np.sqrt(np.clip(df[col], a_min=0, a_max=None))
                            elif skew_method_right == "Cube Root":
                                df[col] = np.cbrt(df[col])
                            elif skew_method_right == "Logarithms":
                                pass
                            elif skew_method_right == "Reciprocal":
                                df[col] = 1 / df[col].replace(0, 1e-9)
                            skew_info.append(f"Column {col}: Right skew ({skew:.2f}) -> {skew_method_right}")
                        elif skew < -0.5 and skew_method_left != "None":
                            if skew_method_left == "Square":
                                df[col] = df[col] ** 2
                            elif skew_method_left == "Cube":
                                df[col] = df[col] ** 3
                            skew_info.append(f"Column {col}: Left skew ({skew:.2f}) -> {skew_method_left}")
                    except Exception as e:
                        logging.warning(f"Error processing skewness for column {col}: {str(e)}")
                        continue
                
                if skew_info:
                    log_step(log_file, "Skewness Handling", "\n".join(skew_info))
        except Exception as e:
            logging.warning(f"Error in skewness handling: {str(e)}")

        try:
            if input_column in df.columns and pd.api.types.is_object_dtype(df[input_column]) or problem_type == "Classification":
                le = LabelEncoder()
                df[input_column] = le.fit_transform(df[input_column])
                log_step(log_file, "Target Encoding", f"Encoded target column: {input_column}")
        except Exception as e:
            logging.warning(f"Error in target encoding: {str(e)}")

        try:
            y = df[input_column]
            X = df.drop(input_column, axis=1)
        except Exception as e:
            error_msg = f"Error preparing features and target: {str(e)}"
            logging.error(error_msg)
            return error_msg, None, None

        try:
            if scaling_method != "None":
                if scaling_method == "Standard Scaler":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
                scaled_vals = scaler.fit_transform(X[numeric_cols])
                X[numeric_cols] = pd.DataFrame(scaled_vals, columns=numeric_cols, index=X.index)
                log_step(log_file, "Feature Scaling", f"Method: {scaling_method}\nScaled columns: {list(numeric_cols)}")
        except Exception as e:
            logging.warning(f"Error in feature scaling: {str(e)}")
    
        try:
            category_column = X.select_dtypes(include='object').columns.tolist()
            columns_to_encode = []
            dropped_columns_info = []
            
            for col in category_column:
                try:
                    if col not in X.columns:
                        continue
                    unique_count = X[col].nunique()
                    if unique_count <= 20:
                        columns_to_encode.append(col)
                    else:
                        dropped_columns_info.append((col, unique_count))
                        X.drop(col, axis=1, inplace=True)
                except Exception as e:
                    logging.warning(f"Error processing categorical column {col}: {str(e)}")
                    continue
            
            if dropped_columns_info:
                log_step(log_file, "Categorical Column Dropped", 
                        f"Columns dropped due to high cardinality:\n" + 
                        "\n".join([f"{col}: {count} unique values" for col, count in dropped_columns_info]))

            if columns_to_encode:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_data = encoder.fit_transform(X[columns_to_encode])
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=encoder.get_feature_names_out(columns_to_encode),
                    index=X.index
                )
                X.drop(columns_to_encode, axis=1, inplace=True)
                X = pd.concat([X, encoded_df], axis=1)
                log_step(log_file, "Categorical Encoding", 
                        f"Encoded columns: {columns_to_encode}\nNew features created: {list(encoded_df.columns)}")
        except Exception as e:
            logging.warning(f"Error in categorical encoding: {str(e)}")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42)
            log_step(log_file, "Train-Test Split", f"Test size: {test_size}%\nTraining set shape: {X_train.shape}\nTest set shape: {X_test.shape}")
        except Exception as e:
            error_msg = f"Error in train-test split: {str(e)}"
            logging.error(error_msg)
            return error_msg, None, None

        try:
            if sampling_method != "None":
                if sampling_method == "Up Sampling":
                    sampler = RandomOverSampler(random_state=42)
                elif sampling_method == "Down Sampling":
                    sampler = RandomUnderSampler(random_state=42)
                elif sampling_method == "SMOTE":
                    min_samples = min(Counter(y_train).values())
                    if min_samples < 6:
                        log_step(log_file, "Sampling", 
                                f"Warning: Not enough samples for SMOTE (minimum class has {min_samples} samples). Using RandomOverSampler instead.")
                        sampler = RandomOverSampler(random_state=42)
                    else:
                        sampler = SMOTETomek(random_state=42)
                
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                log_step(log_file, "Sampling", f"Method: {sampling_method}\nNew training set shape: {X_train.shape}")
        except Exception as e:
            logging.warning(f"Error in sampling: {str(e)}")

        try:
            if model_name == "Apply All":
                results = run_all_models(X_train, X_test, y_train, y_test, problem_type, metric_average)
                table = format_results_table(results, problem_type, metric_average)
                log_step(log_file, "Model Comparison Results", table)
                
                report = f"""
                    Model Comparison Results ({problem_type})
                    Target: {input_column}
                    Metric Averaging Method: {metric_average}
                    
                    {table}
                    """
                return report, results_file, log_file
    
            if problem_type == "Regression":
                if model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Random Forest":
                    model = RandomForestRegressor()
                elif model_name == "Decision Tree":
                    model = DecisionTreeRegressor()
                elif model_name == "SVR":
                    model = SVR()
                elif model_name == "KNN":
                    model = KNeighborsRegressor()
                elif model_name == "XGBoost":
                    model = xgb.XGBRegressor()
                elif model_name == "CatBoost":
                    model = CatBoostRegressor(verbose=False)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results_df = pd.DataFrame({
                    "Actual": y_test.values.ravel(),
                    "Predicted": y_pred.ravel()
                })

                results_df.to_csv(results_file, index=False)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = np.mean(np.abs(y_test - y_pred))
                
                log_step(log_file, "Model Results", f"""
                    Model: {model_name}
                    R² Score: {r2:.4f}
                    Mean Absolute Error: {mae:.4f}
                    Mean Squared Error: {mse:.4f}
                    Root Mean Squared Error: {rmse:.4f}
                    """)
                
                report = f"""
                    Model: {model_name} ({problem_type})
                    Target: {input_column}
                    
                    Model Metrics:
                    - R² Score: {r2:.4f}
                    - Mean Absolute Error: {mae:.4f}
                    - Mean Squared Error: {mse:.4f}
                    - Root Mean Squared Error: {rmse:.4f}
                    
                    """
                return report, results_file, log_file
                
            else:  # Classification
                if model_name == "Logistic Regression":
                    model = LogisticRegression()
                elif model_name == "Random Forest":
                    model = RandomForestClassifier()
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif model_name == "SVC":
                    model = SVC()
                elif model_name == "KNN":
                    model = KNeighborsClassifier()
                elif model_name == "Naive Bayes":
                    model = GaussianNB()
                elif model_name == "XGBoost":
                    model = xgb.XGBClassifier()
                elif model_name == "CatBoost":
                    model = CatBoostClassifier(verbose=False)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results_df = pd.DataFrame({
                    "Actual": y_test.values.ravel(),
                    "Predicted": y_pred.ravel()
                })

                results_df.to_csv(results_file, index=False)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average=metric_average)
                recall = recall_score(y_test, y_pred, average=metric_average)
                f1 = f1_score(y_test, y_pred, average=metric_average)
                class_report = classification_report(y_test, y_pred)
                
                log_step(log_file, "Model Results", f"""
                    Model: {model_name}
                    Metric Averaging Method: {metric_average}
                    Accuracy: {accuracy:.4f}
                    Precision: {precision:.4f}
                    Recall: {recall:.4f}
                    F1 Score: {f1:.4f}
                    
                    Classification Report:
                    {class_report}
                    """)
                
                report = f"""
                    Model: {model_name} ({problem_type})
                    Target: {input_column}
                    Metric Averaging Method: {metric_average}
                    
                    Model Metrics:
                    - Accuracy: {accuracy:.4f}
                    - Precision: {precision:.4f}
                    - Recall: {recall:.4f}
                    - F1 Score: {f1:.4f}
                    
                    Classification Report:
                    {class_report}
                    
                    """
                return report, results_file, log_file
        except Exception as e:
            error_msg = f"Error in model training and evaluation: {str(e)}"
            logging.error(error_msg)
            return error_msg, None, None

    except Exception as e:
        error_msg = f"Unexpected error in run_model: {str(e)}"
        logging.error(error_msg)
        return error_msg, None, None

@safe_operation
def analyze_log_with_ai(log_file, user_query):
    try:
        if not log_file:
            return "No log file available for analysis."
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        try:
            client = Together(api_key='70a19ed73f75eacf938df0612fcbd9c4d3d7fd1089d6a23d9eace82f134d07f5')
        except Exception as e:
            logging.error(f"Error initializing Together client: {str(e)}")
            return "Error initializing AI analysis service."
        
        try:
            prompt = f"""Here is the preprocessing and model training log:
{log_content}

User Query: {user_query}

Please analyze the log and provide insights based on the user's query. Focus on explaining the preprocessing steps, model performance, and any relevant patterns or issues you notice. make it concise and to the point. Don't exceed 200 words."""

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in AI analysis: {str(e)}")
            return f"Error analyzing log: {str(e)}"
    except Exception as e:
        logging.error(f"Error in analyze_log_with_ai: {str(e)}")
        return f"Error analyzing log: {str(e)}"

@safe_operation
def create_data_visualizations(df, missing_threshold=0.3, outlier_method="IQR", outlier_action="Remove Rows", plot_type="all"):
    try:
        plots = []
        
        if plot_type in ["all", "missing"]:
            try:
                missing_values = df.isnull().sum()
                missing_values = missing_values[missing_values > 0]
                if not missing_values.empty:
                    threshold_line = len(df) * missing_threshold
                    fig_missing = px.bar(
                        x=missing_values.index,
                        y=missing_values.values,
                        title=f"Missing Values by Column (Threshold: {missing_threshold*100}%)",
                        labels={'x': 'Columns', 'y': 'Number of Missing Values'}
                    )
                    fig_missing.add_hline(
                        y=threshold_line,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Threshold",
                        annotation_position="top right"
                    )
                    plots.append((fig_missing, "Missing Values"))
            except Exception as e:
                logging.warning(f"Error creating missing values plot: {str(e)}")
        
        if plot_type in ["all", "skewness"]:
            try:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                skewed_cols = [col for col in numeric_cols if abs(df[col].skew()) > 0.5]
                
                if skewed_cols:
                    n_cols = len(skewed_cols)
                    n_rows = (n_cols + 1) // 2 
                    
                    fig_skew = make_subplots(
                        rows=n_rows,
                        cols=2,
                        subplot_titles=[f"{col} (Skew: {df[col].skew():.2f})" for col in skewed_cols]
                    )
                    
                    for idx, col in enumerate(skewed_cols):
                        try:
                            row = idx // 2 + 1
                            col_num = idx % 2 + 1
                            
                            fig_skew.add_trace(
                                go.Histogram(x=df[col], name=col),
                                row=row, col=col_num
                            )
                        except Exception as e:
                            logging.warning(f"Error creating skewness plot for column {col}: {str(e)}")
                            continue
                    
                    fig_skew.update_layout(
                        title="Skewness Analysis",
                        height=300 * n_rows,
                        showlegend=False
                    )
                    plots.append((fig_skew, "Skewness Analysis"))
            except Exception as e:
                logging.warning(f"Error in skewness analysis: {str(e)}")
        
        if plot_type in ["all", "outliers"]:
            try:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                outlier_cols = []
                
                for col in numeric_cols:
                    try:
                        if outlier_method == "IQR":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
                        elif outlier_method == "Z-Score":
                            z_scores = np.abs(stats.zscore(df[col].dropna()))
                            outliers = df[col][z_scores > 3]
                        elif outlier_method == "Percentile":
                            lower = df[col].quantile(0.01)
                            upper = df[col].quantile(0.99)
                            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
                        
                        if len(outliers) > 0:
                            outlier_cols.append(col)
                    except Exception as e:
                        logging.warning(f"Error processing outliers for column {col}: {str(e)}")
                        continue
                
                if outlier_cols:
                    n_cols = len(outlier_cols)
                    n_rows = (n_cols + 1) // 2  
                    
                    fig_outlier = make_subplots(
                        rows=n_rows,
                        cols=2,
                        subplot_titles=[f"{col} (Outliers: {len(df[(df[col] < df[col].quantile(0.01)) | (df[col] > df[col].quantile(0.99))][col])})" for col in outlier_cols]
                    )
                    
                    for idx, col in enumerate(outlier_cols):
                        try:
                            row = idx // 2 + 1
                            col_num = idx % 2 + 1
                            
                            fig_outlier.add_trace(
                                go.Box(y=df[col], name=col),
                                row=row, col=col_num
                            )
                        except Exception as e:
                            logging.warning(f"Error creating outlier plot for column {col}: {str(e)}")
                            continue
                    
                    fig_outlier.update_layout(
                        title=f"Outlier Analysis (Method: {outlier_method})",
                        height=300 * n_rows,
                        showlegend=False
                    )
                    plots.append((fig_outlier, "Outlier Analysis"))
            except Exception as e:
                logging.warning(f"Error in outlier analysis: {str(e)}")
        
        return plots
    except Exception as e:
        logging.error(f"Error in create_data_visualizations: {str(e)}")
        return []

def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Board ({hostname})", css=CSS) as demo:
        with gr.Tabs():
            with gr.TabItem("LLM"):
                if demo_mode:
                    gr.HTML("<h1><center>LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory</center></h1>")
                    gr.HTML(
                        '<h3><center>Visit <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">'
                        "LLaMA Factory</a> for details.</center></h3>"
                    )
                    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

                engine.manager.add_elems("top", create_top())
                lang: gr.Dropdown = engine.manager.get_elem_by_id("top.lang")

                with gr.Tab("Train"):
                    engine.manager.add_elems("train", create_train_tab(engine))

                with gr.Tab("Evaluate & Predict"):
                    engine.manager.add_elems("eval", create_eval_tab(engine))

                with gr.Tab("Chat"):
                    engine.manager.add_elems("infer", create_infer_tab(engine))

                if not demo_mode:
                    with gr.Tab("Export"):
                        engine.manager.add_elems("export", create_export_tab(engine))

                demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
                lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
                lang.input(save_config, inputs=[lang], queue=False)
        
            with gr.TabItem("Machine Learning"):
                with gr.Row():
                    file_upload = gr.File(label="Upload CSV File", file_types=['.csv'])
                    target_column = gr.Dropdown([], label="Target Column", interactive=True)
                with gr.Tabs():
                    with gr.Tab("Model Training"):
                        with gr.Row():
                            problem_type=gr.Dropdown(
                                ["Regression","Classification"],
                                label='Problem Type',
                                value="Regression",interactive=True)
                            
                            model_name=gr.Dropdown(
                                ["Linear Regression","Random Forest","Decision Tree","SVR","KNN","XGBoost","CatBoost","Apply All"],
                                label="Model",
                                value="Linear Regression",
                                interactive=True
                            )

                       
                            metric_average = gr.Dropdown(
                                ["weighted", "macro", "micro", "binary"],
                                label="Classification Metric Averaging Method",
                                value="weighted",
                                interactive=True,
                                visible=False  
                            )

                        drop_column=gr.CheckboxGroup([],label="Columns to Drop")

                        with gr.Row():
                            corr_threshold=gr.Slider(
                                0,1,value=0.95,step=0.01,
                                label="Correlation Threshold (Remove Features Above)"
                            )
                            missing_threshold=gr.Slider(
                                0,1,value=0.30,step=0.05,
                                label="Missing Value Threshold (Remove Features Above)"
                            )
                            imputation=gr.Dropdown(
                                ["Mean","Median"],
                                label="Imputation",
                                value="Mean",
                                interactive=True,
                            )
                            
                        with gr.Row():
                            test_size=gr.Slider(
                                0,50,value=20,step=0.05,
                                label="Test Size"
                            )
                            sampling_method=gr.Dropdown(
                                ["None","Up Sampling","Down Sampling","SMOTE"],
                                label="Sampling Method",
                                value="None",
                                interactive=True,
                            )
                            scaling_method=gr.Dropdown(
                                ["None","Standard Scaler","MinMax Scaler"],
                                label="Scaling Method",
                                value="Standard Scaler",
                                interactive=True
                            )
                        with gr.Row():
                            skew_method_right=gr.Dropdown(
                                ["None","Square Root","Cube Root","Logarithms","Reciprocal"],
                                label="Right Skewness Method",
                                value="Square Root",
                                interactive=True
                            )
                            skew_method_left=gr.Dropdown(
                                ["None","Square","Cube"],
                                label="Left Skewness Method",
                                value="Square",
                                interactive=True
                            )
                        with gr.Row():
                            outlier_method=gr.Dropdown(
                                ["None","IQR","Z-Score","Percentile"],
                                label="Outlier Detection Method",
                                value="IQR",
                                interactive=True
                            )
                            outlier_action=gr.Dropdown(
                                ["None","Remove Rows"],
                                label="Outlier Handling Action",
                                value="Remove Rows",
                                interactive=True
                            )

                        run_button = gr.Button("Run", variant="primary")
                        
                        
                        output_features = gr.TextArea(label="Processed Data", lines=15)
                        with gr.Row():
                            download_csv = gr.File(label="Download Prediction CSV")
                            download_log = gr.File(label="Download Processing Log")

                      
                        with gr.Row():
                            with gr.Column():
                                chatbot = gr.Chatbot(type="messages", label="AI Analysis", height=400)
                                msg = gr.Textbox(label="Ask about your model", placeholder="Type your question here...")
                                clear = gr.ClearButton([msg, chatbot])
                    with gr.Tab("Data Analysis"):
                            with gr.Row():
                                with gr.Column():
                                    data_plots = gr.Plot(
                                        label="Data Analysis Plots",
                                        show_label=True,
                                        elem_id="plots"
                                    )
                                with gr.Column():
                                    plot_type = gr.Radio(
                                        ["Missing Values", "Skewness", "Outliers"],
                                        label="Plot Type",
                                        value="Missing Values"
                                    )

                def update_model_options(problem_type):
                    if problem_type=="Regression":
                        return gr.Dropdown(choices=["Linear Regression","Random Forest","Decision Tree","SVR","KNN","XGBoost","CatBoost","Apply All"],
                        value="Linear Regression",
                        label="Model",
                        interactive=True), gr.Dropdown(visible=False)
                    else:
                        return gr.Dropdown(
                            choices=["Logistic Regression","Random Forest","Decision Tree","SVC","KNN","Naive Bayes","XGBoost","CatBoost","Apply All"],
                            value="Logistic Regression",
                            label="Model",
                            interactive=True), gr.Dropdown(visible=True)

                problem_type.change(
                    update_model_options,
                    inputs=problem_type,
                    outputs=[model_name, metric_average]
                )

                def update_column(csv_file):
                    if csv_file is None:
                        return [gr.Dropdown(choices=[]),gr.CheckboxGroup(choices=[])]
                    df=pd.read_csv(csv_file.name)
                    columns=list(df.columns)
                    return [
                        gr.Dropdown(choices=columns,value=columns[-1] if columns else None,interactive=True),
                        gr.CheckboxGroup(choices=columns)
                    ]

                file_upload.change(
                    update_column,
                    inputs=[file_upload],
                    outputs=[target_column,drop_column]
                )

                def analyze_query(message, chat_history, log_file):
                    if not log_file:
                        chat_history.append({"role": "user", "content": message})
                        chat_history.append({"role": "assistant", "content": "Please run the model first to generate a log file."})
                        return "", chat_history
                    
                    response = analyze_log_with_ai(log_file, message)
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": response})
                    return "", chat_history

                def run_and_analyze(csv_file, problem_type, model_name, input_column, drop_column, 
                                corr_threshold, missing_threshold, imputation, scaling_method, test_size,
                                sampling_method, skew_method_right, skew_method_left,
                                outlier_method, outlier_action, metric_average,flag=False):
                    
                    report, results_file, log_file = run_model(
                        csv_file, problem_type, model_name, input_column, drop_column,
                        corr_threshold, missing_threshold, imputation, scaling_method, test_size,
                        sampling_method, skew_method_right, skew_method_left,
                        outlier_method, outlier_action, metric_average,
                    )
                    if flag:
                        initial_analysis = analyze_log_with_ai(log_file, "Provide a summary of the preprocessing steps and model performance")
                        chat_history = [{"role": "assistant", "content": initial_analysis}]
                    else: 
                        chat_history=""
                    
                    return report, results_file, log_file, chat_history

                run_button.click(
                    run_and_analyze,
                    inputs=[
                        file_upload,
                        problem_type,
                        model_name,
                        target_column,
                        drop_column,
                        corr_threshold,
                        missing_threshold,
                        imputation,
                        scaling_method,
                        test_size,
                        sampling_method,
                        skew_method_right,
                        skew_method_left,
                        outlier_method,
                        outlier_action,
                        metric_average,
                    ],
                    outputs=[output_features, download_csv, download_log, chatbot]
                )

                msg.submit(
                    analyze_query,
                    inputs=[msg, chatbot, download_log],
                    outputs=[msg, chatbot]
                )

                df_state = gr.State(None)
                
                def load_data(csv_file):
                    if csv_file is None:
                        return None, None
                    df = pd.read_csv(csv_file.name)
                    plots = create_data_visualizations(df)
                    return df, plots[0][0] if plots else None
                
                def update_missing_plots(df, missing_threshold):
                    if df is None:
                        return None
                    plots = create_data_visualizations(df, missing_threshold=missing_threshold, plot_type="missing")
                    return plots[0][0] if plots else None
                
                def update_outlier_plots(df, outlier_method, outlier_action):
                    if df is None:
                        return None
                    plots = create_data_visualizations(df, outlier_method=outlier_method, outlier_action=outlier_action, plot_type="outliers")
                    return plots[0][0] if plots else None
                
                def update_plot_type(df, plot_type, missing_threshold, outlier_method, outlier_action):
                    if df is None:
                        return None
                    plot_type_map = {
                        "All Plots": "all",
                        "Missing Values": "missing",
                        "Skewness": "skewness",
                        "Outliers": "outliers"
                    }
                    plots = create_data_visualizations(
                        df, 
                        missing_threshold=missing_threshold,
                        outlier_method=outlier_method,
                        outlier_action=outlier_action,
                        plot_type=plot_type_map[plot_type]
                    )
                    return plots[0][0] if plots else None
                
                file_upload.change(
                    load_data,
                    inputs=[file_upload],
                    outputs=[df_state, data_plots]
                )
                
                missing_threshold.change(
                    update_missing_plots,
                    inputs=[df_state, missing_threshold],
                    outputs=[data_plots]
                )
                
                outlier_method.change(
                    update_outlier_plots,
                    inputs=[df_state, outlier_method, outlier_action],
                    outputs=[data_plots]
                )
                
                outlier_action.change(
                    update_outlier_plots,
                    inputs=[df_state, outlier_method, outlier_action],
                    outputs=[data_plots]
                )
                
                plot_type.change(
                    update_plot_type,
                    inputs=[df_state, plot_type, missing_threshold, outlier_method, outlier_action],
                    outputs=[data_plots]
                )

    return demo


def create_web_demo() -> "gr.Blocks":
    engine = Engine(pure_chat=True)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory Web Demo ({hostname})", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], scale=1)
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


def run_web_demo() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True) 