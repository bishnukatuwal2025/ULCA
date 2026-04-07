import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PATH, MODEL_PATH, SCALER_PATH  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.predict import make_single_prediction  # noqa: E402

st.set_page_config(page_title="UCLA Admission Prediction App", layout="wide")


@st.cache_data
def load_raw_data():
    return load_data(DATA_PATH)


@st.cache_data
def prepare_notebook_data():
    data = load_raw_data().copy()

    raw_shape = data.shape
    raw_missing = data.isnull().sum()

    # Match notebook logic: convert target directly, do not create extra target column
    data["Admit_Chance"] = (data["Admit_Chance"] >= 0.8).astype(int)
    class_counts = data["Admit_Chance"].value_counts().sort_index()

    # Drop Serial_No so shape becomes (500, 8)
    working_data = data.drop(columns=["Serial_No"]).copy()

    # Keep a copy before encoding for EDA and visuals
    eda_data = working_data.copy()

    # Prepare modeling data
    modeling_data = working_data.copy()
    modeling_data["University_Rating"] = modeling_data["University_Rating"].astype("object")
    modeling_data["Research"] = modeling_data["Research"].astype("object")

    clean_data = pd.get_dummies(
        modeling_data,
        columns=["University_Rating", "Research"],
        dtype=int
    )

    X = clean_data.drop(columns=["Admit_Chance"])
    y = clean_data["Admit_Chance"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    xtrain_scaled_df = pd.DataFrame(xtrain_scaled, columns=xtrain.columns)
    xtest_scaled_df = pd.DataFrame(xtest_scaled, columns=xtest.columns)

    # Baseline model
    mlp_base = MLPClassifier(
        hidden_layer_sizes=(3,),
        batch_size=50,
        max_iter=200,
        random_state=123
    )
    mlp_base.fit(xtrain_scaled, ytrain)

    ypred_train_base = mlp_base.predict(xtrain_scaled)
    ypred_test_base = mlp_base.predict(xtest_scaled)

    # Tuned model from notebook
    mlp_tanh = MLPClassifier(
        hidden_layer_sizes=(3,),
        batch_size=50,
        max_iter=200,
        random_state=123,
        activation="tanh"
    )
    mlp_tanh.fit(xtrain_scaled, ytrain)

    ypred_train_tanh = mlp_tanh.predict(xtrain_scaled)
    ypred_test_tanh = mlp_tanh.predict(xtest_scaled)

    return {
        "raw_data": load_raw_data(),
        "binary_data": data,
        "raw_shape": raw_shape,
        "raw_missing": raw_missing,
        "working_data": working_data,
        "eda_data": eda_data,
        "clean_data": clean_data,
        "class_counts": class_counts,
        "xtrain": xtrain,
        "xtest": xtest,
        "ytrain": ytrain,
        "ytest": ytest,
        "xtrain_scaled": xtrain_scaled,
        "xtest_scaled": xtest_scaled,
        "xtrain_scaled_df": xtrain_scaled_df,
        "xtest_scaled_df": xtest_scaled_df,
        "mlp_base": mlp_base,
        "mlp_tanh": mlp_tanh,
        "ypred_train_base": ypred_train_base,
        "ypred_test_base": ypred_test_base,
        "ypred_train_tanh": ypred_train_tanh,
        "ypred_test_tanh": ypred_test_tanh,
    }


@st.cache_resource
def load_saved_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def show_histogram(df, column):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def show_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    ax.set_ylabel(column)
    st.pyplot(fig)


def show_bar(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)


def show_corr_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


def show_scatter_notebook_style(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="GRE_Score",
        y="TOEFL_Score",
        hue="Admit_Chance",
        palette="deep",
        ax=ax
    )
    ax.set_title("GRE Score vs TOEFL Score by Admission Class")
    ax.set_xlabel("GRE Score")
    ax.set_ylabel("TOEFL Score")
    st.pyplot(fig)


def show_cgpa_vs_admit(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="CGPA",
        y="Admit_Chance",
        hue="Research",
        palette="deep",
        ax=ax
    )
    ax.set_title("CGPA vs Admission Class by Research")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Admit Class")
    st.pyplot(fig)


def show_pairplot_sample(df):
    pair_df = df[["GRE_Score", "TOEFL_Score", "CGPA", "SOP", "Admit_Chance"]].copy()
    fig = sns.pairplot(pair_df, hue="Admit_Chance", diag_kind="hist")
    st.pyplot(fig)


def show_scaling_comparison(raw_df, scaled_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.histplot(raw_df["GRE_Score"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Original GRE Score")

    sns.histplot(scaled_df["GRE_Score"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Scaled GRE Score")

    sns.histplot(raw_df["TOEFL_Score"], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Original TOEFL Score")

    sns.histplot(scaled_df["TOEFL_Score"], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Scaled TOEFL Score")

    plt.tight_layout()
    st.pyplot(fig)


def show_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    st.pyplot(fig)


def show_loss_curve(model, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model.loss_curve_)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    st.pyplot(fig)


st.title("UCLA Admission Prediction using Neural Networks")

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "EDA",
        "Visualizations",
        "Correlation",
        "Model Evaluation",
        "Prediction",
    ]
)

try:
    results = prepare_notebook_data()
    saved_model, saved_scaler = load_saved_artifacts()

    raw_data = results["raw_data"]
    binary_data = results["binary_data"]
    working_data = results["working_data"]
    eda_data = results["eda_data"]
    xtrain_scaled_df = results["xtrain_scaled_df"]
    ytrain = results["ytrain"]
    ytest = results["ytest"]

    if section == "Home":
        st.header("Project Overview")
        st.write(
            """
            This Streamlit application presents the complete UCLA admission prediction workflow.
            It includes dataset overview, exploratory data analysis, notebook-based visualizations,
            correlation analysis, neural network model evaluation, and final prediction.
            """
        )

        st.subheader("Application Sections")
        st.write("- Dataset overview")
        st.write("- Exploratory Data Analysis")
        st.write("- Visualizations from the notebook logic")
        st.write("- Correlation analysis")
        st.write("- Model evaluation")
        st.write("- Final prediction")

    elif section == "Dataset":
        st.header("Dataset Overview")

        st.write(f"Raw dataset shape: {results['raw_shape']}")
        st.write(f"Shape after dropping Serial_No: {working_data.shape}")

        st.subheader("First 10 Rows of Raw Data")
        st.dataframe(raw_data.head(10))

        st.subheader("First 10 Rows After Target Conversion and Dropping Serial_No")
        st.dataframe(working_data.head(10))

        st.subheader("Columns")
        st.write(list(raw_data.columns))

        st.subheader("Data Types")
        dtype_df = raw_data.dtypes.astype(str).reset_index()
        dtype_df.columns = ["Column", "Data Type"]
        st.dataframe(dtype_df)

        st.subheader("Missing Values")
        missing_df = results["raw_missing"].reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df)

    elif section == "EDA":
        st.header("Exploratory Data Analysis")

        st.subheader("Summary Statistics of Raw Data")
        st.dataframe(raw_data.describe())

        st.subheader("Summary Statistics After Dropping Serial_No")
        st.dataframe(working_data.describe())

        st.subheader("Binary Target Distribution Based on Admit_Chance >= 0.8")
        class_counts = results["class_counts"].copy()
        class_df = class_counts.reset_index()
        class_df.columns = ["Admit Class", "Count"]
        class_df["Admit Class"] = class_df["Admit Class"].map({
            0: "0 = Lower Chance",
            1: "1 = High Chance"
        })
        st.dataframe(class_df)

        display_counts = class_counts.copy()
        display_counts.index = ["0 = Lower Chance", "1 = High Chance"]
        show_bar(display_counts, "Admission Class Distribution", "Class", "Count")

        st.subheader("Notebook-Based Findings")
        st.write("- Raw dataset contains 500 rows and 9 columns.")
        st.write("- After dropping Serial_No, the dataset contains 500 rows and 8 columns.")
        st.write("- The dataset has no missing values.")
        st.write("- The target was converted to binary using the threshold Admit_Chance >= 0.8.")

    elif section == "Visualizations":
        st.header("Visualizations")

        st.subheader("1. GRE Score vs TOEFL Score by Admission Class")
        show_scatter_notebook_style(binary_data)

        st.subheader("2. Scaling Comparison for GRE Score and TOEFL Score")
        scale_compare_df = pd.DataFrame({
            "GRE_Score": results["xtrain_scaled_df"]["GRE_Score"],
            "TOEFL_Score": results["xtrain_scaled_df"]["TOEFL_Score"]
        })
        show_scaling_comparison(working_data, scale_compare_df)

        st.subheader("3. CGPA Distribution")
        show_histogram(eda_data, "CGPA")
        show_boxplot(eda_data, "CGPA")

        st.subheader("4. GRE Score Distribution")
        show_histogram(eda_data, "GRE_Score")
        show_boxplot(eda_data, "GRE_Score")

        st.subheader("5. TOEFL Score Distribution")
        show_histogram(eda_data, "TOEFL_Score")
        show_boxplot(eda_data, "TOEFL_Score")

        st.subheader("6. CGPA vs Admission Class by Research")
        show_cgpa_vs_admit(binary_data)

        st.subheader("7. Research Distribution")
        research_counts = eda_data["Research"].value_counts().sort_index()
        research_counts.index = ["No Research", "Research"]
        show_bar(research_counts, "Research Experience Distribution", "Category", "Count")

        st.subheader("8. University Rating Distribution")
        uni_counts = eda_data["University_Rating"].value_counts().sort_index()
        show_bar(uni_counts, "University Rating Distribution", "University Rating", "Count")

    elif section == "Correlation":
        st.header("Correlation Analysis")
        st.write("Correlation matrix based on the dataset after dropping Serial_No.")
        show_corr_heatmap(working_data)

    elif section == "Model Evaluation":
        st.header("Model Evaluation")

        base_train_acc = accuracy_score(ytrain, results["ypred_train_base"])
        base_test_acc = accuracy_score(ytest, results["ypred_test_base"])
        tanh_train_acc = accuracy_score(ytrain, results["ypred_train_tanh"])
        tanh_test_acc = accuracy_score(ytest, results["ypred_test_tanh"])

        st.subheader("Accuracy Comparison")
        acc_df = pd.DataFrame({
            "Model": ["Baseline MLP", "MLP with tanh"],
            "Train Accuracy": [base_train_acc, tanh_train_acc],
            "Test Accuracy": [base_test_acc, tanh_test_acc],
        })
        st.dataframe(acc_df)

        st.subheader("Best Model Result")
        col1, col2 = st.columns(2)
        col1.metric("Train Accuracy", f"{tanh_train_acc:.4f}")
        col2.metric("Test Accuracy", f"{tanh_test_acc:.4f}")

        st.subheader("Confusion Matrix - Baseline MLP")
        cm_base = confusion_matrix(ytest, results["ypred_test_base"])
        show_confusion_matrix(cm_base, "Baseline MLP Confusion Matrix")

        st.subheader("Confusion Matrix - MLP with tanh")
        cm_tanh = confusion_matrix(ytest, results["ypred_test_tanh"])
        show_confusion_matrix(cm_tanh, "MLP with tanh Confusion Matrix")

        st.subheader("Classification Report - MLP with tanh")
        report = classification_report(ytest, results["ypred_test_tanh"])
        st.text(report)

        st.subheader("Loss Curve - MLP with tanh")
        show_loss_curve(results["mlp_tanh"], "Loss Curve")

    elif section == "Prediction":
        st.header("Admission Prediction")
        st.write("Enter student information to predict admission class.")

        gre_score = st.slider("GRE Score", 290, 340, 320)
        toefl_score = st.slider("TOEFL Score", 92, 120, 110)
        university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
        sop = st.slider("SOP", 1.0, 5.0, 3.5, 0.5)
        lor = st.slider("LOR", 1.0, 5.0, 3.5, 0.5)
        cgpa = st.slider("CGPA", 6.8, 9.92, 8.5, 0.01)
        research = st.selectbox("Research Experience", [0, 1], index=1)

        if st.button("Predict Admission"):
            input_data = {
                "GRE_Score": gre_score,
                "TOEFL_Score": toefl_score,
                "University_Rating": university_rating,
                "SOP": sop,
                "LOR": lor,
                "CGPA": cgpa,
                "Research": research,
            }

            prediction = make_single_prediction(saved_model, saved_scaler, input_data)

            if prediction == 1:
                st.success("Prediction Result: High chance of admission")
            else:
                st.warning("Prediction Result: Lower chance of admission")

except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")