import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
)

st.title("Titanic Survival Prediction App")
st.write(
    "This app loads the Titanic dataset, preprocesses the data, trains a neural network, and predicts whether a passenger is likely to survive."
)

DATA_PATH = r"C:\Users\Ali\Downloads\Titanic_data\Titanic-Dataset.csv"


@st.cache_data
def load_data(default_path):
    try:
        return pd.read_csv(default_path)
    except FileNotFoundError:
        return None


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, errors="ignore")
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


@st.cache_resource
def build_model(input_dim: int):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        random_state=42,
    )


def train_model(model, X_train, y_train, epochs, batch_size):
    model.max_iter = epochs
    model.set_params(batch_size=batch_size)
    model.fit(X_train, y_train)
    history = {"loss": model.loss_curve_}
    if hasattr(model, "validation_scores_"):
        history["val_accuracy"] = model.validation_scores_
    return history


def plot_history(history, title="Training"):
    history_df = pd.DataFrame(history)
    st.subheader(f"{title} history")
    if "loss" in history_df.columns:
        st.line_chart(history_df[["loss"]].rename(columns={"loss": "Loss"}))
    if "val_accuracy" in history_df.columns:
        st.line_chart(history_df[["val_accuracy"]].rename(columns={"val_accuracy": "Validation Accuracy"}))


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)
    return X, y


def predict_passenger(model, scaler, passenger_data: dict):
    features = np.array([[
        passenger_data["Pclass"],
        passenger_data["Sex"],
        passenger_data["Age"],
        passenger_data["SibSp"],
        passenger_data["Parch"],
        passenger_data["Fare"],
        passenger_data["Embarked"],
        passenger_data["FamilySize"],
        passenger_data["IsAlone"],
    ]])
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0, 1]
    return float(prob)


with st.sidebar:
    st.header("Model settings")
    epochs = st.slider("Training epochs", min_value=10, max_value=200, value=50, step=10)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    st.write("---")
    st.markdown("**Prediction input**")
    pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    age = st.slider("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=32.0, step=0.5)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
    st.write("---")
    st.write("Train the model first, then use this form to predict a passenger.")


uploaded_file = st.file_uploader("Upload Titanic CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded from uploaded file.")
elif st.button("Load default Titanic dataset"):
    df = load_data(DATA_PATH)
    if df is None:
        st.error(f"Default dataset was not found at {DATA_PATH}.")
else:
    df = load_data(DATA_PATH)
    if df is None:
        st.info(
            "No dataset loaded yet. Upload the Titanic CSV or click 'Load default Titanic dataset'."
        )

if df is not None:
    st.subheader("Raw Dataset")
    st.dataframe(df.head())
    st.write(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader("Missing values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.bar_chart(missing[missing > 0])
    else:
        st.write("No missing values detected.")

    st.subheader("Value counts")
    if all(col in df.columns for col in ["Survived", "Pclass", "Sex", "Embarked"]):
        for col in ["Survived", "Pclass", "Sex", "Embarked"]:
            st.write(f"**{col} distribution**")
            counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
            st.dataframe(counts)

    processed = preprocess_data(df)
    X, y = prepare_features(processed)

    st.subheader("Preprocessed features")
    st.dataframe(processed.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.history = None
        st.session_state.scaler = None

    if st.button("Train model"):
        with st.spinner("Training model, please wait..."):
            model = build_model(X_train.shape[1])
            history = train_model(model, X_train_scaled, y_train, epochs, batch_size)
            st.session_state.model = model
            st.session_state.history = history
            st.session_state.scaler = scaler
            st.success("Model training completed.")

    if st.session_state.model is not None:
        st.subheader("Model performance")
        plot_history(st.session_state.history)

        y_pred = (st.session_state.model.predict(X_test_scaled) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Test accuracy", f"{accuracy:.4f}")
        st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

        if st.button("Predict passenger survival"):
            passenger_data = {
                "Pclass": pclass,
                "Sex": 0 if sex == "male" else 1,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": {"S": 0, "C": 1, "Q": 2}[embarked],
                "FamilySize": sibsp + parch + 1,
                "IsAlone": 1 if sibsp + parch == 0 else 0,
            }
            probability = predict_passenger(st.session_state.model, st.session_state.scaler, passenger_data)
            st.write(f"Predicted survival probability: {probability:.2%}")
            st.write(
                "This passenger is predicted to "
                + ("survive." if probability >= 0.5 else "not survive.")
            )
    else:
        st.warning("Train the model first to see performance and make predictions.")

