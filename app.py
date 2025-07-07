import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------ Page Styling ------------

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #1b2b23;
        color: #f0f0f0;
    }
    .css-1cpxqw2, .css-1d391kg {
        background-color: #263b32 !important;
        color: #ffffff !important;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3, h4 {
        color: #7ed957;
    }
    .stSidebar {
        background-color: #21362c;
    }
    .stButton > button {
        background-color: #388e3c !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    .stButton > button:hover {
        background-color: #2e7d32 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Data Clean/Preprocessing -----------

@st.cache_data
def preprocessed_data():
    df = pd.read_csv(r"C:\Code Files\Data Files\realtor-data.csv")
    df.drop(['brokered_by', 'status', 'prev_sold_date', 'state', 'street', 'city', 'acre_lot'], axis=1, inplace=True)
    df = df.drop_duplicates().dropna()
    df.rename(columns={"house_size": "sqft"}, inplace=True)

    df = df[df['bed'] <= 20]
    df = df[df['price'] <= 125000000]
    df = df[df['sqft'] <= 60000]
    df = df[df['bath'] <= 30]

    log_features = df.columns
    for col in log_features:
        df[col] = np.log1p(df[col])

    df = df[df['price'] > 5]
    return df

@st.cache_data
def scaled_data():
    df = preprocessed_data()
    x = df.drop('price', axis=1)
    y = df['price']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y, scaler, x.columns

x_scaled, y, scaler, feature_cols = scaled_data()

@st.cache_data
def predict(customer: pd.DataFrame):

    customer_log = np.log1p(customer)
    customer_scaled = scaler.transform(customer_log)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    
    model = LinearRegression(fit_intercept=True, positive=False)
    model.fit(x_train, y_train)

    y_pred_log = model.predict(customer_scaled)
    y_pred = np.expm1(y_pred_log.item()) 

    y_val_pred = model.predict(x_test)
    y_test_exp = np.expm1(y_test)
    y_val_pred_exp = np.expm1(y_val_pred)

    mse = mean_squared_error(y_test_exp, y_val_pred_exp)
    mae = mean_absolute_error(y_test_exp, y_val_pred_exp)
    baseline_mae = mean_absolute_error(y_test_exp, [np.expm1(y_train.mean())] * len(y_test_exp))
    mae_reduction = ((baseline_mae - mae) / baseline_mae) * 100
    r2 = r2_score(y_test_exp, y_val_pred_exp)

    fi = pd.Series(model.coef_, index=feature_cols)

    return y_pred, mse, mae, mae_reduction, r2, fi

# ----------- Sidebar UI -----------

st.sidebar.title("🏠 Property Details")

customer = {
    "bed": st.sidebar.number_input("Bedrooms", min_value=0.0),
    "bath": st.sidebar.number_input("Bathrooms", min_value=0.0),
    "zip_code": st.sidebar.number_input("ZIP Code", min_value=0.0),
    "sqft": st.sidebar.number_input("House Size (Sqft)", min_value=0.0),
}

# ------------ Main Page ------------

if st.sidebar.button("📈 Predict Price"):
    col1, col2, col3 = st.columns(3)
    prediction, mse, mae, mae_reduction, r2, fi = predict(pd.DataFrame([customer]))
    with col1:
        st.metric("💵 Predicted Value of House", f"${prediction:,.0f}")
    with col2:
        st.metric("🎯 Model R² Score", f"{r2 * 100:.2f}%")
    with col3:
        st.metric("📉 MAE vs Baseline", f"${mae:,.0f} (↓ {mae_reduction:.1f}%)")

    st.subheader("📊 Feature Importance")
    if fi is not None and not fi.empty:
        fig = px.bar(
            x=fi.values,
            y=fi.index,
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig.update_layout(
            height=400,
            margin=dict(l=100, r=20, t=30, b=30),
            yaxis=dict(tickfont=dict(size=10)),
            xaxis_title='Importance',
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown("""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            background: #ffffff22;
            border: 2px solid #52796f;
            border-radius: 16px;
            margin: 2rem 0;
        ">
            <h1 style="
                color: #ffffff;
                font-size: 2rem;
                font-weight: 600;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                text-align: center;
                line-height: 1.3;
            ">
                🔍 Complete the form in the sidebar,<br>
                and click <strong>Predict</strong><br>
                to view Home insights.
            </h1>
        </div>
    """, unsafe_allow_html=True)