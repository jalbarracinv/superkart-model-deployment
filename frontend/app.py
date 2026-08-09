from io import BytesIO

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://backend:7860"
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒", layout="wide")
st.title("SuperKart Sales Prediction")
st.write("Estimate product sales for one record or upload a CSV file for batch prediction.")

st.subheader("Online prediction")
with st.form("prediction_form"):
    left, middle, right = st.columns(3)
    with left:
        product_weight = st.number_input("Product weight", min_value=0.0, value=12.66, step=0.01)
        sugar_content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Allocated area", min_value=0.0, value=0.027, step=0.001, format="%.3f")
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08, step=0.01)
    with middle:
        store_size = st.selectbox("Store size", ["Medium", "High", "Small"])
        city_type = st.selectbox("Store location city type", ["Tier 2", "Tier 1", "Tier 3"])
        store_type = st.selectbox(
            "Store type",
            ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"],
        )
    with right:
        product_prefix = st.selectbox("Product ID prefix", ["FD", "DR", "NC"])
        store_age = st.number_input("Store age in years", min_value=0, max_value=100, value=16, step=1)
        product_category = st.selectbox(
            "Product type category", ["Perishables", "Non Perishables"]
        )
    submitted = st.form_submit_button("Predict sales")

if submitted:
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_prefix,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_category,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
        response.raise_for_status()
        prediction = response.json()["prediction"]
        st.success(f"Estimated product sales: {prediction:,.2f}")
    except (requests.RequestException, KeyError, ValueError) as error:
        st.error(f"The prediction could not be completed: {error}")

st.divider()
st.subheader("Batch prediction")
st.caption("Upload a CSV file containing the ten model features. Extra columns are ignored.")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        preview = pd.read_csv(BytesIO(file_bytes))
        missing_columns = [column for column in FEATURES if column not in preview.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
        else:
            st.dataframe(preview.head(), use_container_width=True)
            if st.button("Run batch prediction"):
                files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch", files=files, timeout=120
                )
                response.raise_for_status()
                predictions = response.json()
                result = preview.copy()
                result["Predicted_Sales"] = [
                    predictions[str(index)] for index in range(len(result))
                ]
                st.success(f"Completed {len(result)} predictions.")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
    except (pd.errors.ParserError, UnicodeDecodeError, requests.RequestException, KeyError, ValueError) as error:
        st.error(f"The batch request could not be completed: {error}")
