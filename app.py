
import streamlit as st
import pandas as pd
import pickle

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()


# ==========================================
# 3. APPLICATION HEADING
# ==========================================

st.title("💎 Diamond Price Prediction")

st.write(
    "Predict diamond prices using Machine Learning "
    "with the K-Nearest Neighbors (KNN) algorithm."
)

st.divider()


# ==========================================
# 4. DIAMOND INPUT DETAILS
# ==========================================

st.subheader("Enter Diamond Details")

col1, col2, col3 = st.columns(3)


# Column 1
with col1:

    carat = st.number_input(
        "Carat Weight",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1
    )

    cut = st.selectbox(
        "Cut Quality",
        ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    )

    color = st.selectbox(
        "Diamond Color",
        ["D", "E", "F", "G", "H", "I", "J"]
    )


# Column 2
with col2:

    clarity = st.selectbox(
        "Clarity",
        [
            "I1", "SI2", "SI1", "VS2",
            "VS1", "VVS2", "VVS1", "IF"
        ]
    )

    depth = st.number_input(
        "Depth (%)",
        min_value=40.0,
        max_value=80.0,
        value=61.5
    )

    table = st.number_input(
        "Table (%)",
        min_value=40.0,
        max_value=80.0,
        value=55.0
    )


# Column 3
with col3:

    x = st.number_input(
        "Length (mm)",
        min_value=0.0,
        max_value=12.0,
        value=4.0
    )

    y = st.number_input(
        "Width (mm)",
        min_value=0.0,
        max_value=12.0,
        value=4.0
    )

    z = st.number_input(
        "Depth (mm)",
        min_value=0.0,
        max_value=12.0,
        value=2.5
    )


st.divider()


# ==========================================
# 5. PREDICTION
# ==========================================

if st.button(
    "Predict Diamond Price",
    type="primary",
    key="predict_diamond"
):

    try:

        # Prepare input data
        input_data = pd.DataFrame([{
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z
        }])

        # Display input data
        st.subheader("Diamond Input Details")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        # Generate prediction
        prediction = model.predict(input_data)

        # Display prediction result
        st.success(
            "Prediction completed successfully!"
        )

        st.metric(
            label="Estimated Diamond Price",
            value=f"${prediction[0]:,.2f}"
        )

        st.info(
            "This is an estimated price generated "
            "by the KNN Machine Learning model."
        )

    except Exception as e:

        st.error("Prediction failed!")

        st.exception(e)


# ==========================================
# 6. FOOTER
# ==========================================

st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-learn, "
    "KNN Regression and Streamlit."
)
