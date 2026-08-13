
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("bike_resell_price_prediction_model_v1_0.joblib")

model = load_model()

st.title("Reselling Bike Price Prediction App")
st.write("This tool predicts the price of an used 2 wheeler bike Re-Selling amount")

st.subheader("Enter the Vehicle Details:")

Category = st.selectbox("Category", ['Bike', 'EV', 'Scooter'])

two_wheelers = {
    "Royal Enfield": ["Himalayan 450", "Classic 350", "Bullet 350", "Meteor 350",
                       "Hunter 350", "Continental GT 650", "Interceptor 650",
                       "Super Meteor 650", "Shotgun 650", "Scram 440", "Goan Classic 350"],
    "Hero": ["Splendor Plus", "Splendor Plus XTEC", "Splendor Plus Flex Fuel",
             "HF Deluxe", "HF Deluxe Flex Fuel", "Passion Pro", "Super Splendor",
             "Glamour", "Xtreme 125R", "Xtreme 160R", "Xtreme 250R", "Xpulse 200 4V"],
    "Honda": ["Shine", "SP125", "Unicorn", "SP160", "Hornet 2.0", "CB350", "CB350RS",
              "Activa 110", "Activa 125", "Activa 6G", "Dio"],
    "Bajaj": ["Pulsar 150", "Pulsar N160", "Pulsar N160 S", "Pulsar N250", "Pulsar F250",
              "Pulsar NS200", "Pulsar NS400Z", "Pulsar 220F", "Dominar 400",
              "Avenger Cruise 220", "Chetak (Electric)", "Freedom 125 (CNG)", "Platina 110"],
    "TVS": ["Raider 125", "Raider 125 SSE", "Apache RTR 160 4V", "Apache RTR 180",
            "Apache RTR 200 4V", "Apache RTR 310", "Apache RR 310", "Ntorq 125",
            "Jupiter 110", "Jupiter CNG", "Ronin", "iQube (Electric)", "XL 100"],
    "Yamaha": ["FZ-S FI", "FZ-X", "MT-15", "R15 V4", "Fascino 125", "Aerox 155", "RayZR 125"],
    "Suzuki": ["Access 125", "Burgman Street 125", "Gixxer 150", "Gixxer SF 250", "V-Strom SX"],
    "KTM": ["Duke 200", "Duke 250", "Duke 390", "RC 200", "RC 390", "390 Adventure"],
    "Jawa": ["Jawa 42", "Jawa 42 Bobber", "Perak"],
    "Yezdi": ["Yezdi Adventure", "Yezdi Roadster", "Yezdi Scrambler"],
    "Aprilia": ["SR 160", "RS 457", "Tuono 457"],
    "BMW": ["G 310 R", "G 310 GS", "F 900 R"],
    "Kawasaki": ["Ninja 300", "Ninja 400", "Z650"],
    "Triumph": ["Speed 400", "Scrambler 400X"],
    "Harley-Davidson": ["X440", "Street 750"],
    "Ducati": ["Monster", "Panigale V2"],
    "Ola Electric": ["S1 Pro", "S1 Air", "S1 X"],
    "Ather": ["450X", "450S", "Rizta"],
    "Ultraviolette": ["F77", "X47"],
    "Mahindra": ["Mojo 300"],
}

Make = st.selectbox("Make", list(two_wheelers.keys()))
Model_Variant = st.selectbox("Model Variant", two_wheelers.get(Make, []))
st.write(f"Selected: **{Make} - {Model_Variant}**")

current_year = datetime.now().year
manufacturing_year = st.selectbox("Year of Manufacture", list(range(current_year, 1999, -1)))
Owner_Count = st.number_input("Number of Owners", min_value=1, step=1)

age_at_purchase = current_year - manufacturing_year
st.write(f"Vehicle Age: **{age_at_purchase} years**")

# ---- Fields NOT present in the trained model ----
# Collected separately and used only for a post-prediction adjustment
st.subheader("Additional Details (used to fine-tune the predicted price)")

Odometer_Value = st.number_input("Odometer Reading (km)", min_value=0, step=100)
HP_Status = st.selectbox("Hypothecation (HP) Status", ["No Active HP", "Active HP"])
Loan_Status = st.selectbox("Active Loan Status", ["No Active Loan", "Active Loan"])

# ---------------------------------------------------------
# WEIGHTAGE / ADJUSTMENT LOGIC
# These are rule-based multipliers applied AFTER model.predict()
# Tune these numbers based on domain knowledge / market data
# ---------------------------------------------------------

def odometer_adjustment(km):
    """Higher km -> higher deduction from predicted price."""
    if km <= 10000:
        return 1.00       # no deduction
    elif km <= 30000:
        return 0.97        # -3%
    elif km <= 60000:
        return 0.92        # -8%
    elif km <= 100000:
        return 0.85        # -15%
    else:
        return 0.75         # -25%

def hp_adjustment(status):
    """Active hypothecation lowers resale value due to paperwork/transfer risk."""
    return 0.90 if status == "Active HP" else 1.00   # -10% if active

def loan_adjustment(status):
    """Active loan lowers resale value (buyer risk, NOC pending)."""
    return 0.92 if status == "Active Loan" else 1.00  # -8% if active

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button("Predict Resale Price"):

    # Build the input row EXACTLY as the model expects
    # (adjust column names/order to match your training pipeline)
    input_df = pd.DataFrame([{
        "Category": Category,
        "Make": Make,
        "Model_Variant": Model_Variant,
        "manufacturing_year": manufacturing_year,
        "Owner_Count": Owner_Count,
        "age_at_purchase": age_at_purchase,
    }])

    try:
        base_prediction = saved_model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Prediction failed — check that input_df columns match the model's training features. Error: {e}")
        st.stop()

    # Apply the three rule-based adjustments sequentially
    adjusted_price = (
        base_prediction
        * odometer_adjustment(Odometer_Value)
        * hp_adjustment(HP_Status)
        * loan_adjustment(Loan_Status)
    )

    st.subheader("Prediction Result")
    st.write(f"Base Model Prediction: ₹{base_prediction:,.0f}")
    st.write(f"Adjustment for Odometer: x{odometer_adjustment(Odometer_Value)}")
    st.write(f"Adjustment for HP Status: x{hp_adjustment(HP_Status)}")
    st.write(f"Adjustment for Loan Status: x{loan_adjustment(Loan_Status)}")
    st.success(f"### Final Estimated Resale Price: ₹{adjusted_price:,.0f}")
