
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

    "Royal Enfield": [ "Himalayan","Himalayan 411","Himalayan 450","Classic 350","Classic 500","Classic 650","Bullet 350","Bullet 500","Bullet Electra","Meteor 350","Hunter 350",
                      "Continental GT 535","Continental GT 650","Interceptor 650","Super Meteor 650","Shotgun 650","Scram 411","Scram 440","Goan Classic 350","Guerrilla 450","Bear 650",
                       "Thunderbird 350","Thunderbird 500","Thunderbird X 350","Thunderbird X 500","Electra 350","Classic Signals 350","Classic Stealth Black","Classic Chrome",
                       "Classic Redditch","Classic Halcyon","Classic Heritage Premium","Classic 650 Twin"
    ],

    "Hero": ["Splendor Plus","Splendor Plus XTEC","Splendor Plus XTEC 2.0","Splendor Plus Flex Fuel","Splendor+","Splendor iSmart","Splendor iSmart 110","HF Deluxe","HF Deluxe Flex Fuel",
             "HF Deluxe i3S","HF 100","Passion Pro","Passion Plus","Passion XPro","Super Splendor","Super Splendor XTEC","Glamour","Glamour Xtec","Glamour X","Glamour XTEC","Glamour Blaze",
             "Xtreme 125R","Xtreme 160R","Xtreme 160R 4V","Xtreme 200R","Xtreme 200S","Xtreme 250R","Xpulse 200","Xpulse 200 4V","Xpulse 200T","Xpulse 210","Xpulse 210 Pro","Karizma ZMR","Karizma XMR",
             "Karizma XMR 250","Mavrick 440","Mavrick 440 Pro","Destini 125","Destini Prime","Pleasure","Pleasure Plus","Pleasure Plus XTEC","Maestro","Maestro Edge","Maestro Edge 125",
             "Maestro Edge 110","Duet","Xoom 110","Xoom 110 Combat Edition","Xoom 125","Xoom 125R","Xoom 160","Vida V1","Vida V1 Plus","Vida V1 Pro","Vida V2","Vida VX2"
    ],

    "Honda": ["Shine","Shine 100","Shine 125","Shine 100 DX","SP125","SP125 Sports Edition","SP160","Unicorn","CB Unicorn 160","Livo","Dream Yuga","Dream Neo","CD 110 Dream","Hornet 2.0",
              "CB200X","CB300F","CB300R","CB350","CB350RS","CB350 H'ness","Hness CB350","CB350 DLX","CB350 Legacy Edition","CB650R","CBR650R","CBR1000RR-R Fireblade",
              "CBR1000RR-R Fireblade SP","Africa Twin","Gold Wing","Activa 3G","Activa 4G","Activa 5G","Activa 6G","Activa 110","Activa 125","Activa e","Activa e:",
              "Dio","Dio 110","Dio 125","Dio Sports","Grazia","Grazia 125","Aviator","Cliq",
              "Navi","PCX","QC1","QC1 Electric"
    ],

    "Bajaj": ["Pulsar 125","Pulsar 125 Neon","Pulsar 150","Pulsar 150 Neon","Pulsar N150","Pulsar N160","Pulsar N160 S","Pulsar N160 SS","Pulsar N160 Dual Channel ABS","Pulsar NS160",
              "Pulsar NS200","Pulsar NS400Z","Pulsar N250","Pulsar F250","Pulsar RS200","Pulsar 180","Pulsar 180F","Pulsar 200NS","Pulsar 220F","Pulsar 220","Pulsar AS150",
              "Pulsar AS200","Pulsar RS200","Dominar 250","Dominar 400","Avenger 160 Street","Avenger Street 160","Avenger Street 180",
              "Avenger Cruise 220","Avenger 220","Avenger 400","Platina 100","Platina 110","Platina 110 ABS","CT 100","CT 110",
              "CT 110X","CT 125X","Discover 100","Discover 110","Discover 125","Discover 150","V15","V12","Boxer","Chetak",
              "Chetak Premium","Chetak Urbane","Chetak 2901","Chetak 3001","Chetak C3001","Chetak C3501","Chetak C3502","Chetak C3503","Freedom 125","Freedom 125 NG04",
              "Freedom 125 CNG"
    ],

    "TVS": [
        "Sport", "Sport ES", "Star City", "Star City Plus", "Radeon", "Victor", "Victor Premium", "Apache RTR 160",
        "Apache RTR 160 2V", "Apache RTR 160 4V", "Apache RTR 180", "Apache RTR 200 4V", "Apache RTR 310", "Apache RR 310", "Apache RTX 300",
        "Apache RR 450", "Raider 125", "Raider 125 SSE", "Raider iGO", "Ronin", "Fiero", "Fiero 125", "Ntorq 125", "Ntorq 125 Race Edition",
        "Ntorq 125 Race XP", "Ntorq 125 SuperSquad", "Jupiter", "Jupiter 110", "Jupiter 125", "Jupiter CNG", "Jupiter ZX", "Jupiter Classic", "Zest 110", "Scooty Pep Plus",
        "Scooty Zest", "Wego", "Wego 110", "XL100", "XL 100", "XL100 Comfort", "iQube", "iQube ST", "iQube 2.2", "iQube 3.4", "iQube 5.3", "Orbiter",
        "Orbiter Electric"
    ],

    "Yamaha": [
        "FZ", "FZ-S", "FZ-S FI", "FZ-S V2", "FZ-S V3", "FZ-S V4", "FZ-X", "FZ-X Chrome", "FZ-X Hybrid", "FZ Blue Flex", "Fazer", "Fazer FI",
        "Fazer 25", "MT-15", "MT-15 V2", "MT-15 V2 Deluxe", "R15", "R15 V2", "R15 V3", "R15 V4", "R15M", "R15S",
        "R3", "R7", "R1", "R1M", "Fascino", "Fascino 125", "Fascino 125 FI", "Fascino 125 Hybrid", "Ray", "Ray Z", "Ray ZR",
        "RayZR 125", "RayZR 125 FI Hybrid", "Aerox 155", "Aerox 155 Version S", "Aerox 155 Monster Energy", "Aerox-E", "NMax 155", "FZ25", "FZS 25", "SZ-RR",
        "Saluto", "Saluto RX", "SS125", "Libero", "Crux", "RX100", "RX135", "RXZ", "FZS-FI Hybrid"
    ],

    "Suzuki": [
        "Access 125", "Access 125 Special Edition", "Access 125 Ride Connect", "Access Electric", "Burgman Street", "Burgman Street 125", "Burgman Street EX", "Burgman Electric", "Avenis 125",
        "Avenis 125 Race Edition", "Gixxer", "Gixxer 150", "Gixxer SF", "Gixxer SF 150", "Gixxer 250", "Gixxer SF 250", "V-Strom SX", "V-Strom 650",
        "Hayabusa", "Katana", "Intruder 150", "GS150R", "Slingshot", "Zeus", "Lets", "Swish", "Burgman Street Electric"
    ],

    "KTM": [
        "125 Duke", "200 Duke", "250 Duke", "390 Duke", "790 Duke", "890 Duke", "1290 Super Duke R", "RC 125", "RC 200", "RC 390", "RC 490",
        "250 Adventure", "390 Adventure", "390 Adventure S", "390 Adventure R", "390 Enduro R", "390 SMC R", "790 Adventure", "890 Adventure",
        "1290 Super Adventure", "450 Rally"
    ],

    "Jawa": [ "Jawa", "Jawa Classic", "Jawa Forty Two", "Jawa 42", "Jawa 42 Bobber", "Jawa 42 FJ", "Jawa Perak", "Jawa 350", "Jawa 350 OHC"
    ],

    "Yezdi": [
        "Yezdi Roadster", "Yezdi Scrambler", "Yezdi Adventure", "Yezdi Roadster 334", "Yezdi Scrambler 334"
    ],

    "Aprilia": [
        "SR 125", "SR 125 hp.e", "SR 150", "SR 160", "SR 175 hp.e", "SR GT 200", "SXR 160", "SXR 125", "RS 457", "RS 660", "RSV4", "Tuono 457",
        "Tuono 660", "Tuareg 660", "Tuono V4"
    ],

    "BMW": [
        "G 310 R", "G 310 GS", "G 310 RR", "F 450 GS", "F 750 GS", "F 850 GS", "F 900 R", "F 900 XR", "F 900 GS", "R 1250 GS", "R 1300 GS",
        "R 1300 GSA", "S 1000 RR", "S 1000 R", "S 1000 XR", "M 1000 RR", "M 1000 R", "CE 04"
    ],

    "Kawasaki": [
        "Ninja 300", "Ninja 400", "Ninja 500", "Ninja 650", "Ninja 1000SX", "Ninja ZX-4R", "Ninja ZX-6R", "Ninja ZX-10R", "Ninja ZX-10RR", "Z650", "Z650RS", "Z900", "Z900 SE",
        "Z H2", "Versys 650", "Versys 1000", "KLX230", "KLX230R S", "KX250", "W175", "Eliminator 400"
    ],

    "Triumph": [
        "Speed 400", "Scrambler 400X", "Speed T4", "Speed 900", "Trident 660", "Tiger Sport 660", "Tiger Sport 800", "Tiger 900", "Tiger 1200", "Street Triple", "Street Triple R", 
        "Street Triple RS", "Bonneville T100", "Bonneville T120", "Scrambler 900", "Rocket 3", "Daytona 660"
    ],

    "Harley-Davidson": [
        "X440", "X350", "Street 750", "Street Rod", "Iron 883", "Forty Eight", "Fat Bob", "Fat Boy", "Low Rider", "Sportster S", "Nightster", "Pan America",
        "Pan America 1250", "Road King", "Street Glide", "Road Glide"
    ],

    "Ducati": [
        "Monster", "Monster 937", "Scrambler Icon", "Scrambler Full Throttle", "Scrambler Nightshift", "Panigale V2", "Panigale V4", "Panigale V4 R", "Streetfighter V2", "Streetfighter V4",
        "Multistrada V2", "Multistrada V4", "Diavel V4", "Hypermotard 950", "DesertX"
    ],

    "Ola Electric": [
        "S1", "S1 Air", "S1 Pro", "S1 Pro Gen 2", "S1 X", "S1 X Plus", "S1 Z", "S1 Z+", "Gig", "Gig Plus", "Roadster",
        "Roadster X", "Roadster Pro", "Roadster X Plus"
    ],

    "Ather": [
        "450X", "450S", "450 Apex", "450 Apex Limited Edition", "450 Gen 3", "450S Gen 3", "Rizta", "Rizta S", "Rizta Z",
        "Rizta S 2.9", "Rizta Z 2.9", "Rizta Z 3.7"
    ],

    "Ultraviolette": [
        "F77", "F77 Original", "F77 Recon", "F77 SuperStreet", "F77 Mach 2", "F77 Mach 2 Recon", "F77 Mach 2 Recon Alpha", "X47", "Tesseract"
    ],

    "Mahindra": [
        "Mojo 300", "Mojo UT 300", "Centuro", "Gusto", "Gusto 110", "Gusto 125", "Rodeo RZ"
    ],

    "Vespa": [
        "VXL 125", "VXL 150", "SXL 125", "SXL 150", "ZX 125", "Elegante 125", "Elegante 150", "Vespa 125", "Vespa 150", "Vespa S", "Vespa Tech"
    ],

    "Piaggio": [
        "Aprilia SR 125", "Aprilia SR 160", "Aprilia SXR 160", "Vespa VXL", "Vespa SXL", "Vespa Elegante", "Liberty 125"
    ],

    "Revolt Motors": [
        "RV400", "RV400 Stealth Black", "RV1", "RV1 Plus", "RVX"
    ],

    "Hero Electric": [
        "Optima", "Optima CX", "Optima HX", "Photon", "Photon LP", "NYX", "NYX HX", "AE-8", "Dash"
    ],

    "Simple Energy": [
        "One", "One Gen 1", "One Gen 2", "Dot One"
    ],

    "Vida": [
        "V1", "V1 Plus", "V1 Pro", "V2", "V2 Lite", "V2 Plus", "V2 Pro", "VX2", "VX2 Go", "VX2 Go 2.2 kWh", "VX2 Go 3.4 kWh",
        "VX2 Plus", "VX2 Plus 4.4 kWh"
    ],

    "Okinawa": [
        "R30", "Ridge", "Ridge Plus", "Praise", "Praise Pro", "i-Praise", "Okhi90", "Cruiser", "Lite", "Dual", "Dual 100", "C90"
    ],

    "Ampere": [
        "Zeal", "Zeal Ex", "Magnus", "Magnus EX", "Magnus LT", "Magnus Neo", "Primus", "Nexus", "Reo", "Reo Plus"
    ],

    "PURE EV": [
        "Epluto", "Epluto 7G", "Etrance", "Etrance Neo", "Etryst 350", "EcoDryft", "EcoDryft 350"
    ],

    "Komaki": [
        "SE", "SE Pro", "Flora", "XGT VP", "XGT X One", "XGT KM", "X2 Vogue", "Ranger", "Venice", "LY Pro", "MG Pro", "DT 3000", "CAT 2.0"
    ],

    "Joy e-bike": [
        "Gen Next Nanu", "Gen Next Nanu Plus", "Wolf", "Wolf Plus", "Glob", "Wolf+", "Beast", "Monster", "Mihos"
    ],

    "Lectrix EV": [
        "LXS G2.0", "LXS G3.0", "LXS 2.0", "SX25", "SX25 Max", "LXS"
    ],

    "Zelio": [
        "Gracy", "Gracy i", "Gracy Plus", "Eeva", "Eeva ZX", "Eeva ZX Plus", "X-Men", "X-Men 2.0", "Legender"
    ],

    "Bounce": [
        "Infinity E1", "Infinity E1+", "Infinity E1X"
    ],

    "River": [
        "Indie"
    ],

    "Tork Motors": [ "Kratos", "Kratos R", "Kratos R Urban", "Kratos R Urban Trim"
    ],

    "Oben": [
        "Rorr", "Rorr EZ", "Rorr EZ Urban", "Rorr EZ Premium"
    ],

    "Matter": [
        "Aera","Aera 5000"
    ],

    "Raptee": [
        "T30"
    ],

    "Bgauss": [
        "A2", "A2 Plus", "C12", "C12i", "D15", "RUV350"
    ],

    "Okaya": [
        "Faast F2F", "Faast F3", "Faast F4", "Faast F2B", "Motofaast"
    ],

    "Kinetic Green": [
        "Zing", "Zing HSS", "Flex", "Flex Electric", "E-Luna", "E-Luna X2"
    ],

    "Yulu": [
        "Miracle",
        "Dex GR"
    ],

    "Keeway": [
        "K-Light 250V", "V302C", "Vieste 300", "Sixties 300i", "SR125", "Hypevolt-R"
    ],

    "Benelli": [
        "Imperiale 400", "Leoncino 500", "TRK 502", "TRK 502X", "302S", "302R", "502C", "752S"
    ],

    "QJ Motor": [
        "SRK 400", "SRK 700", "SRV 300", "SRV 550", "SRC 250", "SRC 500", "SRT 550", "SRT 700", "Flash 500"
    ],

    "CFMoto": [
        "300NK", "300SR", "650NK", "650GT", "650MT", "700CL-X", "800MT", "450MT", "450SR"
    ],

    "Husqvarna": [
        "Svartpilen 125", "Svartpilen 250", "Svartpilen 401", "Vitpilen 250", "Vitpilen 401", "Norden 901"
    ],

    "GasGas": [
        "SM 450F", "ES 700", "EC 250", "EC 300", "MC 250", "MC 350"
    ],

    "Moto Guzzi": [
        "V7", "V7 Stone", "V85 TT", "V100 Mandello"
    ],

    "MV Agusta": [
        "Brutale 800", "Dragster 800", "F3 800", "Turismo Veloce", "Superveloce 800"
    ],

    "Norton": [
        "400", "650", "650 Commando", "V4SV", "V4CR"
    ],

    "BSA": [
        "Gold Star",
        "Gold Star 650"
    ],

    "Indian": [
        "Scout", "Scout Bobber", "FTR", "Chief", "Chief Dark Horse", "Super Chief", "Chieftain", "Roadmaster"
    ],

    "Can-Am": [
        "Pulse", "Origin", "Ryker", "Spyder"
    ],

    "Niu": [
        "NQi", "NQi GTS", "MQi", "MQi GT", "RQi Sport"
    ],

    "Gogoro": [
        "2 Series", "2 Plus", "2 Delight", "S1", "Viva", "Viva Mix"
    ],

    "Segway": [
        "E110S", "E125S", "E300SE", "E300SE Pro", "X260"
    ],

    "TaoTao": [
        "TBR7", "DBX1", "ATA110"
    ],

    "Zontes": [
        "350R", "350T", "350X", "GK350", "350D", "ZT155"
    ],

    "Voge": [
        "300R", "300AC", "300DS", "500DS", "650DS", "650DSX", "650R", "SR4 Max"
    ],

    "Lambretta": [
        "V200", "V200 Special", "X200", "G350"
    ],

    "Lifan": [
        "KPT 400", "KPR 200", "KPX 250", "KPT 150"
    ],

    "Loncin": [
        "CR3", "CR6", "LX300", "Voge 300R"
    ],

    "Zongshen": [
        "RX3", "RX4", "RX6", "Cyclone 400"
    ],

    "Avon": [
        "E-Scoot", "E-Star", "E-Mate", "E-Plus"
    ],

    "Avon Cycles": [
        "E-Scoot", "E-Star", "E-Mate"
    ],

    "WardWizard": [
        "Joy E-Bike Beast", "Joy E-Bike Mihos", "Joy E-Bike Wolf", "Joy E-Bike Gen Next Nanu"
    ],

    "FAME": [
        "FAME II", "FAME Electric"
    ],

    "Tunwal": [
        "Mini Sports", "Strom Advance", "Sport 63", "Lithino Li 3", "Lithino-Li 2"
    ],

    "Srivaru Motors": [
        "Prana", "Prana Grand"
    ],

    "Raptee": [
        "T30"
    ],

    "Ather Energy": [
        "450X", "450S", "450 Apex", "Rizta", "Rizta S", "Rizta Z"
    ],

    "River Mobility": [
        "Indie"
    ],

    "Avore": [
        "EX1", "EX2", "EX2 S"
    ]
}

Make = st.selectbox("Make", list(two_wheelers.keys()))
Model_Variant = st.selectbox("Model Variant", two_wheelers.get(Make, []))
st.write(f"Selected: **{Make} - {Model_Variant}**")

current_year = datetime.now().year
manufacturing_year = st.selectbox("Year of Manufacture", list(range(current_year, 1999, -1)))
'Owner Count' = st.number_input("Number of Owners", min_value=1, step=1)

Age_at_Purchase = current_year - manufacturing_year
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
        base_prediction = model.predict(input_df)[0]
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
