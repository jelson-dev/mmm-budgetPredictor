import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pymc_marketing.mmm import MMM

# Define the color palette
COLOR_SCHEME = {
    "primary": "#3D77FD",
    "secondary": "#112649",
    "tertiary": "#071429",
    "quaternary": "#FFF9F5",
    "highlight": "#FFBC96",
}

# Streamlit App Styling
logo = "/Users/jonathanelson/MMM/Budget Allocation Testing/Bind Logo Full Blue no bg.png" 
st.markdown(
    f"""
    <style>
    body {{
        background-color: {COLOR_SCHEME['secondary']};
        color: {COLOR_SCHEME['quaternary']};
    }}
    .stApp {{
        background-color: {COLOR_SCHEME['secondary']};
        color: {COLOR_SCHEME['quaternary']};
    }}
    h1, h2, h3, {{
        color: {COLOR_SCHEME['quaternary']};
    }}
    .stButton > button {{
        background-color: {COLOR_SCHEME['highlight']};
        color: {COLOR_SCHEME['secondary']};
        border-radius: 5px;
    }}
    .stsuccess {{
        background-color: {COLOR_SCHEME['highlight']};
        color: {COLOR_SCHEME['secondary']};
        border-radius: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# MatplotLib styling
custom_colors = ["#3D77FD", "#FFBC96", "#071429"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

# Title and Description
st.image(logo)
st.title('''Budget Allocation Predictor''')
st.write('''
This app predicts optimal budget allocations across marketing channels based on a pre-fitted model.''')

# Path to Pre-Trained Model
MODEL_PATH = "/Users/jonathanelson/MMM/Budget Allocation Testing/model.nc"  # Update this path to the actual model file location

try:
    st.write("Loading pre-trained model...")
    model = MMM.load(MODEL_PATH)
    st.success("Model loaded successfully!")
    
     # data analysis
    st.subheader("Data Analysis")
    st.write("Reponse curves can be used to indicate each channels point of diminishing returns.")
    if st.button("Plot Response Curves"):
        results = model.plot_direct_contribution_curves()

        st.write("Response Curves")
        st.write(results)

    # Predictions
    st.subheader("Make a prediction")

    # Define Total Budget and Period
    st.write("Define Budget and Period:")
    total_budget = st.number_input(':grey[Total Budget (£)]', min_value=100, step=1, value=50000)
    time_period = st.number_input(':grey[Time Period (days)]', min_value=1, step=1, value=90)

    # Define Channels and Bounds
    st.write("Define Channels and Bounds:")

    # Fixed number of channels
    channel_options = ["google_ads_c", "facebook_ads_c", "amazon_ads_c", "microsoft_ads_c"]
    num_channels = 4  # Fixed number of channels
    channel_data = {}  # Initialize channel data dictionary

    # Collect channel data
    for i in range(num_channels):
        channel_name = channel_options[i]  # Predefined channel names
        st.write(f"Channel {i + 1}: **{channel_name}**")
    
    # Require minimum and maximum budget input
        min_bound = st.number_input(
            f':grey[Minimum Budget for {channel_name} (£)]', 
            min_value=0, 
            max_value=total_budget,
            step=1, 
            key=f"min_{i}"
        )
        max_bound = st.number_input(
            f':grey[Maximum Budget for {channel_name} (£)]', 
            min_value=min_bound + 1,  # Ensure max > min
            max_value=total_budget,
            step=1,
            value=total_budget, 
            key=f"max_{i}"
        )
    
        # Store bounds for the channel
        channel_data[channel_name] = [min_bound, max_bound]

    # Button to trigger predictions
    if st.button("Predict"):
    # Ensure the input is ready for the optimizer
        if total_budget <= 0 or time_period <= 0:
            st.error("Please provide valid Total Budget and Time Period.")
        elif not all(bounds[1] > bounds[0] for bounds in channel_data.values()):
            st.error("Ensure all maximum budgets are greater than minimum budgets.")
        else:
        # Perform budget allocation
            contributions = model.allocate_budget_to_maximize_response(
            budget=total_budget,
            num_periods=time_period,
            time_granularity="daily",
            budget_bounds=channel_data,
        )
        
        # Visualize Contributions
        st.subheader("Channel Contribution Visualization")
        fig, ax = model.plot_budget_allocation(samples=contributions, figsize=(12, 8))
        ax.set_title("Response vs spent per channel", fontsize=18, fontweight="bold")
        st.pyplot(fig)
        

        spend = model.optimal_allocation_dict
        st.subheader("Optimized Budget Allocation")
        st.write(spend)
        st.caption("Disclaimer; The predictions and insights provided by this model are based on historical data and the assumptions made during its development. While every effort has been made to ensure accuracy, the results should be interpreted with caution and used as a guide rather than a definitive outcome. Factors not included in the model, such as unforeseen market changes, external disruptions, or inaccuracies in the input data, may influence the actual results. Users are advised to combine these predictions with domain expertise and other decision-making tools. The creators of this model do not accept liability for decisions made based on these predictions.")




except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please check the file path and try again.")
