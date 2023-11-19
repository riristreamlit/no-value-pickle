import streamlit as st
import pandas as pd

results_df = pd.DataFrame({
    'Method': ['RandomForestRegressor'],
    'RMSE TRAIN': [0.437184],
    'RMSE TEST': [2.195795],
    'MSE TRAIN': [0.19113],
    'MSE TEST': [4.821516],
    'R-squared TRAIN': [0.999998],
    'R-squared TEST': [0.999956]
})

data = {
    'Y_test': [648.0, 798.0, 355.0, 1086.0, 1160.0, 1154.0, 630.0, 296.0, 1170.0, 596.0, 178.0, 717.0, 989.0,
               539.0, 105.0, 1062.0, 287.0, 530.0, 270.0, 599.0, 794.0, 634.0, 766.0, 228.0, 743.0, 539.0, 451.0,
               357.0, 690.0, 303.0, 742.0, 564.0, 355.0, 831.0, 525.0, 51.0, 457.0, 1142.0, 531.0, 874.0, 343.0,
               282.0, 459.0, 464.0, 500.0, 342.0, 803.0, 725.0, 716.0, 969.0],
    'Y_pred': [649.77, 798.52, 354.80, 1090.04, 1164.44, 1159.19, 630.17, 295.64, 1174.29, 595.34, 179.53, 715.79,
               990.81, 539.30, 105.18, 1064.88, 286.42, 530.61, 269.19, 600.39, 795.76, 634.87, 764.71, 229.06, 744.59,
               539.47, 449.82, 355.33, 689.67, 304.23, 740.99, 564.26, 355.12, 830.51, 525.28, 50.83, 455.10, 1145.00,
               530.61, 874.70, 343.84, 279.86, 459.16, 465.05, 500.66, 340.64, 804.42, 725.00, 715.08, 970.85]
}

df = pd.DataFrame(data)

###

st.set_page_config(
    page_title="Hello",
)

###

st.markdown("# Air BnB Price Prediction and Segmentation! 🏨🛎️📊")

st.markdown("""### Full Presentation [Portfolio](https://bit.ly/Portfolio_Oktober_2023) ❤️""")

st.markdown(
    """
    Thank you for checking out my portfolio! Before exploring this Streamlit app, feel free to take a look at my other projects. 🚀📁
    - 👚 **Development of A Color Mapping and Classification System for Batik Textile from Garut and Solo** [Bachelor Degree's Thesis](https://drive.google.com/file/d/1qY1rlhd6_S6CLhb54X8HYneTGZgYliCv/view?usp=sharing)
    - ⛔ **Machine Learning Model to Predict Hotel Booking Cancellation** [Don't Cancel!](https://bit.ly/DontCancel_RiriRaissa)
    - 📦 **SQL Query Study Case: Revolutionize Sales Unleashing the Power of Product** [SQL Query](https://bit.ly/PortfolioSQL_Nov)
    - 👋 **Hand Sign Detector on MATLAB** [Demo Video](https://bit.ly/ImageProcessing_MATLAB)

    ### Please read below before you get started 👇👇👇
"""
)





st.markdown(
    """
🏡 Air BnB is a popular online marketplace and hospitality service that allows people to rent or lease short-term lodging accommodations.

🏡 The problem is that both hosts looking to rent out their homes and guests looking to purchase are unaware of the market for the properties they are offering.
""")
st.image('4.jpg', caption='Heatmap of All Variable', use_column_width=True)

st.markdown(
    """
The presence of multivariate, rather than multicollinearity, indicates that it is acceptable to proceed with machine learning.

# Model 🤖
# **1. Random Forest Regressor 🌲**
"""
)

st.markdown(
    """
🌲 Using Random Forest, I trained the model to predict prices based on Airbnb's facilities.

🌲 These are the error metrics; it has good RMSE, MSE, and R-squared values. The model didn't experience overfitting or underfitting.
""")

st.image('RandomForestGraph.png', caption='Model Performance', use_column_width=True)

st.markdown("## Evaluation Metriks")


# Display the DataFrame with specified precision
st.table(results_df.style.format(precision=6))

st.markdown("## Prediction Value")

st.dataframe(df, hide_index=True)

###


st.markdown("# **2. K-Means Clustering 🛌**")

st.markdown("""
            
I'm utilizing K-Means Clustering to categorize Airbnb listings into 6 groups based on previously summarized variables.
            🏡✨ This approach allows for a more structured and insightful segmentation of the data 📊🔍
            """)

st.image('5.png', caption='K-Means Clustering', use_column_width=True)
st.image('6.png', caption='Air BnB Market Segmentation', use_column_width=True)

###

st.markdown(
    """
    **👈 Select any pages from the sidebar** to see some demonstration!

    Enjoy! Critiques and suggestions are welcome, you can contact me at:
    - 📨 riri.raissa12@gmail.com
    - 👷 Let's get connected on [LinkedIn](https://www.linkedin.com/in/riri-raissa/)
    - 💻 https://github.com/ririraissa
"""
)