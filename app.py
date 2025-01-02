import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load the model
model = joblib.load('./xgb_model_recovery.pkl')


# Define feature names
feature_names = ['拔管时间', '血管超声', '出量2：尿量ml', 'pco2_post'
                 , '腹部超声/CT', '年龄', '顺式阿曲库铵'
                 ,'淋巴细胞数1_post-3.2*10^9/L', '麻醉时长min', '体重kg'
                 , 'BE(B)_post','_postCaprini评分', '身高cm'
                 , '基础收缩压', '丙泊酚mg', '术中最高心率次/分', 'BE(B)',
                '追加止痛药（无=0,有=1）', '白球比1.2-2.4', '高密度胆固醇'
       ]

# Streamlit user interface
st.title("Recovery Predictor")

var1 = st.number_input("拔管时间:", min_value=None, max_value=None, value=0.000)
var2 = st.number_input("血管超声:", min_value=None, max_value=None, value=0.000)
var3 = st.number_input("出量2：尿量ml:", min_value=None, max_value=None, value=0.000)
var4 = st.number_input("pco2_post:", min_value=None, max_value=None, value=0.000)
var5 = st.number_input("腹部超声/CT:", min_value=None, max_value=None, value=0.000)
var6 = st.number_input("年龄:", min_value=None, max_value=None, value=0.000)
var7 = st.number_input("顺式阿曲库铵:", min_value=None, max_value=None, value=0.000)
var8 = st.number_input("淋巴细胞数1_post-3.2*10^9/L:", min_value=None, max_value=None, value=0.000)
var9 = st.number_input("麻醉时长min:", min_value=None, max_value=None, value=0.000)
var10 = st.number_input("体重kg:", min_value=None, max_value=None, value=0.000)
var11= st.number_input("BE(B)_post:", min_value=None, max_value=None, value=0.000)
var12 = st.number_input("_postCaprini评分:", min_value=None, max_value=None, value=0.000)
var13 = st.sidebar.slider("身高cm",min_value=10, max_value=260, value=150,step=1)
var14 = st.number_input("基础收缩压:", min_value=None, max_value=None, value=0.000)
var15 = st.number_input("丙泊酚mg:", min_value=None, max_value=None, value=0.000)
var16 = st.number_input("术中最高心率次/分:", min_value=None, max_value=None, value=0.000)
var17 = st.number_input("BE(B):", min_value=None, max_value=None, value=0.000)
var18 = st.number_input("追加止痛药（无=0,有=1）:", min_value=0, max_value=1, value=0)
var19 = st.number_input("白球比1.2-2.4:", min_value=None, max_value=None, value=0.000)
var20 = st.number_input("高密度胆固醇:", min_value=None, max_value=None, value=0.000)

var1 = (var1-50.822222) / 19.786454
var2 = (var2 + 0.618182) / 0.580554
var3 = (var3 - 385.656566) / 347.521561



feature_values = [var1, var2, var3
                  , var4, var5, var6, var7, var8
                  , var9, var10, var11, var12
                  , var13, var14, var15, var16
                  , var17, var18, var19, var20]



# Process inputs and make predictions
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # # Calculate SHAP values and display force plot
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    # st.image("shap_force_plot.png")
