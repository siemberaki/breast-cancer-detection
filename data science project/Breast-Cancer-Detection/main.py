import streamlit as st
from sklearn import datasets
import numpy as np
import breast_cancer_detection


# declaring a variable so that the selected name will be assigned to the variable
dataset_name = st.sidebar.markdown("<h1 style='text-align: center;'>Data Set: Breast Cancer </h1>", unsafe_allow_html=True)
st.sidebar.write("""### Classifier: Logic regression""")

def get_dataset(dataset_name):
    data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y

x , y = get_dataset(dataset_name)
st.sidebar.write("Shape of the dataset: ", x.shape)
st.sidebar.write("Classes of the dataset: ", len(np.unique(y)))

st.markdown("\n### This model is the smaller version of the implementation in my [Kaggle account](https://www.kaggle.com/code/merryzeray/logistic-reg-breast-cancer-detection)\n")
st.markdown("I used the top 7 columns and the accuracy of this model is 64%. The possible results are either Malignant or Benign.\n")
st.sidebar.markdown("The following are real values from dataset:\n")

st.sidebar.markdown("""
| Columns | Benign | Malignant |
|----------|----------|----------|
|Texture worst    | 24.64    | 47.16  |
|Fractal dimension| 0.003586 | 0.002299 |
|Concavity mean   | 0.08005  | 0.13670  |
|Perimeter worst  | 96.05    | 214.0  |
|Concavity worst  | 0.2671   | 0.3442  |
|Perimeter se     | 2.497    | 7.749  |
|Perimeter mean   | 81.09    | 135.70  |

""")
# starting with the model
# Add input components

# Define names for each input
input_names = ["Texture worst", "Fractal dimension", "Concavity mean", "Perimeter worst",
	"Concavity worst", "Perimeter se", "Perimeter mean"]

# Create 7 rows with 2 columns each
for i in range(7):
    container = st.container()

    with container:
        col1, col2 = st.columns(2)

        with col1:
            st.text(input_names[i])

        with col2:
            user_input = st.text_input("", key=f"input_{i+1}")


# Add a button to trigger prediction
if st.button("Predict"):
    inputs = [st.session_state[f"input_{i+1}"] for i in range(7)]
    if all(input_value.strip() for input_value in inputs):  # Check if all inputs have non-empty values
        prediction = breast_cancer_detection.predict_model(inputs)
        st.write("Prediction:", prediction)
    else:
        st.error("Please provide values for all inputs.")



st.sidebar.markdown("<a href = 'mailto:merry0zeray@gmail.com' style='text-align: center;'>Contact me </a>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color:gray'>Prepared by Merry Zeray Semereab</h6>", unsafe_allow_html=True)
