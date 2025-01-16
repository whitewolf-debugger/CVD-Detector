import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
import base64

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import base64
import streamlit as st

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN
from keras.models import Sequential
from keras.layers import Dense, Input, Bidirectional, LayerNormalization
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Reshape
from keras.optimizers import Adam
from keras import losses
import gc
#---------------Model load and setup part , don't touch : Ritabrata ------------------#
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class CardioRNNCell(Layer, DropoutRNNCell):
    def __init__(
        self,
        units,
        activation="leaky_relu",  # Replaced sigmoid with leaky_relu
        recurrent_activation="tanh",
        use_bias=True,
        kernel_initializer="he_normal",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        implementation = kwargs.pop("implementation", 2)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed=seed)
        self.unit_forget_bias = unit_forget_bias
        self.state_size = [self.units, self.units, self.units]  # Added one more CardioRNN cell for long-term memory
        self.output_size = self.units
        self.implementation = implementation

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return ops.concatenate(
                        [
                            self.bias_initializer(
                                (self.units,), *args, **kwargs
                            ),
                            initializers.get("ones")(
                                (self.units,), *args, **kwargs
                            ),
                            self.bias_initializer(
                                (self.units * 2,), *args, **kwargs
                            ),
                        ]
                    )
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.layer_norm = LayerNormalization()  # Added Layer Normalization
        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + ops.matmul(h_tm1_i, self.recurrent_kernel[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f
            + ops.matmul(
                h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2]
            )
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + ops.matmul(
                h_tm1_c,
                self.recurrent_kernel[:, self.units * 2 : self.units * 3],
            )
        )
        o = self.recurrent_activation(
            x_o
            + ops.matmul(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
        )
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, training=False):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)
        if training and 0.0 < self.dropout < 1.0:
            inputs = inputs * dp_mask
        if training and 0.0 < self.recurrent_dropout < 1.0:
            h_tm1 = h_tm1 * rec_dp_mask
        if self.implementation == 1:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
            k_i, k_f, k_c, k_o = ops.split(self.kernel, 4, axis=1)
            x_i = ops.matmul(inputs_i, k_i)
            x_f = ops.matmul(inputs_f, k_f)
            x_c = ops.matmul(inputs_c, k_c)
            x_o = ops.matmul(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = ops.split(self.bias, 4, axis=0)
                x_i += b_i
                x_f += b_f
                x_c += b_c
                x_o += b_o
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        else:
            z = ops.matmul(inputs, self.kernel)
            z += ops.matmul(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z += self.bias
            z = ops.split(z, 4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)
        h = o * self.activation(c)
        h = self.layer_norm(h)  # Apply Layer Normalization
        return h, [h, c, ops.zeros_like(h)]

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras_export("keras.layers.CardioRNN")
class CardioRNN(RNN):
    def __init__(self, units, **kwargs):
        cell = CardioRNNCell(units, **kwargs)
        super().__init__(cell, **kwargs)

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)
    
from keras.losses import mean_squared_error as mse
autoencoder = load_model("autoencoder.h5",custom_objects={'mse': mse})


# Load the pre-trained weights
rnn_model = load_model("cardio_rnn_model.h5", custom_objects={'CardioRNNCell': CardioRNNCell})

df = pd.read_csv("heart_disease_train.csv")

# Separate the features and target variable
X_train = df.drop("HeartDiseaseorAttack", axis=1)
y_train = df["HeartDiseaseorAttack"]

# Normalize the data
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)


#-----------------Model load and setup part , don't touch : Ritabrata ------------------#

# Main function
if __name__ == '__main__':
    st.markdown("## Group 1 Project:\nCardio Disease Prediction")
    with st.sidebar:
        selected = option_menu('Cardio Disease Predictior',
                          
                          ['Predict Cardio Disease',
                           'Our Prediction Records',
                           'About Us'],
                          icons=['heart','book','info'],
                          default_index=0)
        
    
    if selected =="Predict Cardio Disease":
        # Collect user inputs
        Name = st.text_input('Enter your name:')
        genhealth_options = {
            "Excellent": 1,
            "Good ": 2,
            "Bad": 3,
            "Poor": 4
        }

        GenHealth_key=st.selectbox("General Health (Excellent,Good,Bad,Poor)",options=list(genhealth_options.keys()))
        GenHealth = genhealth_options[GenHealth_key]
        age_options = {
            "18-24": 1,
            "25-29 ": 2,
            "30-34": 3,
            "35-39": 4,
            "40-44": 5,
            "45-49": 6,
            "50-54": 7,
            "55-59": 8,
            "60-64": 9,
            "65-69": 10,
            "70-74": 11,
            "75-79": 12,
            "80+": 13
        }
        Age_key = st.selectbox("Age in Range",options=list(age_options.keys()))
        Age = age_options[Age_key]
        diffwalk_options = {
            "No Difficulty": 0,
            "Some/Much Difficulty": 1,
        }
        Difficultyinwalk_key=st.selectbox('Difficulty in walking:',options=list(diffwalk_options.keys()))
        Difficultyinwalk = diffwalk_options[Difficultyinwalk_key]
        HighBP_options = {
            "No": 0,
            "Yes": 1
        }
        HighBP_key=st.selectbox('High Blood Pressure:',options=list(HighBP_options.keys()))
        HighBP = HighBP_options[HighBP_key]
        Stroke_options = {
            "No History": 0,
            "Yes": 1
        }
        Stroke_key=st.selectbox('History of Stroke:',options=list(Stroke_options.keys()))
        Stroke = Stroke_options[Stroke_key]
        PhysHealth = st.number_input(
            'Number of days in the past month when health was not good:',
            min_value=0,
            max_value=30,
            value=0,
            step=1
        )
        Diabetes_options = {
            "No": 0,
            "Yes": 1
        }
        Diabetes_key=st.selectbox('Diabetes:',options=list(Diabetes_options.keys()))
        Diabetes = Diabetes_options[Diabetes_key]
        High_chols_options = {
            "No": 0,
            "Yes": 1
        }
        High_chols_key=st.selectbox('High Cholestrol:',options=list(High_chols_options.keys()))
        High_chols = High_chols_options[High_chols_key]
        Smkoer_options = {
            "No/less than 100 cigarettes in lifetime": 0,
            "Atleast 100 cigerettes smoked": 1
        }
        Smoker_key=st.selectbox('Smoker:',options=list(Smkoer_options.keys()))
        Smoker = Smkoer_options[Smoker_key]
        input_data = {
            'GenHealth' : GenHealth,
            'Age': Age,
            'Difficultyinwalk' : Difficultyinwalk,
            'HighBP' : HighBP,
            'Stroke' : Stroke,
            'PhysHealth': PhysHealth,
            'Diabetes' : Diabetes,
            'High_chols':High_chols,
            'Smoker':Smoker
            # Add other input fields here
        }

        # Convert input data to a list for prediction
        #input_features = [input_data[feature] for feature in input_data]
         # Transform to the desired format
        input_datas = {
            'GenHlth': [input_data['GenHealth']],
            'Age': [input_data['Age']],            'DiffWalk': [input_data['Difficultyinwalk']],
            'HighBP': [input_data['HighBP']],
            'Stroke': [input_data['Stroke']],
            'PhysHlth': [input_data['PhysHealth']],
            'Diabetes': [input_data['Diabetes']],
            'HighChol': [input_data['High_chols']],
            'Smoker': [input_data['Smoker']]
        }
        
        input_df = pd.DataFrame(input_datas)
        
        # Predict
        if st.button('Predict'):
            # Normalize the input data using the same scaler
            input_normalized = scaler.transform(input_df)

            # Extract features using the autoencoder
            input_features = autoencoder.predict(input_normalized)

            # Reshape input features for the RNN model
            input_features = np.expand_dims(input_features, axis=2)

            # Load the trained RNN model
            model = rnn_model
            # Perform prediction
            predictions = model.predict(input_features)
            predicted_class = (predictions > 0.5).astype(int)
            print(f"Predicted class: {predicted_class[0][0]}")
            print("Prediction:", predicted_class[0][0])

            # st.write('Input Features:', input_data)
            # # prediction = predict_diabetes(input_features)
            # prediction = model.predict(input_data)
            # st.write('Raw Prediction:', prediction[0])
            f = open("user_records.txt", "a")
            f.write("\n")
            new_data = str([Name,GenHealth,Age,Difficultyinwalk,HighBP,Stroke,PhysHealth,Diabetes,High_chols,Smoker,predicted_class[0][0]])
            leng = len(new_data)
            f.write(new_data[1:leng-1]) 
            f.close()
            
            if predicted_class[0][0] == 1:
                st.write('__You may have heart disease.__')
                #st.write('Accuracy:',round(test_data_accuracy,3),'%')
            else:
                st.write('__You may not have heart disease.__')
                #st.write('Accuracy:',round(test_data_accuracy,3),'%')
            gc.collect()

            def generate_report(Name, input_data, prediction):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter

                # Header Section
                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, height - 50, "Heart Disease Prediction Report")
                c.setFont("Helvetica", 12)
                c.drawString(100, height - 70, "--------------------------------------------")
                
                # User Information
                c.drawString(100, height - 90, f"Name: {Name}")
                
                # Add input data dynamically
                y_offset = height - 110
                for field, value in input_data.items():
                    field_name = field.replace("_", " ").title()  # Format field names for readability
                    c.drawString(100, y_offset, f"{field_name}: {value}")
                    y_offset -= 20

                # Prediction Section
                y_offset -= 10
                c.drawString(100, y_offset, "--------------------------------------------")
                y_offset -= 20
                c.setFont("Helvetica-Bold", 12)
                c.drawString(100, y_offset, "Prediction:")
                y_offset -= 20
                if prediction == 1:
                    c.setFillColorRGB(1, 0, 0)  # Red color for positive prediction
                    prediction_text = "High Risk of Heart Disease"
                else:
                    c.setFillColorRGB(0, 1, 0)  # Green color for negative prediction
                    prediction_text = "Low Risk of Heart Disease"
                c.drawString(100, y_offset, prediction_text)
                
                # Footnote
                y_offset -= 40
                c.setFillColor(colors.black)
                c.setFont("Helvetica", 10)
                footnote = ("Note: The prediction is based on probability. Actual results may vary. "
                            "Please consult a doctor for a detailed check.")
                c.drawString(100, y_offset, footnote)
                
                # Finalize PDF
                c.showPage()
                c.save()
                pdf_bytes = buffer.getvalue()
                buffer.close()
                return pdf_bytes

            # Generate Report
            pdf_bytes = generate_report(Name, input_data, predicted_class[0])

            # Base64 Encoding for Download Link
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            download_link = f'<a href="data:application/pdf;base64,{pdf_base64}" download="CardioDisease_report_{Name}.pdf">Download Report</a>'

            # Streamlit Display
            st.markdown(download_link, unsafe_allow_html=True)
            gc.collect()

    if selected == "Our Prediction Records":
        st.markdown("<h3 style='text-align: center;'>PREDICTION RECORDS OF OUR PREVIOUS USERS</h1>", unsafe_allow_html=True)
        f = pd.read_csv("user_records.txt")
        #st.table(f)
        st.table(f.style.set_table_attributes('style="width:100%;"'))
        st.markdown("____")
        st.write("All the records are stored only for academic and research purpose & will not be used for any other means.")
        
    if selected == "About Us":
        st.markdown("<h2 style='text-align: center;'>ABOUT</h2>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<p style='text-align: center;'>This is an academic project made by B.Tech Computer Science And Engineering 4th year student group.</p>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<h4 style='text-align: center;'>Developed and maintained by</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Pankaj Goel</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Protim Debnath</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Soumen Paul</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Soumik Saha</p>", unsafe_allow_html=True)
        st.markdown("____")
