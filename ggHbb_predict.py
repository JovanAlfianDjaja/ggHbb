#================================================
# CODE TO PREDICT M^2 FROM MANDELSTAM INVARIANTS
#================================================

# IMPORT PACKAGES =================================================================
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
#==================================================================================

# LOAD .txt FILE WITH MANDELSTAM INVARIANTS AS PREDICTORS =========================
# MAKE SURE :
# COLUMN 1 = s12, COLUMN 2 = s23
# COLUMN 3 = s34, COLUMN 4 = s45
# COLUMN 5 = s15, COLUMN 6 = s5

# .txt FILE DIRECTORY TO LOAD DATA POINTS TO BE PREDICTED 
# file_path = directory of .txt file
file_path = ...

# LOAD DATA
data = np.loadtxt(file_path)
#==================================================================================

# DATA PREPROCESSING ==============================================================
# NUMBER OF FEATURES
num_features = data.shape[1]

# MANDELSTAM INVARIANTS DATA
x = data[:, :num_features]

# DATA POINTS TRANSFORMATION
pt = QuantileTransformer(output_distribution='uniform')
pt = pt.fit(x)
x_tr = pt.fit_transform(x)  # TRANSFORMED DATA
#==================================================================================

# PREDICTION PROCESS ==============================================================
# LOAD MODEL (LOAD model.keras FILE FROM DIRECTORY OF CHOICE)
model = keras.models.load_model('.../model.keras')

# AMPLITUDE PREDICTION
y = model.predict(x_tr)[:,0]  # PREDICT USING TRANSFORMED INVARIANTS
#==================================================================================

# RESULT ================================================================================
result = pd.DataFrame(x)  # ORIGINAL DATA TO BE RECORDED IN RESULT, NOT TRANSFORMED DATA

result.columns = ['s12', 's23', 's34', 's45', 's15', 's5']  # COLUMNS

result['M^2'] = y                                           # PREDICTED AMPLITUDE COLUMN

# RESULT SAVED TO DIRECTORY OF CHOICE AS TEXT FILE
result.to_excel('.../Prediction.xlsx', index=False)
#========================================================================================



