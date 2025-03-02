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

# LOAD FILE WITH MANDELSTAM INVARIANTS AS PREDICTORS ==============================
# MAKE SURE :
# COLUMN 1 = s12, COLUMN 2 = s23
# COLUMN 3 = s34, COLUMN 4 = s45
# COLUMN 5 = s15, COLUMN 6 = s5

# FILE DIRECTORY (file_path = [directory of file])
file_path = "C:/Users/jovan/Downloads/Predict/test.txt"

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

# LOAD MODEL
model = keras.models.load_model('C:/Users/jovan/Downloads/Predict/model.keras')

# AMPLITUDE PREDICTION
y = model.predict(x)[:,0]
#==================================================================================

# RESULT ============================================================================
result = pd.DataFrame(x)

result.columns = ['s12', 's23', 's34', 's45', 's15', 's5']

result['Predicted Amplitude'] = y

# RESULT SAVED TO DIRECTORY OF CHOICE AS TEXT FILE, TAB SEPARATED
result.to_csv('C:/Users/jovan/Downloads/Predict/Result.txt', sep='\t', index=False)
#====================================================================================



