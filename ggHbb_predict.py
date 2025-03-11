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
x_tr = data[:, :num_features]/1000000  # TRANSFORMED DATA
#==================================================================================

# PREDICTION PROCESS ==============================================================

# LOAD MODEL
model = keras.models.load_model('.../model.keras')

# AMPLITUDE PREDICTION
y = model.predict(x_tr)[:,0]
#==================================================================================

# RESULT ============================================================================
result = pd.DataFrame(x)

result.columns = ['s12', 's23', 's34', 's45', 's15', 's5']

result['M^2'] = np.exp(y)

# RESULT SAVED TO DIRECTORY OF CHOICE AS EXCEL FILE
result.to_excel('..../Result.xlsx', index=False)
#====================================================================================
