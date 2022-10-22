import numpy as np
import pandas as pd
import os
import pickle
import gc
import pandas_profiling as pdp
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("../input/titanic/train.csv")
df_train.head()

print(df_train.shape)
print(len(df_train))
print(len(df_train.columns))

df_train.info()



