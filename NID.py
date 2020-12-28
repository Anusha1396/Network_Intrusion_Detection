import metrics as metrics
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

# Dataset field names
datacols = ["duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"]

# Load NSL_KDD train dataset
dfkdd_train = pd.read_table("KDDTrain+.txt", sep=",",names=datacols)


# Load NSL_KDD test dataset
dfkdd_test = pd.read_table("KDDTest+.txt", sep=",", names=datacols)


mapping = {'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'saint': 'Probe',
           'mscan': 'Probe',
           'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS', 'neptune': 'DoS', 'smurf': 'DoS',
           'mailbomb': 'DoS','udpstorm': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
           'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'buffer_overflow': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
           'sqlattack': 'U2R', 'httptunnel': 'U2R',
           'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L',
           'imap': 'R2L','spy': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L', 'snmpgetattack': 'R2L',
           'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
           'normal': 'Normal'
           }

# Apply attack class mappings to the dataset
dfkdd_train['attack_class'] = dfkdd_train['attack'].apply(lambda v: mapping[v])
dfkdd_test['attack_class'] = dfkdd_test['attack'].apply(lambda v: mapping[v])

# Drop attack field from both train and test data
dfkdd_train.drop(['attack'], axis=1, inplace=True)
dfkdd_test.drop(['attack'], axis=1, inplace=True)

# Attack Class Distribution
attack_class_freq_train = dfkdd_train[['attack_class']].apply(lambda x: x.value_counts())
attack_class_freq_test = dfkdd_test[['attack_class']].apply(lambda x: x.value_counts())
attack_class_freq_train['frequency_percent_train'] = round((100 * attack_class_freq_train / attack_class_freq_train.sum()),2)
attack_class_freq_test['frequency_percent_test'] = round((100 * attack_class_freq_test / attack_class_freq_test.sum()),2)

print("The Attack  class Distribution is: \n ")
attack_class_dist = pd.concat([attack_class_freq_train,attack_class_freq_test], axis=1)
print_full(attack_class_dist)

# Concat the df's together
df = pd.concat([dfkdd_train, dfkdd_test], ignore_index=True)

# Convert the attack label to a binary classification problem 0=normal 1=attack
df["attack"] = df["attack_class"].apply(lambda x: 0 if x=="Normal" else 1)
print(f"\n The classification of Attack Class 0=normal 1=attack is:\n ",df["attack"])


print(f"\n Unique values of target variable : \n {df['attack'].unique()}")
print("\n 0=normal 1=attack")
print(f"\n Number of samples under each target value : \n {df['attack'].value_counts()}")

# Get the one-hot encoding
one_hot = pd.get_dummies(df[["protocol_type", "service", "flag"]])
df = df.join(one_hot)
df.drop(["protocol_type", "service", "flag"], inplace=True, axis=1)

dfkdd_train = df.iloc[0:125973, :]
dfkdd_test = df.iloc[125973:148517, :]

y_train = np.array(dfkdd_train["attack"])
y_test = np.array(dfkdd_test["attack"])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(dfkdd_train.drop(["attack", "attack_class"], axis=1))
X_test = scaler.transform(dfkdd_test.drop(["attack", "attack_class"], axis=1))

print("\n Running the 6 Classifiers...\n")
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# #Train Decision Tree Model
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, y_train)

# # Train KNeighborsClassifier Model
KNN_Classifier = KNeighborsClassifier(n_neighbors = 8)
KNN_Classifier.fit(X_train, y_train)

# # Train Gaussian Naive Baye Model
BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, y_train)

# # # Train LogisticRegression Model
LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, y_train)

#Train RandomForestClassifier Model
RF_Classifier = RandomForestClassifier(n_estimators = 50)
RF_Classifier.fit(X_train, y_train)
#
# #Train SVM Model
SVC_Classifier= SVC(random_state=0)
SVC_Classifier.fit(X_train, y_train)



from sklearn import metrics

models = []
models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('Naive Baye Classifier', BNB_Classifier))
models.append(('LogisticRegression', LGR_Classifier))
models.append(('RandomForest Classifier', RF_Classifier))
models.append(('SVM Classifier', SVC_Classifier))

for i, v in models:
    accuracy = metrics.accuracy_score(y_test, v.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(y_test, v.predict(X_test))
    classification = metrics.classification_report(y_test, v.predict(X_test))
    print()
    print('============================== {} Model Test Results =============================='.format(i))
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()





















