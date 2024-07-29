import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):

    #load CSV
    # file_path = 'winequality-white.csv'
    wine_data = pd.read_csv(file_path,delimiter=';')

    #separate features and target
    X = wine_data.drop('quality', axis=1)
    y = wine_data['quality']

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    #rescale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test





# #return to Pandas dataframe for viewing
# X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
# X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# #view results
# print(X_train_scaled_df.head())
# print(X_test_scaled_df.head())
# print(X_train_scaled_df.shape)
# print(X_test_scaled_df.shape)








