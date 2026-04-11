import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os

from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer 

#group 16 used dataset 4
#we modified the dataset and removed '.' from Ph.D. and converted it to PhD

st.set_page_config(layout='wide', page_title='Dashboard')
st.title("Analytics Dashboard")

os.chdir('C:/Users/patel/Desktop/AU/TOD310/Whole Code by Sir/Data')

tab1, tab2, tab3 = st.tabs(['MLR', 'K-Means', 'KNN'])

with tab1:

    try:
        os.chdir('C:/Users/patel/Desktop/AU/TOD310/Whole Code by Sir/Data')
    except FileNotFoundError:
        print("")

    options = os.listdir()
    flSel = st.selectbox(label='Data Files', options=options)
    df = pd.read_csv(flSel)

    if flSel == 'group - 4.csv':
        target = df.loc[(df.index), 'Employee_Salary']
        features = df.loc[(df.index), ['Years_of_Experience', 'Education_Level', 'Job_Title', 'Industry', 'Location']]
    else:
        target = df.loc[(df.index), 'Employee_Salary']
        features = df.loc[(df.index), ['Years_of_Experience', 'Education_Level', 'Job_Title', 'Industry', 'Location']]

    df2 = df.copy()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Features', 'EDA', 'Label Encoding', 'Correlation and ANOVA', 'Model'])

    with tab1:
        st.header("Features and Target variables")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("Features")
            st.write(features)
        with col2:
            st.write("Target")
            st.write(target)

    with tab2:
        st.header('Exploratory Description')
        g = sns.catplot(x="Employee_Salary", y="Years_of_Experience", col_wrap=3, col="Location", data=df,
                        kind="box", height=5, aspect=0.8)
        st.pyplot(g)
        g = sns.catplot(x="Employee_Salary", y="Years_of_Experience", col_wrap=3, col="Location", data=df,
                        kind="box", height=5, aspect=0.8)
        st.pyplot(g)

    with tab3:
        st.header('Label Encoding')
        if st.button('Encode Labels'):
            df2 = df.copy()
            le = LabelEncoder()
            df2 = pd.get_dummies(df2, columns=['Education_Level'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Job_Title'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Industry'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Location'], drop_first=True)
            st.write(df2.head(10))

    with tab4:
        st.header('Correlation and ANOVA')
        if st.button('Correlation'):
            le = LabelEncoder()
            df2 = pd.get_dummies(df2, columns=['Education_Level'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Job_Title'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Industry'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Location'], drop_first=True)
            st.write(df2.corr()['Employee_Salary'])

    with tab5:
        st.header('Linear Regression Model')
        if st.button('Fit Model'):
            le = LabelEncoder()
            df2 = pd.get_dummies(df2, columns=['Education_Level'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Job_Title'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Industry'], drop_first=True)
            df2 = pd.get_dummies(df2, columns=['Location'], drop_first=True)
            y = df2['Employee_Salary']
            X = df2.drop(['Employee_Salary'], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

            lr = LinearRegression().fit(X_train, y_train)
            y_train_pred = lr.predict(X_train)
            y_test_pred = lr.predict(X_test)
            fig = px.scatter(
                x=y_test,
                y=y_test_pred
            )

            st.code(f'coefficients {lr.coef_}')
            st.write("Plotting Actual Charges and Predicted Charges")
            st.plotly_chart(fig, theme=None, use_container_width=True)

with tab2:
    try:
        os.chdir('C:/Users/patel/Desktop/AU/TOD310/Whole Code by Sir/Data')
    except FileNotFoundError:
        print("")

    df = pd.read_csv('C:/Users/patel/Desktop/AU/TOD310/Whole Code by Sir/Data/group - 4.csv')
    
    features = df[['Years_of_Experience', 'Education_Level', 'Job_Title', 'Location']]
    target = df[['Industry']]

# encode target values with LabelEncoder
    le = LabelEncoder()
    target['Industry'] = le.fit_transform(target['Industry'])

# target['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2],inplace=True)
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    with st.expander('Professor Data'):
        st.write(df.head(10))

    with st.expander('Raw Plot'):
        fig, ax = plt.subplots()

        ax.scatter(df['Education_Level'], df['Industry'])
        ax.set_xlabel('Education_Level')
        ax.set_ylabel('Industry')

        st.pyplot(fig)

    with st.expander('Let us check organic clusters possible'):
        inert = '''#Elbow model
        inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)'''

        st.code(line_numbers=True, body=inert)

        data = list(zip(x, y))

        inertias = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)

            fig, ax = plt.subplots()
        ax.plot(range(1, 11), inertias, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.figure(figsize=(5, 3))
        st.pyplot(fig)

    with st.expander('Splitting Training and Testing data'):
        st.write('Splitting Features and Targets into Train and Test Datasets')
        st.code('X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)')
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)

    with st.expander('Model fit and prediction'):
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(features)
        cds = '''
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(features)
    '''
        st.code(line_numbers=True, body=cds)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=kmeans.labels_)
        st.pyplot(fig)

    with st.expander('Prediction and Evaluation'):
        st.write('Prediction')
        st.code('y_pred = kmeans.predict(X_test)')
        y_pred = kmeans.predict(X_test)

        accs = accuracy_score(y_test['Species'], y_pred)
        st.write(f'Accuracy score: {accs}')
        cm = confusion_matrix(y_test['Species'], y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
        st.write(disp.plot().figure_)
with tab3:
    file_path = 'C:/Users/patel/Desktop/AU/TOD310/Whole Code by Sir/Data/group - 4.csv'  # Replace with the actual file path
    df = pd.read_csv(file_path)

    st.set_page_config(layout='wide', page_title='K Nearest Neighbor Classification', page_icon=':fallen_leaf:')

    # Specify the target and features
    target_column = 'Education_Level'  # Replace with your target column name
    features = df.drop([target_column], axis=1)
    target = df[[target_column]]

    # Apply one-hot encoding to all non-numeric columns
    non_numeric_columns = features.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    for column in non_numeric_columns:
        encoded = encoder.fit_transform(features[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        features = pd.concat([features, encoded_df], axis=1)
        features = features.drop([column], axis=1)

    # Option 2: Impute missing target values with the mode (most frequent value)
    imputer = SimpleImputer(strategy='most_frequent')
    target[target_column] = imputer.fit_transform(target[[target_column]])

    # Splitting Features and Target into Train and Test Datasets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)

    # Create and fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Predict values for the test dataset
    y_pred = knn.predict(X_test)

    # Define the Streamlit tabs
    tab1, tab2, tab3 = st.columns([1, 1, 1])  # Create all tabs without using 'with'

    # Tab 1: Reading CSV file and displaying data
    with tab1:
        st.code(line_numbers=True, body=f"""# Reading CSV file: {file_path}
    df = pd.read_csv('{file_path}')
    st.write(df.head(10))""")
        st.write(df.head(10))

    # Tab 2: Separating Features and Target
    with tab2:
        st.write("Separating Features and Target")
        st.code(line_numbers=True, body=f"target_column = '{target_column}'")
        st.code(line_numbers=True, body="features = df.drop([target_column], axis=1)")
        st.code(line_numbers=True, body="target = df[[target_column]]")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write('Features')
            st.write(features.head(10))
        with col2:
            st.write('Target')
            st.write(target.head(10))
        with col3:
            st.write('Encoded Target (if applicable)')
            st.write(target.head(10))

    # Tab 3: KNN Model, Confusion Matrix, and Accuracy Score
    with tab3:
        st.subheader('KNN Model')
        st.code('knn = KNeighborsClassifier(n_neighbors=3) # Creating Model with 3 clusters')
        st.write(knn)
        st.code('knn.fit(X_train, y_train) # Fitting Model on the training data')
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High School', 'Bachelor', 'Master', 'PhD'])
        st.write(disp.plot().figure_)

        st.subheader("Accuracy Score")
        accuracyScore = accuracy_score(y_true=y_test, y_pred=y_pred)
        st.code("accuracyScore = accuracy_score(y_true=y_test, y_pred=y_pred)")
        st.write("Accuracy Score:", accuracyScore)

