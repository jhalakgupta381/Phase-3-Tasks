import multiprocessing
import time
import concurrent.futures

t1 = time.perf_counter()

def task1():
    print("Task 1: Starting")
    #t1 = time.perf_counter()
    import pandas as pd
    import numpy as np
    import warnings
    
    csv_file_path = './cleaned_data.csv'
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Handling outliers or incorrect values 
    df['Cumulative_cases'] = df['Cumulative_cases'].apply(lambda x: max(0, x))
    print(df.head(100))
    #time.sleep(2)  
    #t2 = time.perf_counter()
    #print(f'Task 1 Finished in {t2-t1} seconds')
    #time.sleep(2)
    
def task2():
    print("Task 2: Starting")
    t1 = time.perf_counter()
    import warnings
    import pandas as pd
    csv_file_path = './cleaned_data.csv'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(csv_file_path,low_memory=False)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        print("\nColumn Names After Uniformity:")
        print(df.columns)
    #time.sleep(2)
    #t2 = time.perf_counter() 
    #print(f'Task 2 Finished in {t2-t1} seconds')
    #time.sleep(2)
    
def task3():
    print("Task 3: Starting")
    t1 = time.perf_counter()
    import warnings
    import pandas as pd
    csv_file_path = './merged_data.csv'
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import confusion_matrix
    import plotly.express as px
    
    target_col = ['Cumulative_death']
    
    # Extract features and target variable
    X = df.drop(target_col, axis=1)  
    y = df[target_col]                
    
    # Drop columns with NaN valuesz
    X_no_nan = X.dropna(axis=1)
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Extract numeric features
    numeric_columns = X_no_nan.select_dtypes(include=['float64', 'int64']).columns
    X_numeric = X_no_nan[numeric_columns]
    
    # Use KFold for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize a model 
    model = LinearRegression()
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X_numeric):
        X_train, X_test = X_numeric.iloc[train_index], X_numeric.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        # Fit the model on the training data
        model.fit(X_train, y_train)
    
        # Print the model accuracy
        score = model.score(X_test, y_test)
        
    import statsmodels.formula.api as smf
    model = smf.ols('Cumulative_cases ~ Wind_speed', data=df)
    results = model.fit()
    print(results.summary())
    #time.sleep(2)
    #t2 = time.perf_counter()
    #print(f'Task 3 Finished in {t2-t1} seconds')
    #time.sleep(2)
    
def task4():
    print("Task 4: Starting")
    t1 = time.perf_counter()
    import warnings
    import pandas as pd
    import graphviz.backend as be
    from IPython.display import Image, display_svg, SVG
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    import dtreeviz
    
    csv_file_path = './cleaned_data.csv'
    df = pd.read_csv(csv_file_path, low_memory=False)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    target_col = 'cumulative_death'
    
    # Extract features and target variable
    X = df.drop(target_col, axis=1)  
    y = df[target_col]               
    
    # Drop columns with NaN values
    X_no_nan = X.dropna(axis=1)
    
    # Convert 'Date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    
    # Extract numeric features
    numeric_columns = X_no_nan.select_dtypes(include=['float64', 'int64']).columns
    X_numeric = X_no_nan[numeric_columns]
    
    # Use KFold for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize a Decision Tree model
    model = DecisionTreeRegressor()
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X_numeric):
        X_train, X_test = X_numeric.iloc[train_index], X_numeric.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        # Fit the Decision Tree model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)
    
        # Print the model accuracy
        score = model.score(X_test, y_test)
        print(f'Model Accuracy: {score:.4f}')

    #time.sleep(2)
    #t2 = time.perf_counter()
    #print(f'Task 4 Finished in {t2-t1} seconds')
    #time.sleep(2)
    
def task5():
    print("Task 5: Starting")
    import requests
    import time
    import concurrent.futures
    
    img_urls = [
        'https://images.unsplash.com/photo-1516117172878-fd2c41f4a759',
        'https://images.unsplash.com/photo-1532009324734-20a7a5813719',
        'https://images.unsplash.com/photo-1524429656589-6633a470097c',
        ]
    
    t1 = time.perf_counter()
    def download_image(img_url):
        img_bytes = requests.get(img_url).content
        img_name = img_url.split('/')[3]
        img_name = f'{img_name}.jpg'
        with open(img_name, 'wb') as img_file:
            img_file.write(img_bytes)
            #print(f'{img_name} was downloaded...')
    
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(download_image, img_urls)

    #time.sleep(2)
    #t2 = time.perf_counter()
    
    #print(f'Task 5 Finished in {t2-t1} seconds')

def task6():
    print("Task 6: Starting")
    t1 = time.perf_counter()
    import pandas as pd
    csv_file_path = './merged_data.csv'
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Filtering based Cumulative_cases > 0
    filtered_data = df[(df['Cumulative_cases'] > 0)]
    
    # Displaying filtered data summary
    print("Filtered Data Summary (Cumulative_cases > 0):")
    print(filtered_data[['Date', 'Country', 'Cumulative_cases']])
    
    # Additional filtering for date-wise and country-wise conditions
    date_filter = (df['Date'] >= '2020-01-22')
    country_filter = (df['Country'] == 'Albania')
    
    # Apply additional filters
    filtered_data_date = df[date_filter]
    filtered_data_country = df[country_filter]
    
    # Displaying filtered data summary for date-wise condition
    print("\nFiltered Data Summary (Date >= '2020-01-22'):")
    print(filtered_data_date[['Date', 'Country', 'Cumulative_cases']])
    
    # Displaying filtered data summary for country-wise condition
    print("\nFiltered Data Summary (Country == 'Albania'):")
    print(filtered_data_country[['Date', 'Country', 'Cumulative_cases']])
    
    # Grouping by 'Date' and finding the sum of 'Cumulative_cases'
    grouped_data_datewise = df.groupby('Date')['Cumulative_cases'].sum()
    
    # Displaying date-wise grouped data summary
    print("\nDate-wise Grouped Data Summary:")
    print(grouped_data_datewise)
    
    # Grouping by 'Country' and finding the sum of 'Cumulative_cases'
    grouped_data_countrywise = df.groupby('Country')['Cumulative_cases'].sum()
    
    # Displaying country-wise grouped data summary
    print("\nCountry-wise Grouped Data Summary:")
    print(grouped_data_countrywise)
    
    # Grouping by 'Country' and 'Date' and finding the sum of 'Cumulative_cases'
    grouped_data_country_datewise = df.groupby(['Country', 'Date'])['Cumulative_cases'].sum()
    
    # Displaying country and date-wise grouped data summary
    print("\nCountry and Date-wise Grouped Data Summary:")
    print(grouped_data_country_datewise)
    
    # Relation between 'Available Beds' and 'Cumulative Deaths' for each grouping
    relation_data = df.groupby('Date')[['Available Beds/1000', 'Cumulative_death']].sum()
    print("\nRelation between Available Beds and Cumulative Deaths (Date-wise):")
    print(relation_data)
    
    relation_data = df.groupby('Country')[['Available Beds/1000', 'Cumulative_death']].sum()
    print("\nRelation between Available Beds and Cumulative Deaths (Country-wise):")
    print(relation_data)
    
    relation_data = df.groupby(['Country', 'Date'])[['Available Beds/1000', 'Cumulative_death']].sum()
    print("\nRelation between Available Beds and Cumulative Deaths (Country and Date-wise):")
    print(relation_data)
    #time.sleep(2)
    #t2 = time.perf_counter()
    #print(f'Task 6 Finished in {t2-t1} seconds')
    #time.sleep(2)
    
if __name__ == '__main__':
    start = time.perf_counter()
    p1 = multiprocessing.Process(target=task1)
    p2 = multiprocessing.Process(target=task2)
    p3 = multiprocessing.Process(target=task3)
    p4 = multiprocessing.Process(target=task4)
    p5 = multiprocessing.Process(target=task5)
    p6 = multiprocessing.Process(target=task6)
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 1)} second(s)')
