from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = '0f4293ee4309ced15b104b98fc33ba14be08d36c92b6da21'  # Replace with a secure key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load the dataset using pandas
        df = pd.read_csv(file_path)
        
        # Store dataset information in session for the next page
        session['file_path'] = file_path
        session['columns'] = df.columns.tolist()
        session['shape'] = tuple(map(int, df.shape))  # Convert shape to tuple of ints
        session['head'] = df.head().to_html(classes='data', header=True, index=False)
        session['duplicates'] = int(df.duplicated().sum())  # Convert to int
        session['isnull'] = int(df.isnull().sum().sum())  # Convert to int
        session['isna'] = int(df.isna().sum().sum())  # Convert to int
        
        return redirect(url_for('dataset_info'))

@app.route('/dataset_info')
def dataset_info():
    # Retrieve dataset information from session
    file_path = session.get('file_path')
    columns = session.get('columns')
    shape = session.get('shape')
    head = session.get('head')
    duplicates = session.get('duplicates')
    isnull = session.get('isnull')
    isna = session.get('isna')
    
    if not file_path:
        return redirect(url_for('welcome'))
    
    return render_template('dataset_info.html', 
                           file_path=file_path, 
                           columns=columns, 
                           shape=shape, 
                           head=head,
                           duplicates=duplicates,
                           isnull=isnull,
                           isna=isna)

@app.route('/select_columns_to_drop', methods=['GET', 'POST'])
def select_columns_to_drop():
    file_path = session.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('dataset_info'))
    
    df = pd.read_csv(file_path)
    
    if request.method == 'POST':
        columns_to_drop = request.form.getlist('drop_columns')
        target_column = request.form.get('target_column')
        
        # Drop selected columns
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            df.to_csv(file_path, index=False)
        
        # Save the target column in session
        session['target_column'] = target_column
        
        # Update session info
        session['columns'] = df.columns.tolist()
        session['shape'] = tuple(map(int, df.shape))  # Update shape in session
        session['head'] = df.head().to_html(classes='data', header=True, index=False)
        
        return redirect(url_for('select_columns_to_encode'))
    
    return render_template('select_columns_to_drop.html', columns=df.columns)

@app.route('/drop_columns', methods=['POST'])
def drop_columns():
    selected_columns = request.form.getlist('columns_to_drop')
    file_path = session.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('welcome'))
    
    df = pd.read_csv(file_path)
    
    # Drop the selected columns
    df.drop(columns=selected_columns, inplace=True)
    
    # Save the updated DataFrame back to the session file
    df.to_csv(file_path, index=False)
    
    # Update the session with the new column list
    session['columns'] = df.columns.tolist()
    session['shape'] = tuple(map(int, df.shape))  # Update shape in session
    session['head'] = df.head().to_html(classes='data', header=True, index=False)
    
    return redirect(url_for('select_columns_to_encode'))

@app.route('/select_columns_to_encode')
def select_columns_to_encode():
    columns = session.get('columns')
    
    if not columns:
        return redirect(url_for('dataset_info'))
    
    return render_template('select_columns_to_encode.html', columns=columns)

@app.route('/apply_encoding', methods=['POST'])
def apply_encoding():
    file_path = session.get('file_path')
    columns = session.get('columns')

    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('welcome'))
    
    df = pd.read_csv(file_path)
    
    ordinal_columns = request.form.getlist('ordinal_columns')
    onehot_columns = request.form.getlist('onehot_columns')
    
    # Apply Ordinal Encoding
    if ordinal_columns:
        encoder = OrdinalEncoder()
        df[ordinal_columns] = encoder.fit_transform(df[ordinal_columns])
    
    # Apply OneHot Encoding
    if onehot_columns:
        df = pd.get_dummies(df, columns=onehot_columns,drop_first=True)
    
    # Save the updated DataFrame back to the session file
    df.to_csv(file_path, index=False)
    
    # Update the session with the new column list
    session['columns'] = df.columns.tolist()
    session['shape'] = tuple(map(int, df.shape))  # Update shape in session
    session['head'] = df.head().to_html(classes='data', header=True, index=False)
    
    return redirect(url_for('show_correlation_heatmap'))

@app.route('/show_correlation_heatmap')
def show_correlation_heatmap():
    file_path = session.get('file_path')
    target_column = session.get('target_column')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('welcome'))
    
    df = pd.read_csv(file_path)
    
    # Generate the correlation matrix
    corr_matrix = df.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(7, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    # Save the heatmap as an image
    heatmap_path = os.path.join(app.config['STATIC_FOLDER'], 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    # Compute the correlation of the target column
    if target_column in df.columns:
        corr_target = abs(corr_matrix[target_column]).sort_values(ascending=False)
    else:
        corr_target = pd.Series(dtype=float)  # Empty Series if target column not present
    
    # Convert the correlation target to HTML
    corr_target_html = corr_target.to_frame(name='Correlation with ' + target_column).reset_index().to_html(classes='data', header=True, index=False)
    
    # Store the top 3 features in session
    top_features = corr_target.drop(target_column).head(3).index.tolist()
    session['top_features'] = top_features
    
    return render_template('show_correlation_heatmap.html', 
                           heatmap_url=url_for('static', filename='heatmap.png'),
                           corr_target_html=corr_target_html,
                           df_html=df.head().to_html(classes='data', header=True, index=False))


@app.route('/show_box_plots', methods=['GET', 'POST'])
def show_box_plots():
    file_path = session.get('file_path')
    target_column = session.get('target_column')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('welcome'))
    
    df = pd.read_csv(file_path)
    
    # Generate the correlation matrix
    corr_matrix = df.corr()
    
    # Compute the correlation of the target column
    if target_column in df.columns:
        corr_target = abs(corr_matrix[target_column]).sort_values(ascending=False)
        top_features = corr_target.index[1:4]  # Exclude target column and take top 3 features
        
        # Plot box plots for the top 3 features
        plt.figure(figsize=(10, 3))
        for feature in top_features:
            plt.subplot(1, 3, top_features.get_loc(feature) + 1)
            sns.boxplot(x=df[feature])
            plt.title(f'Box Plot of {feature}')
        
        plt.tight_layout()
        
        # Ensure old image is removed/overwritten
        boxplot_path = os.path.join(app.config['STATIC_FOLDER'], 'box_plots.png')
        if os.path.exists(boxplot_path):
            os.remove(boxplot_path)
        
        plt.savefig(boxplot_path)
        plt.close()
        
        # Convert the DataFrame to HTML for preview
        df_html = df.head().to_html(classes='data', header=True, index=False)
        
        return render_template('show_box_plots.html', 
                               boxplot_url=url_for('static', filename='box_plots.png'),
                               df_html=df_html,
                               top_features=top_features)



@app.route('/remove_outliers', methods=['POST'])
def remove_outliers():
    file_path = session.get('file_path')
    target_column = session.get('target_column')
    
    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('welcome'))
    
    df = pd.read_csv(file_path)
    
    # Get the columns selected for outlier removal
    columns_to_check = request.form.getlist('columns_to_remove')
    
    for column in columns_to_check:
        if column in df.columns:
            # Compute IQR for each selected column
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Save the filtered DataFrame
    df.to_csv(file_path, index=False)
    
    # Update session with filtered data
    session['columns'] = df.columns.tolist()
    session['shape'] = tuple(map(int, df.shape))  # Update shape in session
    session['head'] = df.head().to_html(classes='data', header=True, index=False)
    
    # Redirect to show_box_plots to display the updated box plots
    return redirect(url_for('show_box_plots'))

@app.route('/train_model', methods=['POST','GET'])
def train_model():
    if request.method=='POST':
        file_path = session.get('file_path')
        target_column = session.get('target_column')
    
        if not file_path or not os.path.exists(file_path):
            return redirect(url_for('welcome'))
    
        df = pd.read_csv(file_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        test_size=0.2
        random_state=42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model=LogisticRegression()
        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)

        accuracy=accuracy_score(y_test,y_pred)
        matrix=confusion_matrix(y_test,y_pred)

        return render_template('model_result.html',
                               test_size=test_size,
                               random_state=random_state,
                               accuracy=accuracy,
                               matrix=matrix)



if __name__ == '__main__':
    app.run(debug=True)

