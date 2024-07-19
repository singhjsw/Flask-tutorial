from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

def load_model(filename="model.bin"):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

model = load_model()

def preprocess_origin_cols(df):
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df

def pipeline_transformer(data):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    class CustomAttrAdder(BaseEstimator, TransformerMixin):
        def __init__(self, acc_on_power=True):
            self.acc_on_power = acc_on_power
            self.acc_ix = 4
            self.hpower_ix = 2
            self.cyl_ix = 0
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            acc_on_cyl = X[:, self.acc_ix] / X[:, self.cyl_ix]
            if self.acc_on_power:
                acc_on_power = X[:, self.acc_ix] / X[:, self.hpower_ix]
                return np.c_[X, acc_on_power, acc_on_cyl]
            return np.c_[X, acc_on_cyl]

    def num_pipeline_transformer(data):
        numerics = ['float64', 'int64']
        num_attrs = data.select_dtypes(include=numerics)
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attrs_adder', CustomAttrAdder()),
            ('std_scaler', StandardScaler()),
        ])
        return num_attrs, num_pipeline

    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
    ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data

def predict_mpg(config, model):
    if isinstance(config, dict):
        df = pd.DataFrame(config)
    else:
        df = config

    preprocessed_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preprocessed_df)
    return model.predict(prepared_df)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    predictions = predict_mpg(df, model)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
