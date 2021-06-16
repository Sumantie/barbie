def ffill_missing(ser):
    return ser.fillna(method="ffill")

def is_weekend(data):
    return (
        data["dteday"]
        .dt.day_name()
        .isin(["Saturday", "Sunday"])
        .to_frame()
    )

def year(data):
    return (data["dteday"].dt.year - 2011).to_frame()

def train_and_persist():
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/IE-Advanced-Python/1-package-ml-model-Sumant-Duggal/master/reference/hour.csv?token=ATN4YKBMRXROYLWH6TGWC5DA2IESS", parse_dates=["dteday"])
    X = df.drop(columns=["instant", "cnt", "casual", "registered"])
    y = df["cnt"]
    import numpy as np
    from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
    ffiller = FunctionTransformer(ffill_missing)
    weather_enc = make_pipeline(
        ffiller,
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
        ),
    )
    ct = make_column_transformer(
        (ffiller, make_column_selector(dtype_include=np.number)),
        (weather_enc, ["weathersit"]),
    )
    from sklearn.pipeline import FeatureUnion, make_union
    preprocessing = FeatureUnion([
        ("is_weekend", FunctionTransformer(is_weekend)),
        ("year", FunctionTransformer(year)),
        ("column_transform", ct)
    ])
    from sklearn.ensemble import RandomForestRegressor
    reg = Pipeline([("preprocessing", preprocessing), ("model", RandomForestRegressor())])
    X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
    X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]
    reg.fit(X_train, y_train)
    import joblib
    filename = 'reg_model.sav'
    joblib.dump(reg, filename)
    
    
def predict(dteday,hr,weathersit,temp,atemp,hum,windspeed):
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    loaded_model = joblib.load('reg_model.sav')
    
    data = {'dteday':[dteday] ,'hr': [hr],'weathersit':weathersit,'temp':[temp],'atemp':[atemp],'hum':[hum],'windspeed':[windspeed]}
    df = pd.DataFrame(data,columns = ['dteday','hr','weathersit','temp','atemp','hum','windspeed'])
    df['dteday'] = pd.to_datetime(df['dteday'])
    predicitions = loaded_model.predict(df)
    print(predicitions)