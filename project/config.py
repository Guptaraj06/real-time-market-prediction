class CFG:
    PATH_DATA = "/kaggle/input/jane-street-real-time-market-data-forecasting"
    PATH_MODELS = "/kaggle/working/janestreet-models"
    COL_TARGET = "responder_6"
    COL_ID = "symbol_id"
    COL_DATE = "date_id"
    COL_TIME = "time_id"
    COL_WEIGHT = "weight"
    COL_RESPONDERS = [f"responder_{i}" for i in range(11)]
