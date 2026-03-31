import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from aws_dynamodb import load_dynamodb_table_by_date_range
from crawler import duplicate_posts_to_minute_boundaries
from etf_historical import get_stock_bars, build_etf_vwap_future_changes


def get_today_and_last_monday_date_strings() -> tuple[str, str]:
    today = datetime.now()
    last_monday = today - timedelta(days=7)

    return today.strftime("%Y-%m-%d"), last_monday.strftime("%Y-%m-%d")



def is_today_monday() -> bool:
    return datetime.now().weekday() == 0



def build_prediction_performance_summary(
    df: pd.DataFrame,
    metric_1_col: str = "vwap_pct_change_30m",
    metric_2_col: str = "vwap_pct_change_3h",
    reasonableness_col: str = "reasonableness_score",
    high_reasonableness_threshold: float = 0.7,
) -> dict[str, pd.DataFrame]:
    working_df = df.loc[
        df[metric_1_col].notna() & df[metric_2_col].notna()
    ].copy()

    for metric_col, metric_name in [
        (metric_1_col, "metric_1"),
        (metric_2_col, "metric_2"),
    ]:
        working_df[f"actual_signal_{metric_name}"] = working_df[metric_col].apply(
            lambda x: "sell" if x < 0 else "buy"
        )
        working_df[f"binary_{metric_name}"] = (
            working_df["predicted_signal"] == working_df[f"actual_signal_{metric_name}"]
        ).astype(int)

    def summarize(sub_df: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "total_signals": len(sub_df),
            "accuracy_metric_1": sub_df["binary_metric_1"].mean(),
            "avg_metric_1_return": sub_df[metric_1_col].mean(),
            "accuracy_metric_2": sub_df["binary_metric_2"].mean(),
            "avg_metric_2_return": sub_df[metric_2_col].mean(),
        })

    summary_dict = {
        "overall": summarize(working_df).to_frame().T,
        "symbol_level": working_df.groupby("symbol").apply(summarize).reset_index(),
        "buy_level": summarize(working_df.loc[working_df["predicted_signal"] == "buy"]).to_frame().T,
        "sell_level": summarize(working_df.loc[working_df["predicted_signal"] == "sell"]).to_frame().T,
        "reasonableness_level": summarize(
            working_df.loc[working_df[reasonableness_col] > high_reasonableness_threshold]
        ).to_frame().T,
    }

    return summary_dict



if __name__ == "__main__":
    load_dotenv()
    
    print(is_today_monday())
    
    today_date, last_monday_date = get_today_and_last_monday_date_strings()
    
    etf_df = get_stock_bars(last_monday_date, today_date)
    
    etf_df = build_etf_vwap_future_changes(etf_df)
    
    published_df = load_dynamodb_table_by_date_range("processed_records", start_date=last_monday_date, end_date=today_date)

    published_df = duplicate_posts_to_minute_boundaries(
        df=published_df,
        datetime_column="created_at",
        post_duplicate=False,
    )
    
    published_df["created_at"] = pd.to_datetime(published_df["created_at"]) - pd.Timedelta(days=3, hours=12)
    
    weekly_evaluation_df = published_df.merge(
        etf_df,
        left_on=["symbol", "created_at"],
        right_on=["symbol", "timestamp"],
        how="inner",
    )

    print(weekly_evaluation_df)
    
    performance_summary = build_prediction_performance_summary(weekly_evaluation_df)
    
    ### SNS