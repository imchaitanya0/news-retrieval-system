import polars as pl
from src.data.build_pipeline import parse_mind_behaviors

df = pl.DataFrame({
    "impression_id": ["1"],
    "user_id": ["U1"],
    "time": ["11/11/2019 9:05:58 AM"],
    "history": ["N1 N2 N3"],
    "impressions": ["N4 N5 N6"]
})

df = parse_mind_behaviors(df)
print(df.select("impressions", "labels"))

df2 = pl.DataFrame({
    "impression_id": ["2"],
    "user_id": ["U2"],
    "time": ["11/11/2019 9:05:58 AM"],
    "history": ["N1 N2 N3"],
    "impressions": ["N4-0 N5-1 N6-0"]
})
df2 = parse_mind_behaviors(df2)
print(df2.select("impressions", "labels"))
