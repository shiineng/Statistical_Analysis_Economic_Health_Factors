import pandas as pd
df = pd.DataFrame({"A": ["a", "b", "c"]})
print(df["A"].dtype)
print(type(df["A"].dtype))
