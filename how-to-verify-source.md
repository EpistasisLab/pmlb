There are a few ways we can check whether a PMLB dataframe (`pmlb_df`) agrees with its source (`source_df`), provided that we have checked their shapes (by printing `pmlb_df.shape` and `pmlb_df.shape`).

- If the two dataframes are exactly the same, the following line of code does not return anything ✔️:
  ``` python
  pd.testing.assert_frame_equal(df_source, df_pmlb)
  ```

- If it gives error, the column names may be different. If we have good reasons to ignore column names, we can check if the values contained in the 2 dataframes are the same with
  ``` python
  from pandas.util import hash_pandas_object
  import hashlib

  rowhashes_pmlb = hash_pandas_object(df_pmlb, index = False).values 
  hash_pmlb = hashlib.sha256(rowhashes_pmlb).hexdigest()
  rowhashes_source = hash_pandas_object(df_source, index = False).values 
  hash_source = hashlib.sha256(rowhashes_source).hexdigest()

  # verify hashes match
  print(hash_pmlb == hash_source)
  ```
  or 
  ``` python
  (df_source.values == df_pmlb.values).all()
  ```
- If we still get `False`, it is possible that the rows have been permuted. To check if they are:
  ``` python
  set(df_pmlb.itertuples(index=False)) == set(df_source.itertuples(index=False))
  ```
  or, if row hashes have been computed,
  ``` python
  set(rowhashes_source) == set(rowhashes_pmlb)
  ```
  or "subtracting" the two datasets row by row
  ``` python
  df_source.merge(df_pmlb, indicator = True, how='left').loc[lambda x : x['_merge']!='both']
  df_pmlb.merge(df_source, indicator = True, how='left').loc[lambda x : x['_merge']!='both']
  ```
  This code will print the rows that are in one dataframe but not the other and can help you see the difference a bit better.
- If the two dataframes have floats that are almost equal to each other, we can use `numpy`'s `isclose` to check if they are element-wise equal within a tolerance:
  ``` python
  from numpy import isclose

  isclose(df_source.values, df_pmlb.values).all()
  ```

We have been using [Google Colab notebooks](https://colab.research.google.com/) to share our checks, but other methods are also welcomed.
Here are a few existing notebooks for reference:
[wine-quality-red](https://colab.research.google.com/drive/1N48BWz6IdeyIDUM3ROhd1wUPjhhL-Vz4#scrollTo=yxujo7a_gjMV),
[wine-quality-white](https://colab.research.google.com/drive/1z_aFLydv2xMjDWwYIGGbW5N8_XFraysT),
[waveform-mushroom-saheart](https://colab.research.google.com/drive/1DyB2oqenINVmJzFLkwjPKYv0iAb5Mz02#scrollTo=5QZDL8Yffx62),
[irish](https://colab.research.google.com/drive/1gB7r_CN8LrWG3nOqCS3AXJ7enj_Ssavk?usp=sharing#scrollTo=ioB2C8bb_WGa),
[adult](https://colab.research.google.com/drive/1s2J0v2Ubzj0-CxzgQnxdmAAVoK33a1AY#scrollTo=-gBzhYeQMi3t).
