import matplotlib.pyplot as plt
import polars as pl
import matplotlib
import seaborn as sns
import numpy
import pandas as pd

"""
Доли невернувших по критерию наличия имущества
total_ds = pl.read_csv("ds/application_train.csv", try_parse_dates= True)
result_all = (total_ds.select(
    pl.col("TARGET"),
    pl.col("FLAG_OWN_CAR"),
    pl.col("FLAG_OWN_REALTY")
).group_by(pl.col("FLAG_OWN_CAR"), pl.col("FLAG_OWN_REALTY")).agg(pl.col("TARGET").count())
              .sort(pl.col("FLAG_OWN_CAR"),pl.col("FLAG_OWN_REALTY")))
total_all = [result_all[0,2], result_all[1,2],result_all[2,2],result_all[3,2]]
result_no_ret = (total_ds.filter(pl.col("TARGET")==1).select(
    pl.col("TARGET"),
    pl.col("FLAG_OWN_CAR"),
    pl.col("FLAG_OWN_REALTY")
).group_by(pl.col("FLAG_OWN_CAR"), pl.col("FLAG_OWN_REALTY")).agg(pl.col("TARGET").count())
                 .sort(pl.col("FLAG_OWN_CAR"),pl.col("FLAG_OWN_REALTY")))
total_no_ret = [result_no_ret[0,2], result_no_ret[1,2],result_no_ret[2,2],result_no_ret[3,2]]
total = [(total_no_ret[0]/total_all[0] * 100),(total_no_ret[1]/total_all[1] * 100),(total_no_ret[2]/total_all[2] * 100),(total_no_ret[3]/total_all[3] * 100)]
names = ["havent_all", "have_realty","have_car","have_all"]

plt.bar(names, total)
plt.title("Доли невернувших по критерию наличия имущества")
plt.show()
print(total)
"""

"""
Доли невернувших по возрастному критерию
total_ds = pl.read_csv("ds/application_train.csv", try_parse_dates= True)
result_all = total_ds.select(
    pl.col("TARGET"),
    (pl.col("DAYS_BIRTH")*-1//365).cut([17,25,30,45,50], labels=["18-","18-25","25-30","30-45","45-50","50+"]).alias("decade")
).group_by("decade").agg(pl.col("TARGET").count().alias("counter")).sort(pl.col("decade"))
names = ["18-25", "25-30", "30-45", "45-50", "50+"]
total_all = [result_all[0,1], result_all[1,1],result_all[2,1],result_all[3,1],result_all[4,1]]
result_no_ret = total_ds.filter(pl.col("TARGET")==1).select(
    pl.col("TARGET"),
    (pl.col("DAYS_BIRTH")*-1//365).cut([17,25,30,45,50], labels=["18-","18-25","25-30","30-45","45-50","50+"]).alias("decade")
).group_by("decade").agg(pl.col("TARGET").count().alias("counter")).sort(pl.col("decade"))
total_no = [result_no_ret[0,1], result_no_ret[1,1],result_no_ret[2,1],result_no_ret[3,1],result_no_ret[4,1]]
total = [total_no[0]/total_all[0] * 100, total_no[1]/total_all[1] * 100,
         total_no[2]/total_all[2] * 100,total_no[3]/total_all[3] * 100,total_no[4]/total_all[4] * 100]
print(total)
plt.bar(names, total)
plt.title("Доля невернушвших по возрастам")
plt.show()
"""

"""
total_ds = pl.read_csv("ds/application_train.csv", try_parse_dates= True)
result = total_ds.select(
    pl.col("TARGET"),
    pl.col("NAME_INCOME_TYPE"),
).group_by(pl.col("NAME_INCOME_TYPE")).agg(pl.col("TARGET").count().alias("total"), pl.col("TARGET").sum().alias("no_ret"))

result = result.with_columns(
    (pl.col("no_ret")/pl.col("total")*100).alias("percent")
)
sns.barplot(data = result, x = "percent", y = "NAME_INCOME_TYPE",)
plt.title("Доля невозвратов по типу занятости")
print(result)
"""

"""
heatmap по типу кредита, типу занятости и доле невоозвратов
total_ds = pl.read_csv("ds/application_train.csv", try_parse_dates= True)
result = total_ds.select(
    pl.col("TARGET"),
    pl.col("NAME_CONTRACT_TYPE"),
    pl.col("NAME_INCOME_TYPE")
).group_by(pl.col("NAME_CONTRACT_TYPE"), pl.col("NAME_INCOME_TYPE")).agg(pl.col("TARGET").count().alias("total"),
            pl.col("TARGET").sum().alias("no_ret"))
result = result.with_columns(
    (pl.col("no_ret") / pl.col("total") * 100).alias("percent")
)

heatmap_result = result.pivot(
    index="NAME_INCOME_TYPE",
    on="NAME_CONTRACT_TYPE",
    values="percent"
)

heatmap_pd = heatmap_result.to_pandas().set_index("NAME_INCOME_TYPE")

plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_pd,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    cbar_kws={'label': 'Доля невозвратов (%)'},
    linewidths=0.5,
    linecolor='white'
)
print(heatmap_pd)
plt.tight_layout()
plt.show()

"""
"""
Важный инсайт!
barplot доли невозвратов по возрасту и типу кредита
total_ds = pl.read_csv("ds/application_train.csv", try_parse_dates= True)
result_all = total_ds.select(
    pl.col("TARGET"),
    (pl.col("DAYS_BIRTH")*-1//365).cut([17,25,30,45,50],
                                       labels=["18-","18-25","25-30","30-45","45-50","50+"]).alias("decade"),
    pl.col("NAME_CONTRACT_TYPE")
).group_by(pl.col("decade"), pl.col("NAME_CONTRACT_TYPE")).agg(pl.col("TARGET").count().alias("counter"),
                         pl.col("TARGET").sum().alias("no_ret")).sort(pl.col("decade"))
result_all = result_all.with_columns(
    (pl.col("no_ret")/pl.col("counter")*100).alias("no_ret_percent")
)

result_all_pd = result_all.to_pandas()

sns.barplot(data = result_all_pd, x = "no_ret_percent", y = "decade",hue="NAME_CONTRACT_TYPE")
plt.title("Зависимость невозвратов от типа кредита и возраста")
plt.show()
"""

total_ds = pl.read_csv("ds/application_train.csv", try_parse_dates= True)
result = total_ds.select(
    pl.col("NAME_CONTRACT_TYPE"),
    pl.col("AMT_INCOME_TOTAL"),
    pl.col("AMT_CREDIT")
).group_by(pl.col("NAME_CONTRACT_TYPE")).agg(pl.col("AMT_INCOME_TOTAL").median().alias("mean_income_total"),
    pl.col("AMT_CREDIT").median().alias("credit_mean"),
    (pl.col("AMT_CREDIT")/pl.col("AMT_INCOME_TOTAL")).median().alias("credit_to_income_mean"))
print(result)

