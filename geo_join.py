import pandas as pd
import numpy as np
import geopandas as gpd


def geojoin(df, shape,
                 dataset_target, #
                 dataset_features_cum,
                 dataset_features_stat,
                 dataset_features_dyn,
                 dataset_features_dyn_ratio
                ):
    """
    Создание датафрейма для географических визуализаций. ВНИМАНИЕ: Ваш датасет должен иметь колонки ISO и COUNTRY. 

    Args:
        df(pd.DataFrame): Датафрейм с Вашими данными.
        shape(gpd.GeoDataFrame): SHP-файл.
        dataset_target(str): название важного признака, ради которого все и затевалось.
        dataset_features_cum(list): Кумулятивные признаки ("Всего заболело" и т.д.).
        dataset_features_stat(list): Статистические признаки ("ВВП на душу населения" и т.д.)
        dataset_features_dyn(list): Признаки события ("Количество участников" и т.д.)
        dataset_features_dyn(list): Признаки события, для которых нужно посчитать доли ("Доля")

    Returns:
        gpd.GeoDataFrame: Агрегированный по странам датафрейм.
    """
     
    df_sum = df[[dataset_target, "ISO", "COUNTRY"]].groupby(by = ["ISO", "COUNTRY"]).sum()
    
    for feature in dataset_features_cum:
        df_sum[feature] = df.groupby(["ISO", "COUNTRY"])[feature].last().fillna(method="ffill")
        
    for feature in dataset_features_stat:
        df_sum[feature] = df.groupby(["ISO", "COUNTRY"])[feature].mean()
    
    for feature in dataset_features_dyn:
        df_sum[feature] = df.groupby(["ISO", "COUNTRY"])[feature].sum()
    
    for feature in dataset_features_dyn_ratio:
        df_sum[feature + "_RATIO"] = df.groupby(["ISO", "COUNTRY"])[feature].sum() / df.groupby(["ISO", "COUNTRY"])[feature].count()
    
    df_sum.reset_index(inplace = True)
    df_sum["ISO"].astype("int64")    
    
    df_geo = shape[["ISO_N3", "geometry", "NAME"]]
    df_geo.loc[:, "ISO"] = pd.to_numeric(df_geo["ISO_N3"]) #Делаем столбец ISO одинаковым для обоих датасетов
    df_geo = df_geo.drop("ISO_N3", axis = 1)
    
    #стыковка ISO
    join_dict = df_sum.set_index("COUNTRY")["ISO"].to_dict()
    iso_unstacked = list(set(df_geo["ISO"]) - set(df_sum["ISO"]))

    for iso in iso_unstacked:
        mask = df_geo["ISO"] == iso
        df_geo.loc[mask, "ISO"] = df_geo.loc[mask, "NAME"].map(join_dict).values
        
    df_geo = pd.merge(df_geo, df_sum, on="ISO")
    return df_geo