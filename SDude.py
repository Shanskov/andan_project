import pandas as pd
import numpy
import geopandas


def dataclear(df,
             df_addition,
             df_fiw,
             useless_features = None,
             na_limit = 5
             ):
    """
    Предобработка ACLED датасета для анализа.

    Args:
        df(pd.DataFrame): Датафрейм с основными данными (ACLED).
        df_addition(pd.DataFrame): Датафрейм с дополнительными данными.
        df_fiw(pd.DataFrame): Датафрейм с индексами свобод (Freedom House).
        useless_features(list): Список ненужных колонок.
        na_limit(int): Процент допустимых пропусков в колонках.
        
    Returns:
        pd.DataFrame: Обработанный датафрейм.
    """
    
    df.set_index("EVENT_ID_CNTY", inplace = True)
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'])
    df['ACTOR1_TYPES'] = df["ASSOC_ACTOR_1"].fillna(df["ACTOR1"])
    
    
    def process_string(row):
        new_string = row['ACTOR1_TYPES'].replace(' (' + row['COUNTRY'] + ')', '')
        return [new_string]
    df['ACTOR1_TYPES'] = df.apply(lambda row: pd.Series(process_string(row)), axis=1)
    df_actorcheck = df["ACTOR1_TYPES"].str.split(",", expand = True)
    
    #Список участников, не относящихся к формальным объединениям.
    
    my_list = ["Protesters", "Labor Group", "Health Workers", "Students", "Rioters", "Teachers", "Farmers", "Women", "Prisoners", "Taxi/Bus Drivers", "Lawyers", "Taxi Drivers", "No Vax", "Muslim Group", "Fishers", "Journalists", "Street Traders", "Orthodox Christian Group", "Protestant Christian Group", "Haredi Jewish Group", "Judges", "Refugees/IDPs", None]
    
    def check_list(row):
        return int(all(elem in my_list for elem in row))
    df['UNORGANIZED'] = df_actorcheck.apply(lambda row: check_list(row), axis=1)
    
    #работа со вторым датасетом
    df_addition['date'] = pd.to_datetime(df_addition['date'])
    df_addition["location"].replace(to_replace = "Czechia", value = "Czech Republic", inplace = True) #оправданный костыль
    
    #работа с третьим датасетом
    df_fiw.columns = df_fiw.iloc[0].values
    df_fiw["Country/Territory"].replace(to_replace = "Congo (Kinshasa)", value = 'Democratic Republic of Congo', inplace = True) #оправданный костыль 2
    
    #стыковка трех датасетов по общим столбцам
    df_addition["LOCDATE"] = df_addition["location"] + df_addition["date"].astype("str")
    df["LOCDATE"] = df["COUNTRY"] + df["EVENT_DATE"].astype("str")
    df_data = pd.merge(df, df_addition, on = "LOCDATE", how = "left")
    df_data["LOCYEAR"] = df_data["COUNTRY"] + df_data["YEAR"].astype("str")
    df_fiw["LOCYEAR"] = df_fiw["Country/Territory"] + df_fiw["Edition"].astype("str")
    df_data = pd.merge(df_data, df_fiw, on="LOCYEAR")
    
    #очистка ненужных столбцов
    df_data = df_data.drop([
    'YEAR',
    'NOTES', #Текстовое описание новости
    'SOURCE', #Источник
    'TIME_PRECISION', #Точность определения времени события
    'TIMESTAMP', #Как оказалось, это время внесения наблюдения в таблицу
    'GEO_PRECISION', #Точность географической оценки
    'TAGS', #Плохо сделанные теги
    'ACTOR1', 
    'ASSOC_ACTOR_1', #Действующие лица
    'ACTOR2', 
    'ASSOC_ACTOR_2', 
    'CIVILIAN_TARGETING', #Было ли направленное именно на гражданские лица насилие
    'ADMIN1', 
    'ADMIN2',
    'ADMIN3', #Место действия
    'LOCATION',
    'INTERACTION',
    'ACTOR1_TYPES', #Служебные колонки
    'LOCDATE',
    'LOCYEAR',
    'Country/Territory', #Повторки
    'Region',
    'C/T',
    'Edition',
    'Status',
    'iso_code',
    'continent',
    'date'
    ], axis = 1)
    if useless_features != None:
        df_data = df_data.drop(useless_features, axis = 1)
    df_data = df_data.drop(df_data[df_data["location"].isna()].index)
    df_data = df_data.drop("location", axis = 1)
    df_data = df_data.drop(df_data.columns[df_data.isna().sum() > (na_limit * 0.01 * df_data.shape[0])], axis = 1)
    
    
    return(df_data)


###ГЕОГРАФИЧЕСКАЯ ОБРАБОТКА


def geo_join(df, shape,
            dataset_target,
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
        dataset_features_dyn_ratio(list): Признаки события, для которых нужно посчитать доли ("Доля неорганизованных событий")

    Returns:
        gpd.GeoDataFrame: Агрегированный по странам датафрейм.
    """

    df_sum = df[[dataset_target, "ISO", "COUNTRY"]].groupby(by=["ISO", "COUNTRY"]).sum()

    for feature in dataset_features_cum:
        df_sum[feature] = df.groupby(["ISO", "COUNTRY"])[feature].last().fillna(method="ffill")

    for feature in dataset_features_stat:
        df_sum[feature] = df.groupby(["ISO", "COUNTRY"])[feature].mean()

    for feature in dataset_features_dyn:
        df_sum[feature] = df.groupby(["ISO", "COUNTRY"])[feature].sum()

    for feature in dataset_features_dyn_ratio:
        df_sum[feature + "_RATIO"] = df.groupby(["ISO", "COUNTRY"])[feature].sum() / df.groupby(["ISO", "COUNTRY"])[
            feature].count()

    df_sum.reset_index(inplace=True)
    df_sum["ISO"].astype("int64")

    df_geo = shape[["ISO_N3", "geometry", "NAME"]]
    df_geo.loc[:, "ISO"] = pd.to_numeric(df_geo["ISO_N3"])  # Делаем столбец ISO одинаковым для обоих датасетов
    df_geo = df_geo.drop("ISO_N3", axis=1)

    # стыковка ISO
    join_dict = df_sum.set_index("COUNTRY")["ISO"].to_dict()
    iso_unstacked = list(set(df_geo["ISO"]) - set(df_sum["ISO"]))

    for iso in iso_unstacked:
        mask = df_geo["ISO"] == iso
        try:
            df_geo.loc[mask, "ISO"] = df_geo.loc[mask, "NAME"].map(join_dict).values
        except(AttributeError, TypeError):
            raise("Все сломалось")
    df_geo = pd.merge(df_geo, df_sum, on="ISO")
    return df_geo