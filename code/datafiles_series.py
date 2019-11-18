import pandas as pd
import numpy as np

def csv_to_series(data_file):
    df = pd.read_csv(data_file, sep = '\t')
    idx = np.unique(df['ID'].values)
    time = np.unique(df['Time'].values)

    df1 = pd.DataFrame(np.nan, index = idx, columns = time)

    for ID in idx:
        t = df[df['ID']==ID]['Time'].values
        obs = df[df['ID']==ID]['Observation'].values
        for i in range(len(t)):
            df1.set_value(ID, t[i], obs[i])
    return df1.stack()

def series_to_csv(filename,
                  data,
                  N = 3):
    # data must be a series with ID, Time, Observation
    ID_vec = np.unique(data.index.get_level_values(0))
    f = open(filename, 'w')
    f.write(('ID' + '\t' + 'Time' + '\t' + 'Observation' + '\n'))
    for ID in ID_vec:
        for i in range(len(data[ID].index)):
            t = data[ID].index[i]
            value = data[ID].values[i]
            Vin = data[ID].values[0]
            tin = data[ID].index[0]
            f.write((np.str(ID) + '\t' + np.str(t) + '\t' + np.str(value) + '\n'))
    f.close()