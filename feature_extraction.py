import librosa
import os
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import plotly.express as px
from pycaret.classification import *
from sklearn.metrics.pairwise import cosine_similarity
import time
import seaborn as sns


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=12, rms=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'var')

    columns = []

    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i + 1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # Sortare indecsi coloane
    return columns.sort_values()


def features(dirpath, tid):
    features = pd.Series(index=columns(), dtype=np.float64, name=tid)

    # "catch warnings as exceptions (audioread leaks file descriptors)"
    warnings.filterwarnings('ignore', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'var'] = stats.variation(values, axis=1)

    try:
        dir = dirpath + os.listdir(dirpath)[tid]
        # kaiser_fast = parametru opțional pentru a reduce timpul de încărcare al fișierelor
        y, sr = librosa.load(dir, sr=22050, mono=True, res_type='kaiser_fast')

        # Elimina portiuni silentioase inainte si dupa audio
        tid, _ = librosa.effects.trim(y)

        # Detalii audio despre fisier
        # print("Audio file", tid, '\n')
        # print("Audio shape: ", np.shape(tid))
        # print('Sample Rate (KHz):', sr, '\n')
        # length = np.shape(y)[0] / sr
        # print('Audio file length:', int(length), 's')

        # Calculul trasaturilor spectrale
        # Zero Crossing-Rate
        f = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        # BPM (Beats Per Minute)
        tempo, _ = librosa.beat.beat_track(y, sr=sr)
        # print('Song Sample Tempo :', int(tempo), 'bpm')

        index_cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7 * 12, tuning=None))
        assert index_cqt.shape[0] == 7 * 12
        assert np.ceil(len(y) / 512) <= index_cqt.shape[1] <= np.ceil(len(y) / 512) + 1

        # Proprietati Chroma
        f = librosa.feature.chroma_cqt(C=index_cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=index_cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)

        # Tonnetz (Vector Central Tonal)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)
        del index_cqt

        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(y) / 512) <= stft.shape[1] <= np.ceil(len(y) / 512) + 1
        del y

        # Chroma Stft
        f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rms(S=stft)
        feature_stats('rms', f)

        # Trasaturi spectrale particulare
        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        del stft

        # MFCC (Coeficienti Mel Cepstrali)
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=12)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))

    return features


def main():
    dirpath = 'wav_data_500/'
    nfiles = len([name for name in os.listdir(dirpath)])
    df_features = pd.DataFrame(columns=columns(), dtype=np.float64)

    # start = time.clock()

    for i in range(nfiles):
        f = features(dirpath, i)
        df_features.loc[f.name] = f
        # print('Sample Name :', os.listdir(dirpath)[i])
    # print('Timp de incarcare fisiere', time.clock() - start)
    print('Dataframe initial de caracteristici cu valori neprelucrate :', df_features)

    # Salveaza feature-urile calculate in .csv
    save(df_features, 10)

    # Citeste trasaturile calculate din .csv pentru a evita conflicte de coloane
    data_csv = pd.read_csv("features.csv", index_col=0)
    # print('Coloane', data.columns)

    # Elimina valorile NaN
    data_csv = data_csv.dropna(axis='columns')
    print('Dataframe de caracteristici dupa eliminarea valorilor NaN :', data_csv)

    # Selecteaza valorile numerice din dataframe
    features_array = np.array(df_features.loc[:, :])
    print('Vectori de caracteristici : ', features_array)

    # Clusterizare k-Means si standardizarea datelor in pipeline
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5,
                                                                                      verbose=2, n_jobs=4))],
                                                                                        verbose=True)

    # Selecteaza toate coloanele mai putin 'length' (nu e caracteristica relevanta de asemanare)
    # features_data = scaled_features_df.drop(columns=['length'])

    # Primeste doar coloanele cu valorile trasaturilor
    # names_columns = features_data['filename']
    # print('filename', names_columns)
    col = data_csv.iloc[3: 9, 0: 132]
    print('Columns :', col)
    # columns = list(col.columns)
    # print('Columns', columns)

    # Salveaza modelul
    pickle.dump(song_cluster_pipeline, open('cluster_model.sav', 'wb'))

    # Selectam datele pentru normalizare
    features_data = np.array(df_features.loc[:, :])
    print('Features data', features_data)
    X = np.vstack(features_data)
    print('X :', X)
    col = data_csv.iloc[3: 9, 0: 132]
    number_cols = list(col.columns)
    print('Col', col)

    # Normalizarea datelor
    data_scaled = X
    scaler = preprocessing.StandardScaler()
    # Salvez scaler-ul pentru model
    pickle.dump(scaler, open("scaler.sav", "wb"))
    data_scaled = scaler.fit_transform(data_scaled)
    scaled_features_df = pd.DataFrame(data_scaled, columns=number_cols)
    print("Scaled features frame :", scaled_features_df)

    # Adauga coloana 'filename' in dataframe
    wavs = []
    for i in range(nfiles):
        wavs.append(os.listdir(dirpath)[i])
    scaled_features_df['filename'] = wavs

    # Adauga coloana 'tempo' in dataframe
    bpm = []
    for tid in range(nfiles):
        dir = dirpath + os.listdir(dirpath)[tid]
        y, sr = librosa.load(dir, sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y, sr=sr)
        # print('Song Sample Tempo :', int(tempo), 'bpm')
        bpm.append(int(tempo))
    scaled_features_df['tempo'] = bpm

    # Plot cu exprimarea genurilor in BPM
    x = scaled_features_df[["filename", "tempo"]]
    print('x', x)
    plt.subplots(figsize=(16, 9))
    sns.boxplot(x="filename", y="tempo", data=x)

    plt.title('Clasificarea genurilor în funcție de BPM', fontsize=25)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel("Gen", fontsize=15)
    plt.ylabel("Bătăi pe minut (BPM)", fontsize=15)
    plt.savefig('clasificare_genuri_mydata.png')
    plt.close()

    # Metoda "Elbow"
    sse = {}
    # Trece prin valori multiple ale lui k pentru a gasi valoarea sa ideala
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100).fit(X)
        df_features["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_
    # Ploteaza graficul cu estimarea erorii (SSE) in functie de valorile lui k
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("k - număr de clustere")
    plt.ylabel("Estimarea erorii")
    # Salveaza graficul in directorul curent
    plt.savefig('elbow_method.png')

    print('Dataframe cu coloanele filename si tempo adaugate :', scaled_features_df)

    # Toate coloanele df-ului
    # all_columns = list(scaled_features_df.columns)
    # print('Scaled data columns', all_columns)

    # Antreneaza datele scalate (vectorul)
    song_cluster_pipeline.fit(data_scaled)
    song_cluster_labels = song_cluster_pipeline.fit_predict(data_scaled)

    # Adauga in tabel coloana cu clusterul din care face parte piesa
    # label-ul piesei = nr. clusterului
    scaled_features_df['cluster_label'] = song_cluster_labels
    # scaled_features_df['tempo'] = tempo

    # Label pentru fiecare .wav
    # print('Numar cluster : ', song_cluster_labels)

    # Vizualizarea clusterelor cu PCA
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = scaled_features_df['filename']
    projection['tempo'] = scaled_features_df['tempo']
    # Primii 13 coeficienti Mel cepstrali
    projection['chroma_cens'] = scaled_features_df['chroma_cens']
    projection['chroma_cens.1'] = scaled_features_df['chroma_cens.1']
    projection['chroma_cens.2'] = scaled_features_df['chroma_cens.2']
    projection['chroma_cens.3'] = scaled_features_df['chroma_cens.3']
    projection['chroma_cens.4'] = scaled_features_df['chroma_cens.4']
    projection['chroma_cens.5'] = scaled_features_df['chroma_cens.5']
    projection['chroma_cens.6'] = scaled_features_df['chroma_cens.6']
    projection['chroma_cens.7'] = scaled_features_df['chroma_cens.7']
    projection['chroma_cens.8'] = scaled_features_df['chroma_cens.8']
    projection['chroma_cens.9'] = scaled_features_df['chroma_cens.9']
    projection['chroma_cens.10'] = scaled_features_df['chroma_cens.10']
    projection['chroma_cens.11'] = scaled_features_df['chroma_cens.11']
    projection['chroma_cens.12'] = scaled_features_df['chroma_cens.12']
    projection['chroma_cqt'] = scaled_features_df['chroma_cqt']
    projection['chroma_cqt.1'] = scaled_features_df['chroma_cqt.1']
    projection['chroma_stft'] = scaled_features_df['chroma_stft']
    projection['chroma_stft.1'] = scaled_features_df['chroma_stft.1']
    projection['mfcc'] = scaled_features_df['mfcc']
    projection['mfcc.1'] = scaled_features_df['mfcc.1']
    projection['mfcc.2'] = scaled_features_df['mfcc.2']
    projection['mfcc.3'] = scaled_features_df['mfcc.3']
    projection['mfcc.4'] = scaled_features_df['mfcc.4']
    # media
    projection['rms'] = scaled_features_df['rms']
    # .1 = varianța
    projection['rms.1'] = scaled_features_df['rms.1']
    projection['spectral_bandwidth'] = scaled_features_df['spectral_bandwidth']
    projection['spectral_bandwidth.1'] = scaled_features_df['spectral_bandwidth.1']
    projection['spectral_centroid'] = scaled_features_df['spectral_centroid']
    projection['spectral_centroid.1'] = scaled_features_df['spectral_centroid.1']
    projection['spectral_contrast'] = scaled_features_df['spectral_contrast']
    projection['spectral_contrast.1'] = scaled_features_df['spectral_contrast.1']
    projection['spectral_rolloff'] = scaled_features_df['spectral_rolloff']
    projection['spectral_rolloff.1'] = scaled_features_df['spectral_rolloff.1']
    projection['tonnetz'] = scaled_features_df['tonnetz']
    projection['tonnetz.1'] = scaled_features_df['tonnetz.1']
    projection['zcr'] = scaled_features_df['zcr']
    projection['zcr.1'] = scaled_features_df['zcr.1']
    projection['cluster'] = scaled_features_df['cluster_label']

    fig = px.scatter(projection, x='x', y='y', color='cluster',
                     hover_data=['x', 'y', 'title', 'tempo', 'chroma_cens.1', 'chroma_cens.2', 'chroma_cens.3',
                                 'chroma_cens.4', 'chroma_cens.5', 'chroma_cens.6', 'chroma_cens.7', 'chroma_cens.8',
                                 'chroma_cens.9', 'chroma_cens.10', 'chroma_cens.11', 'chroma_cens.12', 'chroma_cqt',
                                 'chroma_cqt.1', 'chroma_stft', 'chroma_stft.1', 'mfcc', 'mfcc.1', 'mfcc.2',
                                 'mfcc.3', 'mfcc.4', 'rms', 'rms.1', 'spectral_bandwidth', 'spectral_bandwidth.1',
                                 'spectral_centroid', 'spectral_centroid.1', 'spectral_contrast', 'spectral_contrast.1',
                                 'spectral_rolloff', 'spectral_rolloff.1', 'tonnetz', 'tonnetz.1', 'zcr', 'zcr.1'
                                 ])
    fig.show()

    # Pregateste datele pentru cosine similarity
    # Setam index filename prima coloana
    scaled_features_df = scaled_features_df.set_index('filename')
    print('Dataframe cu numele fisierului ca index', scaled_features_df)

    # Retine clusterele prezise pt fiecare .wav
    labels = scaled_features_df['cluster_label']
    data_scaled_df = scaled_features_df.select_dtypes(np.number)  # echivalentul lui X din codul GTZAN

    print("data_scaled_df echivalentul lui X", data_scaled_df)

    # Standardizarea datelor pentru metoda "cosine similarity"
    data_scaled = song_cluster_pipeline.fit_transform(data_scaled_df)

    # Noile valori scalate
    print('Valori scalate pentru aplicare cosine similarity:', data_scaled)
    # print('labels index', labels.index)

    # Recomandare
    # Calculeaza similaritatea dintre fisiere
    similarity = cosine_similarity(data_scaled_df)
    # print("Similarity shape:", similarity.shape)

    # Converteste in dataframe si seteaza indexul randului si al coloanei din labels
    sim_df_labels = pd.DataFrame(similarity)
    # print(sim_df_labels)

    # Primeste numele fisierelor
    sim_df_names = sim_df_labels.set_index(labels.index)
    sim_df_names.columns = labels.index

    print('Repartitia fisierelor audio pe clustere :', scaled_features_df['cluster_label'])

    def find_similar_songs(filename):
        # Afiseaza de la cel mai similar fisier
        series = sim_df_names[filename].sort_values(ascending=True)

        # Elimina cosine similarity == 1 (o piesa se va potrivi cu ea insasi, deci eliminam reafisarea acesteia)
        series = series.drop(filename)

        print("Fișiere audio similare în caracteristici cu", filename)
        print(series.sample(15))

    # print('Timp de incarcare fisiere', int(time.clock() - start), 's')
    # Returneaza indecsii pieselor din dataframe
    find_similar_songs('classical.00019.wav')

    # with open('finalized_`model`.sav', 'rb') as fid:
    # pca_pipeline = pickle.load(fid)


def save(features, ndigits):
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv('features.csv', float_format='%.{}e'.format(ndigits))


if __name__ == "__main__":
    main()
