from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pandas.io import pickle
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import plotly.express as px
from pycaret.classification import *
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def main():
    features_data = pd.read_csv("features_30_sec_GTZAN.csv")

    # Colecteaza datele pentru aflarea k-ului potrivit
    elbow_data = pd.read_csv("features_30_sec_GTZAN.csv")
    elbow_data = elbow_data.drop(columns=['filename', 'length'])

    print('Features Data : ', features_data)
    print('Coloanele dataframe-ului initial :', features_data.columns)

    # Clusterizare k-Means si standardizarea datelor in pipeline
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                      ('kmeans', KMeans(n_clusters=5,
                                                        verbose=2, n_jobs=4))], verbose=True)

    # Selecteaza toate coloanele mai putin 'length' (nu e caracteristica relevanta)
    features_data = features_data.drop(columns=['length'])

    # Selecteaza doar valorile numerice
    X = features_data.select_dtypes(np.number)
    print("X :", X)

    # Metoda "Elbow"
    sse = {}
    # Trece prin valori multiple ale lui k pentru a gasi valoarea sa ideala
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100).fit(X)
        elbow_data["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_
    # Ploteaza graficul cu estimarea erorii (SSE) in functie de valorile lui k
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("k - număr de clustere")
    plt.ylabel("Estimarea erorii")
    # Salveaza graficul in directorul curent
    plt.savefig('elbow_method_GTZAN.png')

    # Toate coloanele (inainte de scalarea datelor)
    # no_columns = list(X.columns)
    # print('Coloanele df-ului: ', no_columns)

    # Prezice clusterul (label-ul) pentru fiecare fisier audio .wav
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)

    # Adauga in tabel coloana cu clusterul din care face parte fiecare piesa
    features_data['cluster_label'] = song_cluster_labels
    print('Song cluster labels :', song_cluster_labels)

    # Adauga coloana cu numele fisierelor .wav
    # names_columns = features_data['filename']
    # print('Shape x', X.shape)

    # Vizualizarea clusterelor cu PCA
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = features_data['filename']
    projection['tempo'] = features_data['tempo']
    # Caracteristicile calculate in pachetul GTZAN
    projection['chroma_stft_mean'] = features_data['chroma_stft_mean']
    projection['chroma_stft_var'] = features_data['chroma_stft_var']
    projection['rolloff_mean'] = features_data['rolloff_mean']
    projection['rolloff_var'] = features_data['rolloff_var']
    projection['rms_mean'] = features_data['rms_mean']
    projection['rms_var'] = features_data['rms_var']
    projection['mfcc1_mean'] = features_data['mfcc1_mean']
    projection['mfcc1_var'] = features_data['mfcc1_var']
    projection['mfcc2_mean'] = features_data['mfcc2_mean']
    projection['mfcc2_var'] = features_data['mfcc2_var']
    projection['mfcc3_mean'] = features_data['mfcc3_mean']
    projection['mfcc3_var'] = features_data['mfcc3_var']
    projection['mfcc4_mean'] = features_data['mfcc4_mean']
    projection['mfcc4_var'] = features_data['mfcc4_var']
    projection['mfcc5_mean'] = features_data['mfcc5_mean']
    projection['mfcc5_var'] = features_data['mfcc5_var']
    projection['spectral_bandwidth_mean'] = features_data['spectral_bandwidth_mean']
    projection['spectral_bandwidth_var'] = features_data['spectral_bandwidth_var']
    projection['spectral_centroid_mean'] = features_data['spectral_centroid_mean']
    projection['spectral_centroid_var'] = features_data['spectral_centroid_var']
    projection['zero_crossing_rate_mean'] = features_data['zero_crossing_rate_mean']
    projection['zero_crossing_rate_var'] = features_data['zero_crossing_rate_var']
    projection['cluster'] = features_data['cluster_label']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster',
        hover_data=['x', 'y', 'title', 'tempo', 'chroma_stft_mean', 'chroma_stft_var',
                    'rolloff_mean', 'rolloff_var', 'rms_mean', 'rms_var',
                    'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var',
                    'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var',
                    'mfcc5_mean', 'mfcc5_var',
                    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                    'spectral_centroid_mean', 'spectral_centroid_var',
                    'zero_crossing_rate_mean', 'zero_crossing_rate_var'
                    ])
    fig.show()

    # Plot cu exprimarea genurilor in BPM
    x = features_data[["label", "tempo"]]
    print(x)
    plt.subplots(figsize=(16, 9))
    sns.boxplot(x="label", y="tempo", data=x)

    plt.title('Clasificarea genurilor în functie de BPM', fontsize=25)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel("Gen", fontsize=15)
    plt.ylabel("Bătăi pe minut (BPM)", fontsize=15)
    plt.savefig('clasificare_genuri_GTZAN.png')
    plt.close()

    # Salveaza modelul
    # filename = 'finalized_model.sav'
    # pickle.dump(pca_pipeline, open(filename, 'wb'))

    # Seteaza filename ca index
    features_data = pd.read_csv('features_30_sec_GTZAN.csv', index_col='filename')

    labels = features_data['label']
    # features_data = features_data.drop(columns=['length', 'label'])

    # print("X echivalentul lui data_scaled_df", X)

    # Standardizarea datelor pentru metoda "cosine similarity"
    data_scaled = song_cluster_pipeline.fit_transform(X)
    # Noile valori scalate
    print('Valori scalate pentru aplicare cosine:', data_scaled)

    """ 
    features_data = pd.DataFrame(data_scaled, columns=number_cols)
    print('Scaled data type:', type(data_scaled))
    # print('Scaled dataframe', features_data)
    """
    # Afiseaza numele .wav-urilor
    # print('labels index', labels.index)

    # Recommender - Cosine similarity
    similarity = cosine_similarity(data_scaled)
    # print("Similarity shape:", similarity.shape)

    # Converteste in dataframe si seteaza indexul randului si al coloanei din labels
    sim_df_labels = pd.DataFrame(similarity)
    # print(sim_df_labels)
    sim_df_names = sim_df_labels.set_index(labels.index)
    sim_df_names.columns = labels.index

    # print('sim_df_names', sim_df_names)

    def find_similar_songs(filename):
        # Afiseaza de la cel mai similar fisier
        series = sim_df_names[filename].sort_values(ascending=True)

        # Elimina cosine similarity == 1 (o piesa se va potrivi cu ea insasi)
        series = series.drop(filename)

        print("Fișiere audio similare în caracteristici cu", filename)
        print(series.sample(15))

    # Returneaza numele pieselor similare
    find_similar_songs('classical.00019.wav')

    # with open('finalized_model.sav', 'rb') as fid:
    # pca_pipeline = pickle.load(fid)


if __name__ == "__main__":
    main()
