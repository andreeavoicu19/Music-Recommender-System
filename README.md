# Music-Recommender-System
Machine Learning algorithm based on sound processing and audio feature extraction 

This project studies music information retrieval (MIR) based on audio signal analysis and the importance and applicability of the data obtained in recommender systems. The trained model assists the user in identifying songs that may become of interest depending on their audio content, such as unique spectral and timbral features of audio signals. 

The algorithm is applied on the GTZAN Dataset, a public dataset used for evaluation in machine listening research for music genre recognition and focuses on grouping audio files by similar audio features, using unsupervised learning, through "k-Means” clustering technique. 

When the user chooses a song, a playlist of songs with similar audio properties is recommended and updated after each choice. Thus, the purpose of this recommender system is generating songs that may be of interest to the user in the future, regardless of musical genre.

Libraries mainly used :

- Librosa, for sound processing and audio feature extraction 
- Pandas, for data analysis and manipulation.

Eventually, the algorithm was integrated into a web API using Django framework.
