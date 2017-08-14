from sklearn.cluster import KMeans
import pickle
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from models.w2vecModel import Word2vecModel

if __name__ == '__main__':

    w2v_model = Word2vecModel()
    w2v_model.initialize("../data/glove.6B/glove.6B.50d.txt")

    author_df = pd.read_csv('../data/authors_processed.csv')
    doc_list = [t.split() for t in author_df['doc']]

    X = w2v_model.transform(doc_list)

    ## Visualizing on first to PCs
    # THIS PART IS TAKEN FROM SKLEARN SITE

    print ('reducing the dimension to first two PCs...')
    reduced_data = PCA(n_components=2).fit_transform(X)

    print ('Training Kmeans...')
    kmeans = KMeans(init='k-means++', n_clusters=100, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    print('Making the image...')
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap='Paired',
               aspect='auto', origin='lower')

    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on embedding space (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
