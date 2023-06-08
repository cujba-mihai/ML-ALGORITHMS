from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from sklearn.cluster import KMeans
from decision_tree import DecisionTreeEndpoint
from apriori import Apriori
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO


app = Flask(__name__)
CORS(app)
api = Api(app)

api.add_resource(Apriori, '/apriori')
api.add_resource(DecisionTreeEndpoint, '/decision_tree')


def generate_kmeans_plot(data_points, labels):
    plt.scatter(data_points[:,0], data_points[:,1], c=labels)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

@app.route('/kmeans', methods=['POST'])
def perform_kmeans():
    data = request.get_json()
    data_points = np.array(data['data_points'])
    n_clusters = int(data['n_clusters'])
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, max_iter=1, random_state=42)
    iterations = []

    for i in range(10):
        kmeans.fit(data_points)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        plot_image = generate_kmeans_plot(data_points, labels)
        iteration = {'iteration': i+1, 'labels': labels.tolist(), 'centroids': centroids.tolist(), 'plot_image': f'data:image/png;base64,{plot_image}'}
        iterations.append(iteration)

    response = {'iterations': iterations}
    return jsonify(response)

@app.route("/")
def home_view():
        return "<h1>Hello World!</h1>"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
