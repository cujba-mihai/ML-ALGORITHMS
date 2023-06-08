import base64
import pandas as pd
from flask import request, jsonify
from flask_restful import Resource
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io

class DecisionTreeEndpoint(Resource):
    def post(self):
        data = request.get_json()  # get data from post request
        input_data = data['data_points']

        # Define features and target
        features = ['x', 'y']
        target = 'class'

        # Convert data_points into a DataFrame
        df_input = [list(dp) + [i] for i, dp in enumerate(input_data)]
        df = pd.DataFrame(df_input, columns=features + [target])

        # Splitting the data into features and target
        X = df[features]
        y = df[target]

        # Perform decision tree classification
        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        # Create decision tree plot
        plt.figure(figsize=(10, 6))
        plot_tree(clf, feature_names=features, class_names=[str(cls) for cls in clf.classes_], filled=True)
        plt.title('Decision Tree')

        # Encode decision tree image to base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=300)  # Increase dpi for higher resolution
        plt.close()  # Close the figure to free up memory
        img_bytes.seek(0)
        img_base64 = "data:image/png;base64," + base64.b64encode(img_bytes.read()).decode('utf-8')

        # Create step-by-step calculations
        steps = []
        for node in range(clf.tree_.node_count):
            if clf.tree_.children_left[node] != clf.tree_.children_right[node]:  # It's not a leaf
                feature = clf.tree_.feature[node]
                threshold = clf.tree_.threshold[node]
                step = {
                    'feature': features[feature],
                    'threshold': threshold,
                    'left_child': int(clf.tree_.children_left[node]),   # Conversion to int
                    'right_child': int(clf.tree_.children_right[node])  # Conversion to int
                }
                steps.append(step)

        # Create conclusion
        conclusion = {
            'num_classes': int(len(clf.classes_)),   # Conversion to int
            'class_labels': [str(cls) for cls in clf.classes_]
        }

        result = {
            'steps': steps,
            'conclusion': conclusion,
            'decision_tree_image': img_base64
        }

        return jsonify(result)
