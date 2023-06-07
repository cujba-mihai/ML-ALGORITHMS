from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

app = Flask(__name__)
api = Api(app)

class Apriori(Resource):
    def post(self):
        data = request.get_json()  # get data from post request
        transactions = data['transactions']
        min_support = data.get('min_support', 0.01)  # must be > 0
        min_confidence = data.get('min_confidence', 0)
        min_lift = data.get('min_lift', 0)
        min_length = data.get('min_length', 1)  # minimum length of itemsets

        # Encoding the transactions
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)

        # Applying apriori to generate all frequent itemsets that meet the minimum support threshold
        frequent_itemsets_df = apriori(df, min_support=min_support, use_colnames=True, verbose=1)

        # Convert frozenset objects to lists for JSON serialization
        frequent_itemsets_df['itemsets'] = frequent_itemsets_df['itemsets'].apply(list)

        # Generate association rules
        rules = association_rules(frequent_itemsets_df, metric="confidence", min_threshold=min_confidence)

        # Filtering the rules by the lift and min_length
        rules = rules[(rules['lift'] >= min_lift) & (rules['antecedents'].apply(lambda x: len(x)) >= min_length)]

        # build steps dictionary
        steps = {}
        total_transactions = len(transactions)

        for index, row in frequent_itemsets_df.iterrows():
            itemset = list(row['itemsets'])
            support = row['support']

            step_number = len(itemset)
            step_key = f"step{step_number}"

            if step_key not in steps:
                steps[step_key] = f"Step {step_number}: Calculate support for {step_number}-tuple items\n"

            steps[step_key] += f'Support for {itemset}: {support*total_transactions} (occurrences) / {total_transactions} (total transactions) = {support}\n'

        # Convert frozenset objects to lists for JSON serialization
        rules['antecedents'] = rules['antecedents'].apply(list)
        rules['consequents'] = rules['consequents'].apply(list)

        result = {
            'steps': steps,
#             'frequent_itemsets': frequent_itemsets_df.to_dict('records'),
            'association_rules': rules.to_dict('records'),
        }
        return jsonify(result)

api.add_resource(Apriori, '/apriori')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
