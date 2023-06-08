from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

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
                steps[step_key] = [f"Pasul {step_number}: Calculează asistenta pentru seturi de {step_number} elemente"]

            # For each combination of size greater than 1, display all possible rules
            if step_number > 1:
                for i in range(step_number):
                    antecedent = itemset[:i] + itemset[i+1:]
                    consequent = itemset[i]
                    steps[step_key].append(f'Asistenta pentru {antecedent} -> {consequent}: {support*total_transactions} (apariții) / {total_transactions} (tranzacții totale) = {support}')
            else:
                steps[step_key].append(f'Asistenta pentru {itemset}: {support*total_transactions} (apariții) / {total_transactions} (tranzacții totale) = {support}')



        # Convert frozenset objects to lists for JSON serialization
        rules['antecedents'] = rules['antecedents'].apply(list)
        rules['consequents'] = rules['consequents'].apply(list)

        # Generate conclusion with combinations that have strong correlations
        conclusion = []
        for _, row in rules.iterrows():
            antecedents = row['antecedents']
            consequents = row['consequents']
            lift = row['lift']
            support = row['support']
            antecedent_support = row['antecedent support']
            consequent_support = row['consequent support']
            lift_calculation = f'Increderea({antecedents} -> {consequents}) = Suport({antecedents} și {consequents}) / (Asistenta({antecedents}) * Asistenta({consequents})) = {support} / ({antecedent_support} * {consequent_support}) = {lift}'
            conclusion.append({
#                 'combination': antecedents + consequents,
#                 'lift': lift,
                'calcul_incredere': lift_calculation
            })


        conclusion = [dict(t) for t in set(tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())) for d in conclusion)]

        result = {
            'concluzie': conclusion,
            'etapele': steps,
        }

        return jsonify(result)
