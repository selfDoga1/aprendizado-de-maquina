import os
import joblib
import numpy as np
import pandas as pd

dados_ficticios = pd.DataFrame({
    'school': ['GP', 'MS', 'GP'],
    'sex': ['F', 'M', 'F'],
    'age': [17, 18, 16],
    'address': ['U', 'R', 'U'],
    'famsize': ['GT3', 'LE3', 'GT3'],
    'Pstatus': ['T', 'A', 'T'],
    'Medu': [3, 2, 4],
    'Fedu': [2, 3, 1],
    'Mjob': ['health', 'teacher', 'at_home'],
    'Fjob': ['services', 'other', 'teacher'],
    'reason': ['course', 'home', 'reputation'],
    'guardian': ['mother', 'father', 'mother'],
    'traveltime': [1, 2, 3],
    'studytime': [2, 3, 1],
    'failures': [0, 1, 2],
    'schoolsup': ['no', 'yes', 'no'],
    'famsup': ['yes', 'no', 'yes'],
    'paid': ['no', 'yes', 'no'],
    'activities': ['yes', 'no', 'yes'],
    'nursery': ['yes', 'yes', 'no'],
    'higher': ['yes', 'yes', 'no'],
    'internet': ['yes', 'no', 'yes'],
    'romantic': ['no', 'yes', 'no'],
    'famrel': [4, 3, 5],
    'freetime': [3, 2, 4],
    'goout': [3, 4, 2],
    'Dalc': [1, 2, 1],
    'Walc': [2, 3, 1],
    'health': [5, 3, 4],
    'absences': [4, 10, 2],
    'G1': [12, 10, 15],
    'G2': [14, 11, 16],
})

models_folder = "models"
results_df = []

for file in os.listdir(models_folder):

    if file.endswith(".joblib"):
        file_source = os.path.join(models_folder, file)
        model = joblib.load(file_source)
        model_name = file.replace("_best_model.joblib", "")

        pred = model.predict(dados_ficticios)

        if pred.ndim > 1:
            pred = pred[:, 0]

        df_pred = pd.DataFrame({
            "modelo": [model_name] * len(pred),
            "aluno": [f"Aluno {i + 1}" for i in range(len(pred))],
            "G3_pred": np.round(pred)
        })

        results_df.append(df_pred)

df_final = pd.concat(results_df, ignore_index=True)
df_final.to_excel("results/predicoes_modelos.xlsx", index=False)

print("predicoes finalizadas com sucesso!")

