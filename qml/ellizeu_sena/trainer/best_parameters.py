import csv
import json
import os
import pandas as pd


def get_best_result_from_csv(
    csv_path,
    metric="accuracy_mean",
):
    best_row = None
    best_score = float(
        "-inf"
    )

    with open(
        csv_path,
        "r",
        newline="",
    ) as csvfile:

        reader = csv.DictReader(
            csvfile
        )

        for row in reader:

            score = float(
                row[metric]
            )

            if score > best_score:

                best_score = score
                best_row = row

    return best_row

def build_best_parameters_json(
    model_name,
    save_dir="data/ellizeu_sena/grid_search",
    metric="accuracy_mean",
):
    """
    Generate best parameter JSON from CSV results, 
    filtering out models with fake perfect recall.
    """

    csv_path = os.path.join(save_dir, f"{model_name}_results.csv")
    json_path = os.path.join(save_dir, f"{model_name}_best_parameters.json")

    # --- Nova Lógica de Filtragem ---
    # Carrega o CSV para aplicar a regra de exclusão de "chutadores"
    df = pd.read_csv(csv_path)

    # Filtro: Remove linhas onde o recall_mean é exatamente 1.0 (ou 100) 
    # MAS a precisão é correspondentemente baixa/suspeita.
    # Se você quiser banir TOTALMENTE recall de 100% (1.0), use apenas: df['recall_mean'] < 1.0
    validador_recall = 1.0 if df['recall_mean'].max() <= 1.0 else 100.0
    
    # Filtra fora os modelos com recall máximo absoluto (geralmente gerados por dummies/overfitting)
    df_filtrado = df[df['recall_mean'] < validador_recall]

    # Se o filtro esvaziar o DataFrame (caso raro), mantemos o original para não quebrar
    if df_filtrado.empty:
        df_filtrado = df

    # Encontra a melhor linha com base na métrica escolhida dentro dos modelos válidos
    best_row_index = df_filtrado[metric].idxmax()
    best_row = df_filtrado.loc[best_row_index].to_dict()
    # --------------------------------

    metric_fields = {
        "accuracy_mean", "accuracy_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "execution_time_seconds",
    }

    best_params = {
        k: v for k, v in best_row.items() 
        if k not in metric_fields
    }

    best_results = {
        k: float(v) for k, v in best_row.items() 
        if k in metric_fields
    }

    summary = {
        "best_params": best_params,
        "best_score": float(best_row[metric]),
        "results": best_results,
    }

    with open(json_path, "w") as jsonfile:
        json.dump(summary, jsonfile, indent=4)

    return summary