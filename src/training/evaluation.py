import pandas as pd

def results_to_dataframe(results_list):
    """
    Converte lista de resultados do trainer em DataFrame.
    """
    rows = []
    for res in results_list:
        if "error" in res:
            rows.append({
                "model": res["model"],
                "params": res["params"],
                "error": res["error"]
            })
            continue

        row = {
            "model": res["model"],
            "params": res["params"],
        }

        for split in ["train", "test"]:
            metrics = res[split]
            for k, v in metrics.items():
                if k in ["y_true", "y_pred", "residuals"]:
                    continue  # não salvar vetores no DF
                row[f"{k}_{split}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def select_best_model(df, metric="r2_test", top_k=1):
    """
    Retorna o(s) melhor(es) modelo(s) com base em uma métrica.
    """
    if df.empty:
        return None

    sorted_df = df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    if top_k == 1:
        return sorted_df.iloc[0].to_dict()
    else:
        return sorted_df.head(top_k)
