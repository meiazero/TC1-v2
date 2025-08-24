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
        # Flatten cross-validation summary metrics if available
        for cv_key, cv_val in res.get('cv', {}).items():
            # include CV mean and std metrics
            row[cv_key] = cv_val
        rows.append(row)

    return pd.DataFrame(rows)


def select_best_model(df, metrics=None, weights=None, top_k=1):
    """
    Seleciona o(s) melhor(es) modelo(s) baseado em múltiplas métricas de teste.

    Gera uma pontuação composta para cada experimento utilizando métricas normalizadas
    e pesos definidos. Modelos com maior pontuação são considerados mais robustos.

    Args:
        df: DataFrame contendo colunas de métricas (_test) para cada experimento.
        metrics: Lista de nomes de colunas de métrica a considerar. Se None, usa métricas padrão.
        weights: Dicionário {métrica: peso} definindo a importância de cada métrica.
            Se None, usa pesos padrão.
        top_k: Número de melhores experimentos a retornar.

    Returns:
        dict do experimento com maior pontuação se top_k==1, ou DataFrame dos top_k melhores.
    """
    if df is None or df.empty:
        return None

    # Métricas e pesos padrão
    default_metrics = [
        "r2_test", "mae_test", "mape_test", "pearson_test", "spearman_test"
    ]
    default_weights = {
        "r2_test": 0.4,
        "mae_test": 0.2,
        "mape_test": 0.1,
        "pearson_test": 0.2,
        "spearman_test": 0.1,
    }

    metrics = metrics or default_metrics
    weights = weights or default_weights

    # Pré-calcular normalização das métricas
    norm_scores = {}
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"Métrica '{m}' não encontrada no DataFrame.")
        col = df[m]
        min_val = col.min()
        max_val = col.max()
        if max_val - min_val == 0:
            # Todos iguais; atribui 1.0
            norm = pd.Series(1.0, index=df.index)
        else:
            # Métricas de erro (lower is better)
            if m in ["mse_test", "rmse_test", "mae_test", "medae_test", "mape_test"]:
                norm = (max_val - col) / (max_val - min_val)
            else:
                # higher is better
                norm = (col - min_val) / (max_val - min_val)
        norm_scores[m] = norm

    # Construir pontuação composta
    comp_score = pd.Series(0.0, index=df.index)
    for m, w in weights.items():
        if m not in norm_scores:
            continue
        comp_score += norm_scores[m] * w

    # Ordenar por pontuação composta
    df_scored = df.copy()
    df_scored["composite_score"] = comp_score
    sorted_df = df_scored.sort_values(by="composite_score", ascending=False).reset_index(drop=True)

    if top_k == 1:
        return sorted_df.iloc[0].to_dict()
    return sorted_df.head(top_k)
