import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize







def get_clustering_ratio(signal, df_regimes, alpha):
    
    intra_dists = []
    centroids = []
    signal = signal.ewm(alpha = alpha, adjust = False).mean()
    for _, row in df_regimes.iterrows():
        mask = (signal.index >= row['start_date']) & (signal.index <= row['end_date'])
        regime_data = signal.loc[mask]
        if len(regime_data) > 1:
            d = abs(regime_data.values[:, np.newaxis] - regime_data.values)
            intra_dists.append(np.mean(d[np.triu_indices(d.shape[0], k=1)]))
        centroids.append(regime_data.mean())
    
    centroids = np.array(centroids)
    d = abs(centroids[:, np.newaxis] - centroids)
    mean_inter = np.mean(d)
    ratio = np.mean(intra_dists) / (mean_inter + 1e-9)

    #print(alpha, ratio)

    return ratio


if __name__ == "__main__":
    

    df_macro = pd.read_parquet('s3://jdinvestment/simulation_data/macro_data.parquet')
    df_regimes = pd.read_parquet('strategy/semantic_regimes.parquet')
    
    
    results = []
    df_signal = pd.DataFrame()
    for col in df_macro.columns:

        
        
        # 2. GENERATE THE CURVE
        alphas = np.linspace(0.005, 0.2, 50)
        ratios = [get_clustering_ratio(df_macro[col], df_regimes, a) for a in alphas]
        ratios = np.array(ratios)

        # 3. FIND THE KNEE (Kneedle algorithm - Distance from Line approach)
        # Normalize points to [0, 1] for geometric distance comparison
        n_alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
        n_ratios = (ratios - ratios.min()) / (ratios.max() - ratios.min())

        # Calculate distance from the line connecting first and last points
        # The line is y = x (in normalized space)
        distances = np.abs(n_ratios - n_alphas) 
        knee_idx = np.argmax(distances)
        optimal_alpha = alphas[knee_idx]
        knee_ratio = ratios[knee_idx]
        


        results += [[col, optimal_alpha, knee_ratio]]
        df_signal[col] = df_macro[col].ewm(alpha = optimal_alpha, adjust = False).mean()


    df = pd.DataFrame(results, columns = ['signal', 'alpha', 'ratio'])
    df = df.sort_values(by = 'ratio')

    df_signal = df_macro.loc[:, [col for col in df.signal.values if col.endswith('_0_1')]]
    df_signal.to_parquet('s3://jdinvestment/simulation_data/macro_signals.parquet')



