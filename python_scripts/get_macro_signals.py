import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def get_clustering_ratio_expanding(signal_slice, df_regimes_slice, alpha):
    intra_dists = []
    centroids = []
    # Apply EWM strictly on the historical slice up to time t
    smoothed = signal_slice.ewm(alpha=alpha, adjust=False).mean()
    
    for _, row in df_regimes_slice.iterrows():
        # Mask only regimes that have fully completed or overlap within the available past data
        mask = (smoothed.index >= row['start_date']) & (smoothed.index <= row['end_date'])
        regime_data = smoothed.loc[mask]
        if len(regime_data) > 1:
            d = abs(regime_data.values[:, np.newaxis] - regime_data.values)
            intra_dists.append(np.mean(d[np.triu_indices(d.shape[0], k=1)]))
        centroids.append(regime_data.mean() if len(regime_data) > 0 else np.nan)
    
    centroids = np.array([c for c in centroids if not np.isnan(c)])
    if len(centroids) < 2:
        return np.inf
    d = abs(centroids[:, np.newaxis] - centroids)
    mean_inter = np.mean(d)
    ratio = np.mean(intra_dists) / (mean_inter + 1e-9)
    return ratio

def get_macro_signals():
    print("Loading datasets...")
    df_macro = pd.read_parquet('s3://jdinvestment/simulation_data/macro_data.parquet')
    df_regimes = pd.read_parquet('strategy/semantic_regimes.parquet')
    
    alphas = np.linspace(0.005, 0.2, 30) # Reduced search space slightly for performance
    target_cols = [col for col in df_macro.columns if col.endswith('_0_1')]
    
    df_signal = pd.DataFrame(index=df_macro.index)
    
    REOPT_STEP = 13
    total_rows = len(df_macro)
    
    for col_idx, col in enumerate(target_cols, 1):
        print(f"\n--- Processing column {col_idx}/{len(target_cols)}: {col} ---")
        dynamic_smoothed_values = []
        current_optimal_alpha = 0.05 # Default fallback start
        
        for i in range(total_rows):
            history_slice = df_macro[col].iloc[:i+1]
            
            # Print progress update every 100 rows or on the final row
            if i % 100 == 0 or i == total_rows - 1:
                print(f"  [Progress] Row {i+1}/{total_rows} ({(i+1)/total_rows*100:.1f}%) | Current Alpha: {current_optimal_alpha:.4f}")
            
            # Only run the heavy optimization search at specified step intervals to save compute time
            if i >= 60 and (i % REOPT_STEP == 0 or i == total_rows - 1):
                current_date = df_macro.index[i]
                past_regimes = df_regimes[df_regimes['end_date'] <= current_date]
                
                if len(past_regimes) > 1:
                    ratios = []
                    for a in alphas:
                        r = get_clustering_ratio_expanding(history_slice, past_regimes, a)
                        ratios.append(r)
                    
                    ratios = np.array(ratios)
                    if not np.all(np.isinf(ratios)):
                        n_alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
                        n_ratios = (ratios - ratios.min()) / (ratios.max() - ratios.min() + 1e-9)
                        distances = np.abs(n_ratios - n_alphas)
                        optimal_idx = np.nanargmax(distances)
                        current_optimal_alpha = alphas[optimal_idx]
            
            # Calculate the recursive EWM value for the current timestamp using the point-in-time alpha
            point_in_time_ewm = history_slice.ewm(alpha=current_optimal_alpha, adjust=False).mean()
            dynamic_smoothed_values.append(point_in_time_ewm.iloc[-1])
            
        df_signal[col] = dynamic_smoothed_values

    print("\nSaving final point-in-time signals to parquet...")
    df_signal.to_parquet('s3://jdinvestment/simulation_data/macro_signals.parquet')
    print("Done!")