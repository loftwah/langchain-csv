import pandas as pd
import numpy as np

def calculate_fantasy_points(stats_df, scoring_system='standard'):
    """Calculate fantasy points based on different scoring systems"""
    # Create a copy to avoid modifying the original
    df = stats_df.copy()
    
    # Ensure required columns exist
    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']:
        if col not in df.columns:
            df[col] = 0
    
    if scoring_system == 'standard':
        # Standard scoring: PTS=1, REB=1.2, AST=1.5, STL=3, BLK=3, TOV=-1, 3PM=0.5
        df['FANTASY_POINTS'] = (
            df['PTS'] * 1.0 +
            df['REB'] * 1.2 +
            df['AST'] * 1.5 +
            df['STL'] * 3.0 +
            df['BLK'] * 3.0 +
            df['TOV'] * -1.0 +
            df['FG3M'] * 0.5
        )
    elif scoring_system == 'points':
        # Points league: PTS=1, REB=1, AST=1, STL=1, BLK=1, TOV=-1
        df['FANTASY_POINTS'] = (
            df['PTS'] * 1.0 +
            df['REB'] * 1.0 +
            df['AST'] * 1.0 +
            df['STL'] * 1.0 +
            df['BLK'] * 1.0 +
            df['TOV'] * -1.0
        )
    elif scoring_system == 'categories':
        # For category leagues, we calculate z-scores for each category
        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3M']
        negative_cats = ['TOV']
        
        # Calculate z-score for each category
        for cat in categories:
            if cat in df.columns:
                mean = df[cat].mean()
                std = df[cat].std()
                if std > 0:  # Avoid division by zero
                    df[f'{cat}_ZSCORE'] = (df[cat] - mean) / std
                else:
                    df[f'{cat}_ZSCORE'] = 0
        
        # For negative categories, multiply by -1 so that lower is better
        for cat in negative_cats:
            if cat in df.columns:
                mean = df[cat].mean()
                std = df[cat].std()
                if std > 0:
                    df[f'{cat}_ZSCORE'] = -1 * (df[cat] - mean) / std
                else:
                    df[f'{cat}_ZSCORE'] = 0
        
        # Sum all z-scores to get total value
        z_cols = [f'{cat}_ZSCORE' for cat in categories + negative_cats if f'{cat}_ZSCORE' in df.columns]
        df['FANTASY_POINTS'] = df[z_cols].sum(axis=1)
    
    return df

def calculate_consistency(games_df, scoring_system='standard'):
    """Calculate player consistency based on fantasy points variation"""
    if games_df.empty:
        return 0, 0, 0, 0, []
    
    # Calculate fantasy points for each game
    fantasy_games = calculate_fantasy_points(games_df, scoring_system)
    
    # Calculate stats
    mean_fp = fantasy_games['FANTASY_POINTS'].mean()
    std_fp = fantasy_games['FANTASY_POINTS'].std()
    min_fp = fantasy_games['FANTASY_POINTS'].min()
    max_fp = fantasy_games['FANTASY_POINTS'].max()
    
    # Calculate coefficient of variation (lower means more consistent)
    cv = std_fp / mean_fp if mean_fp > 0 else float('inf')
    
    # Game-by-game fantasy points
    fp_trend = fantasy_games['FANTASY_POINTS'].tolist()
    
    return mean_fp, cv, min_fp, max_fp, fp_trend 