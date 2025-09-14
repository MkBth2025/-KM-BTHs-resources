import pandas as pd
import numpy as np

def select_optimal_prompts(input_file='data/prompt_score.xlsx',

                          output_file='report/selected_prompts.csv',
                          num_samples=9):
    """
    Selects optimal representative sample of 9 prompts based on complexity scores
    using stratified random sampling with optimized distribution.
    """
    
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Verify required columns exist
    if not all(col in df.columns for col in ['Id', 'Score']):
        raise ValueError("Excel file must contain 'Id' and 'Score' columns")
    
    # Sort by score for stratification
    df_sorted = df.sort_values('Score').reset_index(drop=True)
    total_prompts = len(df_sorted)
    
    # Create optimal strata distribution for 9 samples
    # Focus more on medium and high complexity prompts for better differentiation
    low_stratum = df_sorted.iloc[:int(0.25 * total_prompts)]          # Bottom 25%
    medium_stratum = df_sorted.iloc[int(0.25 * total_prompts):int(0.75 * total_prompts)]  # Middle 50%
    high_stratum = df_sorted.iloc[int(0.75 * total_prompts):]         # Top 25%
    
    # Optimal distribution for 9 samples: 2 Low, 4 Medium, 3 High
    # This emphasizes medium and high complexity where models differ most
    n_low = min(2, len(low_stratum))
    n_medium = min(4, len(medium_stratum))
    n_high = min(3, len(high_stratum))
    
    # Adjust if any stratum doesn't have enough samples
    if n_low + n_medium + n_high < num_samples:
        remaining = num_samples - (n_low + n_medium + n_high)
        # Prioritize medium stratum for additional samples
        n_medium += min(remaining, len(medium_stratum) - n_medium)
        remaining = num_samples - (n_low + n_medium + n_high)
        if remaining > 0:
            n_high += min(remaining, len(high_stratum) - n_high)
    
    # Random sampling from each stratum with ensured diversity
    low_samples = low_stratum.sample(n=n_low, random_state=42, replace=False) if n_low > 0 else pd.DataFrame()
    medium_samples = medium_stratum.sample(n=n_medium, random_state=43, replace=False) if n_medium > 0 else pd.DataFrame()
    high_samples = high_stratum.sample(n=n_high, random_state=44, replace=False) if n_high > 0 else pd.DataFrame()
    
    # Combine selected samples
    selected_samples = pd.concat([low_samples, medium_samples, high_samples])
    
    # Add stratum information and ensure we have exactly 9 samples
    selected_samples = selected_samples.head(num_samples)
    selected_samples['Stratum'] = ''
    
    # Assign strata based on original positions
    for idx, row in selected_samples.iterrows():
        if idx in low_stratum.index:
            selected_samples.loc[idx, 'Stratum'] = 'Low'
        elif idx in medium_stratum.index:
            selected_samples.loc[idx, 'Stratum'] = 'Medium'
        else:
            selected_samples.loc[idx, 'Stratum'] = 'High'
    
    # Calculate score statistics for quality assurance
    score_stats = {
        'min_score': selected_samples['Score'].min(),
        'max_score': selected_samples['Score'].max(),
        'mean_score': selected_samples['Score'].mean(),
        'score_range': selected_samples['Score'].max() - selected_samples['Score'].min()
    }
    
    # Save to CSV
    selected_samples[['Id', 'Score', 'Stratum']].to_csv(output_file, index=False)
    
    # Print comprehensive summary
    print(f"🎯 Optimal selection of {len(selected_samples)} prompts completed!")
    print(f"📊 Original dataset: {total_prompts} prompts")
    print(f"📈 Score range in selection: {score_stats['min_score']:.3f} - {score_stats['max_score']:.3f}")
    print(f"📐 Mean score: {score_stats['mean_score']:.3f}")
    print(f"🔢 Stratum distribution:")
    print(f"   • Low complexity: {len(selected_samples[selected_samples['Stratum'] == 'Low'])}")
    print(f"   • Medium complexity: {len(selected_samples[selected_samples['Stratum'] == 'Medium'])}")
    print(f"   • High complexity: {len(selected_samples[selected_samples['Stratum'] == 'High'])}")
    print(f"💾 Results saved to: {output_file}")
    
    return selected_samples

def validate_selection(original_df, selected_df):
    """
    Validates that the selected samples represent the original distribution.
    """
    original_mean = original_df['Score'].mean()
    selected_mean = selected_df['Score'].mean()
    
    original_std = original_df['Score'].std()
    selected_std = selected_df['Score'].std()
    
    print(f"\n✅ Validation Results:")
    print(f"   Original mean: {original_mean:.3f}")
    print(f"   Selected mean: {selected_mean:.3f}")
    print(f"   Difference: {abs(original_mean - selected_mean):.3f}")
    print(f"   Original std: {original_std:.3f}")
    print(f"   Selected std: {selected_std:.3f}")
    
    return abs(original_mean - selected_mean) < 0.1 * original_std  # Within 10% of std

# Run the function
if __name__ == "__main__":
    # Read original data for validation
    original_df = pd.read_excel('data/prompt_score.xlsx')
    
    # Select optimal prompts
    selected_prompts = select_optimal_prompts(
        input_file='data/prompt_score.xlsx',
        output_file='report/optimal_selected_prompts.csv',
        num_samples=9
    )
    
    # Validate the selection
    is_representative = validate_selection(original_df, selected_prompts)
    
    if is_representative:
        print("🎉 Selection is statistically representative!")
    else:
        print("⚠️  Selection may not fully represent the original distribution.")
    
    # Display selected prompts
    print("\n📋 Selected Prompts (ID, Score, Stratum):")
    print(selected_prompts[['Id', 'Score', 'Stratum']].to_string(index=False))