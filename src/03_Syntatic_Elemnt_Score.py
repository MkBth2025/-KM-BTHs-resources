import pandas as pd
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy import stats

# Path to the Excel file
#-------------------------
# Initialize tkinter (needed for filedialog even without a visible window)
root = tk.Tk()
root.withdraw()


file_path = r"data/test_dataset1.xlsx"
print(f"Selected file: {file_path}")

# Function to calculate non-empty lines
def calculate_non_empty_lines(code):
    """Calculate the number of non-empty lines in PlantUML code"""
    if pd.isna(code) or code is None:
        return 0
        
    code = str(code)
    lines = code.split('\n')
    non_empty_count = 0
    
    for line in lines:
        stripped_line = line.strip()
        # Count lines that have actual content
        if stripped_line and not stripped_line.startswith("'") and not stripped_line.startswith("@"):
            non_empty_count += 1
    
    return non_empty_count

# Corrected activity counting function
def count_activities(code):
    """Count PlantUML activities with precise parsing"""
    if pd.isna(code) or code is None:
        return 0
        
    code = str(code)
    lines = code.split('\n')
    activity_count = 0
    
    for line in lines:
        stripped_line = line.strip()
        
        # Very precise pattern: line must start with : and end with ; and be standalone
        if stripped_line.startswith(':') and stripped_line.endswith(';'):
            # Make sure it's not part of other syntax
            if not any(keyword in stripped_line.lower() for keyword in 
                      ['then', 'else', 'case', 'while', 'until', 'repeat']):
                # Additional check: make sure it has actual activity content
                content = stripped_line[1:-1].strip()  # Remove : and ;
                if content and len(content) > 0:
                    activity_count += 1
    
    return activity_count

# **ENHANCED**: Improved decision counting function to handle all if statement formats
def count_decisions(code):
    """Count PlantUML decisions with improved parsing to handle all if statement formats"""
    if pd.isna(code) or code is None:
        return 0
        
    code = str(code)
    lines = code.split('\n')
    decision_count = 0
    
    for line in lines:
        stripped_line = line.strip()
        
        # **IMPROVED**: Handle both formats - 'if (condition) then' and 'if "" then' and 'if condition then'
        if re.match(r'^if\s+.*\s+then\b', stripped_line, re.IGNORECASE):
            decision_count += 1
        # Count 'switch' statements (decisions)
        elif re.match(r'^switch\s*\(.*\)', stripped_line, re.IGNORECASE):
            decision_count += 1
    
    return decision_count

# Corrected fork counting function
def count_forks(code):
    """Count PlantUML forks with precise parsing"""
    if pd.isna(code) or code is None:
        return 0
        
    code = str(code)
    lines = code.split('\n')
    fork_count = 0
    
    for line in lines:
        stripped_line = line.strip()
        
        # Count standalone 'fork' or 'split' statements
        if re.match(r'^\s*fork\s*$', stripped_line, re.IGNORECASE):
            fork_count += 1
        elif re.match(r'^\s*split\s*$', stripped_line, re.IGNORECASE):
            fork_count += 1
    
    return fork_count

# Corrected swimlane counting function
def count_swimlanes(code):
    """Count PlantUML swimlanes with precise parsing"""
    if pd.isna(code) or code is None:
        return 0
        
    code = str(code)
    lines = code.split('\n')
    swimlane_count = 0
    
    for line in lines:
        stripped_line = line.strip()
        
        # Count swimlane declarations |Name|
        if re.match(r'^\|[^|]+\|$', stripped_line):
            swimlane_count += 1
    
    return swimlane_count

# Enhanced flow analysis function
def analyze_plantuml_flow(code):
    """Analyze PlantUML flow structure to understand connections"""
    if pd.isna(code) or code is None:
        return []
        
    code = str(code)
    lines = code.split('\n')
    flow_elements = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            element_type = None
            if re.match(r'^:[^:]*;$', stripped):
                element_type = 'activity'
            elif re.match(r'^\s*if\s*\(.*\)', stripped, re.IGNORECASE):
                element_type = 'decision'
            elif re.match(r'^\s*start\s*$', stripped, re.IGNORECASE):
                element_type = 'start'
            elif re.match(r'^\s*(stop|end)\s*$', stripped, re.IGNORECASE):
                element_type = 'stop'
            elif re.match(r'^\s*(then|else)', stripped, re.IGNORECASE):
                element_type = 'branch'
            elif re.match(r'^\s*endif\s*$', stripped, re.IGNORECASE):
                element_type = 'endif'
            elif re.match(r'^\s*fork\s*$', stripped, re.IGNORECASE):
                element_type = 'fork'
            elif re.match(r'^\s*(join|end\s*fork)\s*$', stripped, re.IGNORECASE):
                element_type = 'join'
            
            if element_type:
                flow_elements.append({
                    'line': stripped,
                    'type': element_type,
                    'position': i,
                    'line_number': i + 1
                })
    
    return flow_elements

# **MODIFIED**: Enhanced syntax balance scoring with if-endif imbalances treated as orphaned_activities
def calculate_syntax_balance_score(code):
    """Calculate negative scores with if-endif imbalances treated as orphaned activities (no penalty)"""
    if pd.isna(code) or code is None:
        return 0, []
        
    code = str(code)
    negative_scores = []
    total_negative_score = 0
    
    # **MODIFIED**: Check if-endif balance but treat imbalances as orphaned_activities with 0 penalty
    if_count = count_decisions(code)
    endif_count = len([line for line in code.split('\n') 
                     if re.match(r'^\s*endif\s*$', line.strip(), re.IGNORECASE)])
    
    if_imbalance = abs(if_count - endif_count)
    if if_imbalance > 0:
        score = 0  # **NO PENALTY - Set to 0**
        negative_scores.append({
            'element': 'orphaned_activities',
            'count': if_imbalance,
            'score': score,
            'explanation': f'Info: Potential if-endif imbalance of {if_imbalance} treated as orphaned activities (no penalty)'
        })
    
    # Check fork-join balance
    fork_count = count_forks(code)
    join_count = len([line for line in code.split('\n') 
                     if re.match(r'^\s*(join|end\s*fork)\s*$', line.strip(), re.IGNORECASE)])
    
    fork_imbalance = abs(fork_count - join_count)
    if fork_imbalance > 0:
        score = -3 * fork_imbalance
        total_negative_score += score
        negative_scores.append({
            'element': 'fork-join_imbalance',
            'count': fork_imbalance,
            'score': score,
            'explanation': f'Unbalanced fork-join statements: {fork_count} fork vs {join_count} join'
        })
    
    # Check switch-endswitch balance
    switch_count = len([line for line in code.split('\n') 
                       if re.match(r'^\s*switch\s*\(.*\).*$', line.strip(), re.IGNORECASE)])
    endswitch_count = len([line for line in code.split('\n') 
                          if re.match(r'^\s*endswitch\s*$', line.strip(), re.IGNORECASE)])
    
    switch_imbalance = abs(switch_count - endswitch_count)
    if switch_imbalance > 0:
        score = -3 * switch_imbalance
        total_negative_score += score
        negative_scores.append({
            'element': 'switch-endswitch_imbalance',
            'count': switch_imbalance,
            'score': score,
            'explanation': f'Unbalanced switch-endswitch statements: {switch_count} switch vs {endswitch_count} endswitch'
        })
    
    # Critical penalty for missing start/stop (-10 each)
    has_start = any(re.match(r'^\s*start\s*$', line.strip(), re.IGNORECASE) 
                   for line in code.split('\n'))
    has_stop = any(re.match(r'^\s*(stop|end)\s*$', line.strip(), re.IGNORECASE) 
                  for line in code.split('\n'))
    
    if not has_start:
        score = -10  # CRITICAL PENALTY -10
        total_negative_score += score
        negative_scores.append({
            'element': 'missing_start',
            'count': 1,
            'score': score,
            'explanation': 'CRITICAL: Missing start statement - essential for proper activity diagram flow'
        })
    
    if not has_stop:
        score = -10  # CRITICAL PENALTY -10
        total_negative_score += score
        negative_scores.append({
            'element': 'missing_stop',
            'count': 1,
            'score': score,
            'explanation': 'CRITICAL: Missing stop/end statement - essential for proper activity diagram flow'
        })
    
    # Smart line penalty with dynamic thresholds
    non_empty_lines = calculate_non_empty_lines(code)
    activities = count_activities(code)
    decisions = count_decisions(code)
    forks = count_forks(code)
    
    # Calculate functional complexity for smart thresholds
    functional_elements = activities + decisions + forks
    
    if functional_elements > 0 and activities > 0:
        # Dynamic threshold based on complexity
        if functional_elements <= 5:
            threshold = 4.5  # Simple diagrams: 4.5 lines per activity
        elif functional_elements <= 15:
            threshold = 3.5  # Medium diagrams: 3.5 lines per activity  
        else:
            threshold = 3.0  # Complex diagrams: 3.0 lines per activity
        
        lines_to_activities_ratio = non_empty_lines / activities
        
        if lines_to_activities_ratio > threshold:
            excess_factor = lines_to_activities_ratio - threshold
            score = -0.3 * excess_factor
            total_negative_score += score
            negative_scores.append({
                'element': 'pss',
                'count': round(excess_factor, 1),
                'score': round(score, 1),
                'explanation': f'Minor: Excessive lines ratio: {lines_to_activities_ratio:.1f} lines per activity (smart threshold: {threshold})'
            })
    
    # Traditional orphaned activities penalty (also set to 0)
    orphaned_activities = 0
    activity_count = count_activities(code)
    
    if activity_count > 1:
        flow_elements = analyze_plantuml_flow(code)
        activity_elements = [elem for elem in flow_elements if elem['type'] == 'activity']
        
        # Check each activity for connections (for reporting only)
        for i, activity_elem in enumerate(activity_elements):
            is_connected = False
            
            # Check for explicit arrows
            activity_pattern = re.escape(activity_elem['line'])
            if re.search(f'(->{activity_pattern}|{activity_pattern}->)', code):
                is_connected = True
            
            # Check for implicit flow connections
            elif i == 0:
                # First activity - check if connected to start
                start_elements = [elem for elem in flow_elements if elem['type'] == 'start']
                if start_elements and start_elements[0]['position'] < activity_elem['position']:
                    is_connected = True
            
            elif i == len(activity_elements) - 1:
                # Last activity - check if connected to stop
                stop_elements = [elem for elem in flow_elements if elem['type'] == 'stop']
                if stop_elements and stop_elements[0]['position'] > activity_elem['position']:
                    is_connected = True
            
            else:
                # Middle activities - check for sequential flow or decision branches
                preceding_elements = [elem for elem in flow_elements 
                                    if elem['position'] < activity_elem['position'] 
                                    and elem['type'] in ['activity', 'decision', 'branch', 'endif', 'start']]
                
                following_elements = [elem for elem in flow_elements 
                                    if elem['position'] > activity_elem['position'] 
                                    and elem['type'] in ['activity', 'decision', 'stop', 'endif']]
                
                if preceding_elements and following_elements:
                    closest_before = max(preceding_elements, key=lambda x: x['position'])
                    closest_after = min(following_elements, key=lambda x: x['position'])
                    
                    # If activity is within reasonable distance of other flow elements
                    if (activity_elem['position'] - closest_before['position'] <= 10 and 
                        closest_after['position'] - activity_elem['position'] <= 10):
                        is_connected = True
            
            if not is_connected:
                orphaned_activities += 1
    
    # **MODIFIED**: Add traditional orphaned activities info for reporting but NO PENALTY
    if orphaned_activities > 0:
        negative_scores.append({
            'element': 'orphaned_activities',
            'count': orphaned_activities,
            'score': 0,  # **NO PENALTY - Set to 0**
            'explanation': f'Info: Activities detected as potentially disconnected: {orphaned_activities} (no penalty applied)'
        })
    
    return total_negative_score, negative_scores

# Enhanced function to extract UML features
def evaluate_plantuml(code):
    """Extract UML features with updated scoring system"""
    if pd.isna(code) or code is None:
        return {
            'activities': 0,
            'decisions': 0,
            'forks': 0,
            'swimlanes': 0,
            'detaches': 0,
            'options': 0,
            'edges': 0,
            'notes': 0,
            'partitions': 0,
            'numberline': 0,
            'negative_score': 0,
            'negative_details': []
        }
    
    code = str(code)
    
    # Use precise counting functions
    activities_count = count_activities(code)
    decisions_count = count_decisions(code)
    forks_count = count_forks(code)
    swimlanes_count = count_swimlanes(code)
    
    # Count other elements with precise patterns
    detaches = len([line for line in code.split('\n') 
                   if re.match(r'^\s*detach\s*$', line.strip(), re.IGNORECASE)])
    
    # Count options (more precise)
    options = 0
    for line in code.split('\n'):
        if ':' in line and 'option' in line.lower() and line.strip().endswith(';'):
            options += 1
    
    # Count edges (arrows) - More comprehensive
    edges = len(re.findall(r'(->{1,2}|<-{1,2})', code))
    
    # Count notes
    notes = len([line for line in code.split('\n') 
                if re.match(r'^\s*note\s+', line.strip(), re.IGNORECASE)])
    
    # Count partitions
    partitions = len([line for line in code.split('\n') 
                     if re.match(r'^\s*partition\s+', line.strip(), re.IGNORECASE)])
    
    # Calculate non-empty lines
    non_empty_lines = calculate_non_empty_lines(code)
    
    # Calculate negative scores with if-endif imbalances treated as orphaned activities
    negative_score, negative_details = calculate_syntax_balance_score(code)
    
    return {
        'activities': activities_count,
        'decisions': decisions_count,
        'forks': forks_count,
        'swimlanes': swimlanes_count,
        'detaches': detaches,
        'options': options,
        'edges': edges,
        'notes': notes,
        'partitions': partitions,
        'numberline': non_empty_lines,
        'negative_score': negative_score,
        'negative_details': negative_details
    }

# Function to calculate average metrics across all PlantUML codes
def calculate_average_metrics(df, target_columns):
    """Calculate average metrics across all PlantUML codes for comparison"""
    all_metrics = {
        'activities': [],
        'decisions': [],
        'forks': [],
        'swimlanes': [],
        'detaches': [],
        'options': [],
        'edges': [],
        'notes': [],
        'partitions': [],
        'numberline': [],
        'negative_score': []
    }
    
    for index, row in df.iterrows():
        for col in target_columns:
            try:
                code = row[col]
                if pd.isna(code) or code is None:
                    continue
                    
                metrics = evaluate_plantuml(str(code))
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
            except:
                continue
    
    # Calculate averages
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
            avg_metrics[f'median_{key}'] = np.median(values)
        else:
            avg_metrics[f'avg_{key}'] = 0
            avg_metrics[f'std_{key}'] = 0
            avg_metrics[f'median_{key}'] = 0
    
    return avg_metrics

# Function to calculate statistical metrics for columns
def calculate_statistical_metrics_for_columns(df, output_file='report/01_Syntatic_Statistical_Analysis.csv'):
    """Calculate statistical metrics for each numerical column of the DataFrame."""
    results = []
    
    # Get only numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_columns:
        values = df[col].dropna().values
        
        if len(values) < 3:
            continue
            
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        # Shapiro-Wilk test for normality
        if len(values) >= 3 and len(values) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(values)
            except:
                shapiro_p = np.nan
        else:
            shapiro_p = np.nan
        
        # Z-statistic and P-value
        null_mean = 0
        n = len(values)
        if std_val > 0:
            std_err = std_val / np.sqrt(n)
            z_stat = (mean_val - null_mean) / std_err
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0
            std_err = 0
        
        # Effect size (Cohen's d)
        if std_val != 0:
            effect_size = (mean_val - null_mean) / std_val
        else:
            effect_size = 0
        
        # Statistical power calculation
        alpha = 0.05
        if std_val > 0 and n > 0:
            critical_z = stats.norm.ppf(1 - alpha / 2)
            power = 1 - stats.norm.cdf(critical_z - abs(z_stat)) + stats.norm.cdf(-critical_z - abs(z_stat))
        else:
            power = 0
        
        mannwhitney_p = np.nan
        
        # 95% Confidence Interval
        if std_val > 0 and n > 0:
            critical_z = stats.norm.ppf(1 - alpha / 2)
            ci_lower = mean_val - critical_z * std_err
            ci_upper = mean_val + critical_z * std_err
        else:
            ci_lower = mean_val
            ci_upper = mean_val
        
        results.append({
            'Column': col,
            'Mean': round(mean_val, 6),
            'Std Dev': round(std_val, 6),
            'Shapiro-Wilk p': round(shapiro_p, 6) if not np.isnan(shapiro_p) else np.nan,
            'Z-stat': round(z_stat, 6),
            'P-value': round(p_value, 6),
            'Effect Size': round(effect_size, 6),
            'Power (1-β)': round(power, 6),
            'Mann-Whitney U p': mannwhitney_p,
            '95% CI Lower': round(ci_lower, 6),
            '95% CI Upper': round(ci_upper, 6)
        })
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(output_file, index=False)
    
    print(f"Statistical metrics saved to {output_file}")
    return metrics_df

# Scoring weights with balanced priorities
weights = {
    'activities': 3,           # High value for core functionality
    'decisions': 4,            # Highest value for logical complexity
    'forks': 3,               # High value for parallel processing
    'swimlanes': 2,           # Good organizational value
    'detaches': -1.5,         # Moderate penalty
    'options': 1,             # Standard value
    'edges': 1.2,             # Slight increase for connectivity importance
    'notes': 0.8,             # Increased documentation value
    'partitions': 1.5,        # Good organizational value
    'numberline': 0.05,       # Minimal weight for basic structure
    'negative_score': 1       # Direct inclusion of negative scores
}

# Calculate percentage score
def calculate_percentage_score(raw_score, metrics):
    """Convert raw score to percentage based on potential maximum"""
    
    # Calculate theoretical maximum score (if diagram was perfect)
    activities = metrics.get('activities', 0)
    decisions = metrics.get('decisions', 0)
    forks = metrics.get('forks', 0)
    swimlanes = metrics.get('swimlanes', 0)
    edges = metrics.get('edges', 0)
    notes = metrics.get('notes', 0)
    partitions = metrics.get('partitions', 0)
    options = metrics.get('options', 0)
    numberline = metrics.get('numberline', 0)
    
    # Calculate maximum possible positive score
    max_positive_score = (
        (activities * weights['activities']) +
        (decisions * weights['decisions']) +
        (forks * weights['forks']) +
        (swimlanes * weights['swimlanes']) +
        (edges * weights['edges']) +
        (notes * weights['notes']) +
        (partitions * weights['partitions']) +
        (options * weights['options']) +
        (numberline * weights['numberline'])
    )
    
    # Add bonus for having start/stop (no penalty)
    max_positive_score += 20  # Bonus for complete diagram structure
    
    # Ensure minimum baseline for percentage calculation
    if max_positive_score < 10:
        max_positive_score = 10
    
    # Calculate percentage
    percentage = max(0, min(100, (raw_score / max_positive_score) * 100))
    
    return round(percentage, 1)

# Function to compute enhanced score with percentage
def compute_score(metrics):
    """Compute score with updated balanced weighting system and percentage"""
    raw_score = sum(metrics[key] * weights.get(key, 0) for key in metrics if key != 'negative_details')
    percentage_score = calculate_percentage_score(raw_score, metrics)
    
    return {
        'raw_score': round(raw_score, 2),
        'percentage_score': percentage_score,
        'formatted_score': f"{percentage_score}% ({round(raw_score, 2)})"
    }

# Enhanced quality score calculation
def calculate_quality_score(metrics):
    """Calculate additional quality metrics"""
    activities = metrics.get('activities', 0)
    decisions = metrics.get('decisions', 0)
    forks = metrics.get('forks', 0)
    swimlanes = metrics.get('swimlanes', 0)
    negative_score = metrics.get('negative_score', 0)
    
    # Functional complexity score
    functional_complexity = (activities * 1.5) + (decisions * 2) + (forks * 1.5)
    
    # Organization score
    organization_score = swimlanes * 2
    
    # Structural integrity (negative penalties impact this)
    structural_integrity = max(0, 10 + negative_score)  # Base 10, reduced by negatives
    
    return {
        'functional_complexity': round(functional_complexity, 2),
        'organization_score': round(organization_score, 2),
        'structural_integrity': round(structural_integrity, 2)
    }

# TEST SECTION - Verify the updated scoring system with if-endif as orphaned_activities
print("\n" + "="*70)
print("TESTING THE IF-ENDIF AS ORPHANED_ACTIVITIES SCORING SYSTEM")
print("="*70)

test_code = '''@startuml
start
:Visitor views article;
:Click "Add Comment" button;

if (Visitor logged in?) then (yes)
  :Display comment form;
else (no)
  :Display login form;
  :Visitor logs in;
  :Display comment form;
endif

:Enter comment text;
:Click "Submit" button;

if (Comment valid?) then (yes)
  :Save comment to database;
  :Display comment below article;
  :Show success message;
else (no)
  :Display validation error;
  :Return to comment form;
endif

stop
@enduml'''

test_metrics = evaluate_plantuml(test_code)
test_score_result = compute_score(test_metrics)
test_quality = calculate_quality_score(test_metrics)

print(f"✅ IF-ENDIF AS ORPHANED_ACTIVITIES SCORING RESULTS:")
print(f"   Activities: {test_metrics['activities']} × 3.0 = {test_metrics['activities'] * 3.0}")
print(f"   Decisions: {test_metrics['decisions']} × 4.0 = {test_metrics['decisions'] * 4.0}")
print(f"   Raw Score: {test_score_result['raw_score']}")
print(f"   Percentage Score: {test_score_result['percentage_score']}%")
print(f"   Formatted Score: {test_score_result['formatted_score']}")
print(f"   Total Negative Score: {test_metrics['negative_score']}")
print(f"   Negative Details: {test_metrics['negative_details']}")
print(f"\n📊 QUALITY METRICS:")
print(f"   Functional Complexity: {test_quality['functional_complexity']}")
print(f"   Organization Score: {test_quality['organization_score']}")
print(f"   Structural Integrity: {test_quality['structural_integrity']}")
print("="*70)

# Read the Excel file
df = pd.read_excel(file_path, engine='openpyxl')

# Target columns
target_columns = ['GBT001','GBT002','GBT003','GBT051','GBT052',
                  'GBT053','GBT091','GBT092','GBT093','CLD001',
                  'CLD002','CLD003','CLD051','CLD052','CLD053',
                  'CLD091','CLD092','CLD093','DPS001','DPS002',
                  'DPS003','DPS051','DPS052','DPS053','DPS091',
                  'DPS092','DPS093','GBN001','GBN002','GBN003',
                  'CLM001','CLM051','CLM091','CLM002','CLM052',
                  'CLM092','CLM003','CLM053','CLM093','DPM001',
                  'DPM051','DPM091','DPM002','DPM052','DPM092',
                  'DPM003','DPM053','DPM093','GPM001','GPM051',
                  'GPM091','GPM002','GPM052','GPM092','GPM003',
                  'GPM053','GPM093','GPN001','GPN002','GPN003']

# Calculate average metrics across all codes
avg_metrics = calculate_average_metrics(df, target_columns)

# Prepare results and parameter counting
results = []
parameter_counts = {}
negative_score_details = []

# Initialize parameter counting structure
for col in target_columns:
    parameter_counts[col] = {
        'total_activities': 0,
        'total_decisions': 0,
        'total_forks': 0,
        'total_swimlanes': 0,
        'total_detaches': 0,
        'total_options': 0,
        'total_edges': 0,
        'total_notes': 0,
        'total_partitions': 0,
        'total_numberline': 0,
        'total_negative_score': 0,
        'total_raw_score': 0,
        'total_percentage_score': 0,
        'total_functional_complexity': 0,
        'total_organization_score': 0,
        'total_structural_integrity': 0,
        'valid_entries': 0
    }

# Process each row with enhanced metrics treating if-endif as orphaned_activities
for index, row in df.iterrows():
    row_result = {}
    
    # Add original row index for reference
    row_result['original_row_index'] = index
    
    for col in target_columns:
        try:
            code = row[col]
            
            # Handle missing or null values
            if pd.isna(code) or code is None:
                code = ""
            
            metrics = evaluate_plantuml(code)
            score_result = compute_score(metrics)
            quality_metrics = calculate_quality_score(metrics)
            
            # Add all metrics including negative scores
            for key in metrics:
                if key != 'negative_details':
                    row_result[f'{col}_{key}'] = metrics[key]
            
            # Add both raw and percentage scores
            row_result[f'{col}_raw_score'] = score_result['raw_score']
            row_result[f'{col}_percentage_score'] = score_result['percentage_score']
            row_result[f'{col}_formatted_score'] = score_result['formatted_score']
            
            # Add quality metrics
            row_result[f'{col}_functional_complexity'] = quality_metrics['functional_complexity']
            row_result[f'{col}_organization_score'] = quality_metrics['organization_score']
            row_result[f'{col}_structural_integrity'] = quality_metrics['structural_integrity']
            
            # Calculate ratios
            activities_count = metrics['activities']
            decisions_count = metrics['decisions']
            decisions_activities_ratio = decisions_count / max(activities_count, 1)
            row_result[f'{col}_decisions_activities_ratio'] = round(decisions_activities_ratio, 3)
            
            # Calculate comparison with average
            avg_lines = avg_metrics.get('avg_numberline', 0)
            if avg_lines > 0:
                lines_vs_avg = metrics['numberline'] / avg_lines
                row_result[f'{col}_lines_vs_average'] = round(lines_vs_avg, 3)
            else:
                row_result[f'{col}_lines_vs_average'] = 0
            
            # **ENHANCED**: Store negative score details WITH original PlantUML code
            if metrics['negative_details']:
                for detail in metrics['negative_details']:
                    negative_score_details.append({
                        'row_index': index,
                        'column': col,
                        'element': detail['element'],
                        'count': detail['count'],
                        'score': detail['score'],
                        'explanation': detail['explanation'],
                        'original_code': code  # **ADDED: Original PlantUML code**
                    })
            
            # Accumulate parameter counts with percentage scores
            if code.strip():  # Only count non-empty entries
                parameter_counts[col]['total_activities'] += metrics['activities']
                parameter_counts[col]['total_decisions'] += metrics['decisions']
                parameter_counts[col]['total_forks'] += metrics['forks']
                parameter_counts[col]['total_swimlanes'] += metrics['swimlanes']
                parameter_counts[col]['total_detaches'] += metrics['detaches']
                parameter_counts[col]['total_options'] += metrics['options']
                parameter_counts[col]['total_edges'] += metrics['edges']
                parameter_counts[col]['total_notes'] += metrics['notes']
                parameter_counts[col]['total_partitions'] += metrics['partitions']
                parameter_counts[col]['total_numberline'] += metrics['numberline']
                parameter_counts[col]['total_negative_score'] += metrics['negative_score']
                parameter_counts[col]['total_raw_score'] += score_result['raw_score']
                parameter_counts[col]['total_percentage_score'] += score_result['percentage_score']
                parameter_counts[col]['total_functional_complexity'] += quality_metrics['functional_complexity']
                parameter_counts[col]['total_organization_score'] += quality_metrics['organization_score']
                parameter_counts[col]['total_structural_integrity'] += quality_metrics['structural_integrity']
                parameter_counts[col]['valid_entries'] += 1
            
        except Exception as e:
            print(f"Error processing row {index}, column {col}: {e}")
            # Set default values in case of error
            for key in weights.keys():
                row_result[f'{col}_{key}'] = 0
            row_result[f'{col}_raw_score'] = 0
            row_result[f'{col}_percentage_score'] = 0
            row_result[f'{col}_formatted_score'] = "0% (0.0)"
            row_result[f'{col}_functional_complexity'] = 0
            row_result[f'{col}_organization_score'] = 0
            row_result[f'{col}_structural_integrity'] = 0
            row_result[f'{col}_decisions_activities_ratio'] = 0
            row_result[f'{col}_lines_vs_average'] = 0
    
    results.append(row_result)

# Calculate column-level ratios with percentage scores
for col in target_columns:
    total_decisions = parameter_counts[col]['total_decisions']
    total_activities = parameter_counts[col]['total_activities']
    parameter_counts[col]['decisions_activities_ratio'] = round(
        total_decisions / max(total_activities, 1), 3
    )
    
    # Calculate averages
    valid_entries = parameter_counts[col]['valid_entries']
    if valid_entries > 0:
        parameter_counts[col]['avg_raw_score'] = round(
            parameter_counts[col]['total_raw_score'] / valid_entries, 2
        )
        parameter_counts[col]['avg_percentage_score'] = round(
            parameter_counts[col]['total_percentage_score'] / valid_entries, 1
        )
        parameter_counts[col]['avg_negative_score'] = round(
            parameter_counts[col]['total_negative_score'] / valid_entries, 2
        )
        parameter_counts[col]['avg_functional_complexity'] = round(
            parameter_counts[col]['total_functional_complexity'] / valid_entries, 2
        )
        parameter_counts[col]['avg_organization_score'] = round(
            parameter_counts[col]['total_organization_score'] / valid_entries, 2
        )
        parameter_counts[col]['avg_structural_integrity'] = round(
            parameter_counts[col]['total_structural_integrity'] / valid_entries, 2
        )
    else:
        parameter_counts[col]['avg_raw_score'] = 0
        parameter_counts[col]['avg_percentage_score'] = 0
        parameter_counts[col]['avg_negative_score'] = 0
        parameter_counts[col]['avg_functional_complexity'] = 0
        parameter_counts[col]['avg_organization_score'] = 0
        parameter_counts[col]['avg_structural_integrity'] = 0

# Create enhanced parameter counts DataFrame with percentage scores
parameter_data = []
for col in target_columns:
    param_row = {
        'column_name': col,
        'total_activities': parameter_counts[col]['total_activities'],
        'total_decisions': parameter_counts[col]['total_decisions'],
        'total_forks': parameter_counts[col]['total_forks'],
        'total_swimlanes': parameter_counts[col]['total_swimlanes'],
        'total_detaches': parameter_counts[col]['total_detaches'],
        'total_options': parameter_counts[col]['total_options'],
        'total_edges': parameter_counts[col]['total_edges'],
        'total_notes': parameter_counts[col]['total_notes'],
        'total_partitions': parameter_counts[col]['total_partitions'],
        'total_numberline': parameter_counts[col]['total_numberline'],
        'total_negative_score': parameter_counts[col]['total_negative_score'],
        'decisions_activities_ratio': parameter_counts[col]['decisions_activities_ratio'],
        'avg_raw_score': parameter_counts[col]['avg_raw_score'],
        'avg_percentage_score': parameter_counts[col]['avg_percentage_score'],
        'avg_negative_score': parameter_counts[col]['avg_negative_score'],
        'avg_functional_complexity': parameter_counts[col]['avg_functional_complexity'],
        'avg_organization_score': parameter_counts[col]['avg_organization_score'],
        'avg_structural_integrity': parameter_counts[col]['avg_structural_integrity'],
        'valid_entries': parameter_counts[col]['valid_entries']
    }
    parameter_data.append(param_row)

# Add totals row
totals_row = {'column_name': 'TOTAL'}
for param in ['total_activities', 'total_decisions', 'total_forks', 'total_swimlanes', 
              'total_detaches', 'total_options', 'total_edges', 'total_notes', 
              'total_partitions', 'total_numberline', 'total_negative_score', 
              'total_raw_score', 'total_percentage_score',
              'total_functional_complexity', 'total_organization_score', 
              'total_structural_integrity', 'valid_entries']:
    totals_row[param] = sum(parameter_counts[col][param] for col in target_columns)

# Calculate overall ratios and averages
total_decisions = totals_row['total_decisions']
total_activities = totals_row['total_activities']
totals_row['decisions_activities_ratio'] = round(total_decisions / max(total_activities, 1), 3)

total_valid = totals_row['valid_entries']
totals_row['avg_raw_score'] = round(totals_row['total_raw_score'] / max(total_valid, 1), 2)
totals_row['avg_percentage_score'] = round(totals_row['total_percentage_score'] / max(total_valid, 1), 1)
totals_row['avg_negative_score'] = round(totals_row['total_negative_score'] / max(total_valid, 1), 2)
totals_row['avg_functional_complexity'] = round(totals_row['total_functional_complexity'] / max(total_valid, 1), 2)
totals_row['avg_organization_score'] = round(totals_row['total_organization_score'] / max(total_valid, 1), 2)
totals_row['avg_structural_integrity'] = round(totals_row['total_structural_integrity'] / max(total_valid, 1), 2)

parameter_data.append(totals_row)

# Create DataFrames
results_df = pd.DataFrame(results)
parameter_df = pd.DataFrame(parameter_data)
negative_details_df = pd.DataFrame(negative_score_details)  # **NOW INCLUDES 'original_code' COLUMN**

# Create average metrics DataFrame
avg_metrics_data = []
for key, value in avg_metrics.items():
    avg_metrics_data.append({
        'metric': key,
        'value': round(value, 4)
    })
avg_metrics_df = pd.DataFrame(avg_metrics_data)

# Calculate statistical metrics
statistical_metrics = calculate_statistical_metrics_for_columns(results_df)

# **ENHANCED**: Save results with if-endif treated as orphaned_activities
output_excel_file = "report/01_Syntatic_analysis.xlsx"

with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
    # Main results sheet
    results_df.to_excel(writer, sheet_name='Main Results', index=False)
    
    # Enhanced parameter counts sheet
    parameter_df.to_excel(writer, sheet_name='Parameters Summary', index=False)
    
    # **ENHANCED**: Negative scores details sheet WITH original PlantUML code
    negative_details_df.to_excel(writer, sheet_name='Negative Scores Details', index=False)
    
    # Average metrics sheet
    avg_metrics_df.to_excel(writer, sheet_name='Average Metrics', index=False)
    
    # Statistical metrics sheet
    statistical_metrics.to_excel(writer, sheet_name='Statistical Metrics', index=False)

print(f"\n✅ IF-ENDIF AS ORPHANED_ACTIVITIES results successfully saved to {output_excel_file}")

# Also save CSV for compatibility
output_csv_file = "report/01_Syntatic_Analysis.csv"
if results:
    keys = results[0].keys()
    
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

# Enhanced summary statistics with if-endif as orphaned_activities
print(f"\n📊 IF-ENDIF AS ORPHANED_ACTIVITIES SCORING SYSTEM SUMMARY:")
print(f"   Total rows processed: {len(results)}")
print(f"   Total Activities: {totals_row['total_activities']}")
print(f"   Total Decisions: {totals_row['total_decisions']}")
print(f"   Total Negative Score: {totals_row['total_negative_score']}")
print(f"   Overall Average Raw Score: {totals_row['avg_raw_score']}")
print(f"   Overall Average Percentage Score: {totals_row['avg_percentage_score']}%")
print(f"   Overall Average Functional Complexity: {totals_row['avg_functional_complexity']}")
print(f"   Overall Average Organization Score: {totals_row['avg_organization_score']}")
print(f"   Overall Average Structural Integrity: {totals_row['avg_structural_integrity']}")

print(f"\n🎯 KEY CHANGES MADE:")
print(f"   ✅ **MODIFIED**: If-endif imbalances now treated as 'orphaned_activities'")
print(f"   ✅ **NO PENALTY**: If-endif imbalances receive 0 penalty points")
print(f"   ✅ **IMPROVED**: Enhanced if-detection handles all PlantUML formats")
print(f"   ✅ **ADDED**: Original PlantUML code included in Negative Scores Details sheet")
print(f"   ✅ **MAINTAINED**: All other penalty and scoring features preserved")

print(f"\n📈 CURRENT PENALTY SYSTEM:")
print(f"   🔴 Critical Penalties:")
print(f"      Missing Start/Stop: -10 points each")
print(f"      Fork-join Imbalances: -3 points each")
print(f"      Switch-endswitch Imbalances: -3 points each")
print(f"   🟡 High Value Elements:")
print(f"      Decisions: +4.0 points each")
print(f"      Activities: +3.0 points each") 
print(f"      Forks: +3.0 points each")
print(f"   🟢 Medium Value Elements:")
print(f"      Swimlanes: +2.0 points each")
print(f"      Partitions: +1.5 points each")
print(f"      Edges: +1.2 points each")
print(f"   🔵 Supporting Elements:")
print(f"      Options: +1.0 points each")
print(f"      Notes: +0.8 points each")
print(f"      Lines: +0.05 points each")
print(f"   ⚪ NO PENALTY:")
print(f"      If-endif Imbalances: 0 points (recorded as orphaned_activities)")
print(f"      Traditional Orphaned Activities: 0 points")

print(f"\n✅ IF-ENDIF AS ORPHANED_ACTIVITIES processing completed successfully!")
