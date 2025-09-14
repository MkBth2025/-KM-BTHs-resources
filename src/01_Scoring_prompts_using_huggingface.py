from transformers import pipeline, AutoTokenizer, AutoModel
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import os
import re
import torch
import math

# Hugging Face models configuration
MODELS_CONFIG = {
    'ner': {
        'english': 'dbmdz/bert-large-cased-finetuned-conll03-english',
        'multilingual': 'xlm-roberta-large-finetuned-conll03-english',
        'persian': 'HooshvareLab/bert-fa-base-uncased'
    },
    'pos': {
        'english': 'vblagoje/bert-english-uncased-finetuned-pos',
        'multilingual': 'wietsedv/xlm-roberta-base-finetuned-udpos28-en'
    }
}

def initialize_huggingface_models(language='english'):
    """Initialize Hugging Face models"""
    models = {}
    
    try:
        print("🚀 Initializing Hugging Face models...")
        print("⏳ This may take a few moments for first-time download...")
        
        # NER model
        print("📅 Loading NER model...")
        try:
            ner_model_name = MODELS_CONFIG['ner'].get(language, MODELS_CONFIG['ner']['english'])
            models['ner'] = pipeline("ner", model=ner_model_name, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
            print(f"✅ NER model loaded: {ner_model_name}")
        except Exception as e:
            print(f"⚠️ NER model failed: {e}")
            models['ner'] = pipeline("ner", aggregation_strategy="simple")
        
        # Sentiment model
        print("📅 Loading sentiment analysis model...")
        try:
            if language == 'persian':
                models['sentiment'] = pipeline("sentiment-analysis", model="HooshvareLab/bert-fa-base-uncased-sentiment-digikala")
            else:
                models['sentiment'] = pipeline("sentiment-analysis")
            print("✅ Sentiment model loaded")
        except Exception as e:
            print(f"⚠️ Sentiment model failed: {e}")
            models['sentiment'] = None
        
        # Feature extraction
        print("📅 Loading feature extraction model...")
        try:
            if language == 'persian':
                models['features'] = pipeline("feature-extraction", model="HooshvareLab/bert-fa-base-uncased")
            else:
                models['features'] = pipeline("feature-extraction")
            print("✅ Feature extraction model loaded")
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            models['features'] = None
        
        print("✅ All models initialized successfully")
        return models
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

def calculate_complexity_rate(analysis_info, char_count):
    num_entities = len(set(analysis_info.get('entities', [])))
    num_verbs = len(set(analysis_info.get('verbs', [])))
    num_flows = len(analysis_info.get('activity_flow', [])) if 'activity_flow' in analysis_info else 0
    complexity_score = num_entities + num_verbs + num_flows
    rate = complexity_score / math.log(char_count + 1)
    return min(round(rate, 2), 10.0), complexity_score

def analyze_with_huggingface(text, models):
    if models is None:
        return analyze_without_nlp(text)
    try:
        analysis_result = {
            'entities': [],
            'relationships': [],
            'verbs': [],
            'analysis_method': 'Hugging Face Transformers'
        }
        
        if models.get('ner'):
            ner_results = models['ner'](text)  # Fixed: added parentheses to call the function
            entities = []
            for entity in ner_results:
                if entity['entity_group'] in ['PERSON', 'ORG', 'MISC'] or entity['score'] > 0.8:
                    clean_entity = entity['word'].replace('##', '').strip()
                    if len(clean_entity) > 2:
                        entities.append(clean_entity)
            analysis_result['entities'] = list(set(entities))
        
        analysis_result.update(extract_linguistic_features(text))
        
        if models.get('sentiment'):
            try:
                sentiment = models['sentiment'](text)  # Fixed: added parentheses to call the function
                analysis_result['sentiment'] = sentiment[0]['label'] if sentiment else 'NEUTRAL'
            except:
                analysis_result['sentiment'] = 'NEUTRAL'
        
        return analysis_result
    except Exception as e:
        print(f"⚠️ Analysis failed: {e}")
        return analyze_without_nlp(text)

def extract_linguistic_features(text):
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = re.findall(r'\b\w+\b', text_clean)
    
    noun_patterns = ['system', 'manager', 'service', 'controller', 'handler', 'processor', 'interface', 'class', 'object', 'entity', 'model', 'data', 'base', 'tion', 'sion', 'ness', 'ment', 'ship', 'hood', 'dom', 'ism']
    entities = []
    
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
    entities.extend([word for word in capitalized if len(word) > 2])
    
    for word in words:
        if any(pattern in word for pattern in noun_patterns):
            entities.append(word.capitalize())
    
    verb_patterns = ['create', 'make', 'build', 'design', 'implement', 'manage', 'handle', 'process', 'execute', 'run', 'start', 'stop', 'update', 'delete', 'insert', 'select', 'modify', 'send', 'receive', 'store', 'retrieve', 'validate', 'authenticate', 'authorize', 'connect', 'disconnect', 'login', 'logout', 'register', 'submit', 'approve', 'reject', 'notify']
    found_verbs = [word for word in words if word in verb_patterns]
    
    activity_flow = generate_activity_flow(text, entities, found_verbs)
    
    return {
        'entities': list(set(entities)) if entities else ['User', 'System'],
        'verbs': found_verbs if found_verbs else ['start', 'process', 'end'],
        'activity_flow': activity_flow,
        'relationships': generate_relationships(text, entities)
    }

def generate_activity_flow(text, entities, verbs):
    flow_patterns = []
    if any(word in text.lower() for word in ['if', 'when', 'condition', 'check']):
        flow_patterns.append('decision')
    if any(word in text.lower() for word in ['parallel', 'simultaneously', 'at the same time']):
        flow_patterns.append('parallel')
    if any(word in text.lower() for word in ['loop', 'repeat', 'iterate', 'while']):
        flow_patterns.append('loop')
    return flow_patterns

def generate_relationships(text, entities):
    relationships = []
    relation_patterns = [
        (r'(\w+)\s+then\s+(\w+)', 'sequence'),
        (r'(\w+)\s+before\s+(\w+)', 'precedes'),
        (r'(\w+)\s+after\s+(\w+)', 'follows'),
        (r'(\w+)\s+triggers\s+(\w+)', 'triggers'),
        (r'(\w+)\s+leads\s+to\s+(\w+)', 'leads_to')
    ]
    
    for pattern, relation_type in relation_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            subj, obj = match.groups()
            if len(subj) > 2 and len(obj) > 2:
                relationships.append((relation_type, subj.capitalize(), obj.capitalize()))
    
    return relationships[:3]

def analyze_without_nlp(text):
    words = re.findall(r'\b[A-Za-z]+\b', text)
    system_entities = ['user', 'system', 'admin', 'database', 'server', 'client', 'visitor']
    activity_entities = ['form', 'page', 'button', 'field', 'message', 'notification']
    entities = [word.capitalize() for word in words if word.lower() in system_entities + activity_entities or len(word) > 8]
    
    activity_verbs = ['start', 'login', 'submit', 'validate', 'process', 'approve', 'notify', 'end']
    found_verbs = [verb for verb in activity_verbs if verb in text.lower()]
    
    return {
        'entities': list(set(entities)) if entities else ['User', 'System'],
        'relationships': [('sequence', 'Start', 'End')] if len(entities) >= 2 else [],
        'verbs': found_verbs if found_verbs else ['start', 'process', 'end'],
        'analysis_method': 'Basic Pattern Matching (No NLP models)'
    }

def process_csv(input_csv, output_csv, models, batch_size=10):
    if not os.path.exists(input_csv):
        print(f"❌ Input file '{input_csv}' not found!")
        return
    
    df = pd.read_csv(input_csv)
    results = []
    errors = []
    start_time = time.time()
    
    with tqdm(total=len(df), desc="🧐 Processing", unit="prompt") as pbar:
        for index, row in df.iterrows():
            try:
                raw_prompt = next((str(row[col]).strip() for col in ['prompt', 'text', 'description', 'input', 'query'] if col in row and pd.notna(row[col])), str(row.iloc[0]).strip())
                if not raw_prompt or raw_prompt.lower() in ['nan', 'null', '']:
                    raise ValueError("Empty or invalid prompt")
                
                analysis_info = analyze_with_huggingface(raw_prompt, models)
                complexity_rate, complexity_score = calculate_complexity_rate(analysis_info, len(raw_prompt))
                
                results.append({
                    'row_number': index + 1,
                    'original_prompt': raw_prompt,
                    'ai_analysis_method': analysis_info['analysis_method'],
                    'detected_entities': ', '.join(analysis_info['entities'][:5]),
                    'detected_actions': ', '.join(analysis_info['verbs'][:3]),
                    'character_count': len(raw_prompt),
                    'complexity_score': complexity_score,
                    'complexity_rate': complexity_rate,
                    'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_status': 'success'
                })
            except Exception as e:
                errors.append(f"Row {index + 1}: {e}")
                results.append({
                    'row_number': index + 1,
                    'original_prompt': raw_prompt if 'raw_prompt' in locals() else 'N/A',
                    'ai_analysis_method': 'Error',
                    'detected_entities': 'Error',
                    'detected_actions': 'Error',
                    'character_count': 0,
                    'complexity_score': 0,
                    'complexity_rate': 0,
                    'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_status': 'error'
                })
            finally:
                pbar.update(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"📅 Done! Saved results to {output_csv}")
    
    if errors:
        print(f"⚠️ {len(errors)} errors occurred:")
        for e in errors[:5]:
            print(" -", e)

if __name__ == "__main__":
    input_file = "data/row_promt.csv"
    output_file = "report/Scored_prompt.csv"
    
    print("🚀 Initializing models...")
    models = initialize_huggingface_models("english")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("report", exist_ok=True)
    
    process_csv(input_file, output_file, models)