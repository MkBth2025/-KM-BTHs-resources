from transformers import pipeline, AutoTokenizer, AutoModel
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import os
import re
import torch

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
        
        # NER model (Named Entity Recognition)
        print("📥 Loading NER model...")
        try:
            ner_model_name = MODELS_CONFIG['ner'].get(language, MODELS_CONFIG['ner']['english'])
            models['ner'] = pipeline(
                "ner", 
                model=ner_model_name,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ NER model loaded: {ner_model_name}")
        except Exception as e:
            print(f"⚠️  NER model failed, using basic model: {e}")
            models['ner'] = pipeline("ner", aggregation_strategy="simple")
        
        # Sentiment analysis model (optional for quality enhancement)
        print("📥 Loading sentiment analysis model...")
        try:
            if language == 'persian':
                models['sentiment'] = pipeline("sentiment-analysis", 
                                             model="HooshvareLab/bert-fa-base-uncased-sentiment-digikala")
            else:
                models['sentiment'] = pipeline("sentiment-analysis")
            print("✅ Sentiment analysis model loaded")
        except Exception as e:
            print(f"⚠️  Sentiment model not loaded: {e}")
            models['sentiment'] = None
        
        # Feature extraction model (for deeper analysis)
        print("📥 Loading feature extraction model...")
        try:
            if language == 'persian':
                models['features'] = pipeline("feature-extraction", 
                                            model="HooshvareLab/bert-fa-base-uncased")
            else:
                models['features'] = pipeline("feature-extraction")
            print("✅ Feature extraction model loaded")
        except Exception as e:
            print(f"⚠️  Feature extraction model not loaded: {e}")
            models['features'] = None
        
        print("✅ Hugging Face models initialized successfully")
        return models
        
    except Exception as e:
        print(f"❌ Failed to initialize Hugging Face models: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Install required packages: pip install transformers torch")
        print("   2. Check internet connection for model download")
        print("   3. Try with smaller models if memory is limited")
        return None

def analyze_with_huggingface(text, models):
    """Analyze text using Hugging Face models"""
    if models is None:
        return analyze_without_nlp(text)
    
    try:
        analysis_result = {
            'entities': [],
            'relationships': [],
            'verbs': [],
            'analysis_method': 'Hugging Face Transformers'
        }
        
        # Named Entity Recognition
        if models.get('ner'):
            ner_results = models['ner'](text)
            
            # Extract entities
            entities = []
            for entity in ner_results:
                if entity['entity_group'] in ['PERSON', 'ORG', 'MISC'] or entity['score'] > 0.8:
                    # Clean entity name
                    clean_entity = entity['word'].replace('##', '').strip()
                    if len(clean_entity) > 2:
                        entities.append(clean_entity)
            
            analysis_result['entities'] = list(set(entities))
        
        # POS analysis with pattern matching (when direct POS unavailable in pipeline)
        analysis_result.update(extract_linguistic_features(text))
        
        # Sentiment analysis (for prompt quality enhancement)
        if models.get('sentiment'):
            try:
                sentiment = models['sentiment'](text)
                analysis_result['sentiment'] = sentiment[0]['label'] if sentiment else 'NEUTRAL'
            except:
                analysis_result['sentiment'] = 'NEUTRAL'
        
        return analysis_result
        
    except Exception as e:
        print(f"⚠️  Hugging Face analysis failed, using fallback: {e}")
        return analyze_without_nlp(text)

def extract_linguistic_features(text):
    """Extract linguistic features using advanced pattern matching"""
    
    # Clean text
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = re.findall(r'\b\w+\b', text_clean)
    
    # Advanced patterns for noun detection
    noun_patterns = [
        'system', 'manager', 'service', 'controller', 'handler', 'processor',
        'interface', 'class', 'object', 'entity', 'model', 'data', 'base',
        'tion', 'sion', 'ness', 'ment', 'ship', 'hood', 'dom', 'ism'
    ]
    
    # Identify potential entities
    entities = []
    
    # 1. Capitalized words (likely class names)
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
    entities.extend([word for word in capitalized if len(word) > 2])
    
    # 2. Words matching noun patterns
    pattern_entities = []
    for word in words:
        if any(pattern in word for pattern in noun_patterns):
            pattern_entities.append(word.capitalize())
    
    entities.extend(pattern_entities)
    
    # Identify verbs and activity patterns
    verb_patterns = [
        'create', 'make', 'build', 'design', 'implement', 'manage', 'handle',
        'process', 'execute', 'run', 'start', 'stop', 'update', 'delete',
        'insert', 'select', 'modify', 'send', 'receive', 'store', 'retrieve',
        'validate', 'authenticate', 'authorize', 'connect', 'disconnect',
        'login', 'logout', 'register', 'submit', 'approve', 'reject', 'notify'
    ]
    
    found_verbs = [word for word in words if word in verb_patterns]
    
    # Generate activity flow based on linguistic patterns
    activity_flow = generate_activity_flow(text, entities, found_verbs)
    
    return {
        'entities': list(set(entities)) if entities else ['User', 'System'],
        'verbs': found_verbs if found_verbs else ['start', 'process', 'end'],
        'activity_flow': activity_flow,
        'relationships': generate_relationships(text, entities)
    }

def generate_activity_flow(text, entities, verbs):
    """Generate activity flow patterns from text"""
    flow_patterns = []
    
    # Common activity flow patterns
    if any(word in text.lower() for word in ['if', 'when', 'condition', 'check']):
        flow_patterns.append('decision')
    if any(word in text.lower() for word in ['parallel', 'simultaneously', 'at the same time']):
        flow_patterns.append('parallel')
    if any(word in text.lower() for word in ['loop', 'repeat', 'iterate', 'while']):
        flow_patterns.append('loop')
    
    return flow_patterns

def generate_relationships(text, entities):
    """Generate relationships based on text patterns"""
    relationships = []
    
    # Relationship patterns for activities
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
    """Basic analysis without complex models"""
    
    # Simple entity extraction
    words = re.findall(r'\b[A-Za-z]+\b', text)
    
    # System keywords
    system_entities = ['user', 'system', 'admin', 'database', 'server', 'client', 'visitor']
    activity_entities = ['form', 'page', 'button', 'field', 'message', 'notification']
    
    entities = []
    for word in words:
        word_lower = word.lower()
        if word_lower in system_entities + activity_entities or len(word) > 8:
            entities.append(word.capitalize())
    
    # Activity verbs
    activity_verbs = ['start', 'login', 'submit', 'validate', 'process', 'approve', 'notify', 'end']
    found_verbs = [verb for verb in activity_verbs if verb in text.lower()]
    
    return {
        'entities': list(set(entities)) if entities else ['User', 'System'],
        'relationships': [('sequence', 'Start', 'End')] if len(entities) >= 2 else [],
        'verbs': found_verbs if found_verbs else ['start', 'process', 'end'],
        'analysis_method': 'Basic Pattern Matching (No NLP models)'
    }

def create_standard_activity_prompt(raw_prompt, models):
    """Generate standardized prompt for PlantUML activity diagram"""
    analysis = analyze_with_huggingface(raw_prompt, models)
    
    entities_str = ', '.join(analysis['entities'][:10]) if analysis['entities'] else 'None detected'
    verbs_str = ', '.join(analysis['verbs'][:8]) if analysis['verbs'] else 'None detected'
    relationships_str = str(analysis['relationships'][:3]) if analysis['relationships'] else 'None detected'
    
    # Sentiment analysis (optional)
    sentiment_info = ""
    if 'sentiment' in analysis and analysis['sentiment']:
        sentiment_info = f"\n- Sentiment: {analysis['sentiment']}"
    
    # Activity flow patterns
    flow_info = ""
    if 'activity_flow' in analysis and analysis['activity_flow']:
        flow_info = f"\n- Flow Patterns: {', '.join(analysis['activity_flow'])}"
    
    prompt_template = f"""
Generate PlantUML ACTIVITY diagram for:
"{raw_prompt}"

 AI Analysis Method: {analysis['analysis_method']}
- Key Entities: {entities_str}
- Action Verbs: {verbs_str}
- Relationships: {relationships_str}{sentiment_info}{flow_info}

 Requirements:
1. Start with (*) and end with (*)
2. Represent all main actions as activities in rectangles
3. Use decision diamonds <> for conditional flows
4. Include fork and join for parallel activities
5. Use proper activity flow arrows -->
6. Add swimlanes if multiple actors are involved
7. Include notes and comments for clarity
8. Show error handling and alternative flows

Output Format: Complete PlantUML activity diagram code starting with @startuml and ending with @enduml

Enhanced AI Insight: Focus on modeling the complete workflow with proper activity flow, decisions, and parallel processes as identified by the AI analysis.
"""
    return prompt_template

def process_csv(input_csv, output_csv, models, batch_size=10):
    """Process CSV with Hugging Face models for Activity Diagrams"""
    
    # Check input file existence
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file '{input_csv}' not found!")
        print(f"📝 Please ensure your CSV file is named '{input_csv}' and contains a 'prompt' column")
        return
    
    try:
        # Load CSV
        print(f"📖 Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        total_rows = len(df)
        
        if total_rows == 0:
            print("❌ No data found in the CSV file!")
            return
        
        # Show CSV info
        print(f"📊 Found {total_rows} rows to process")
        print(f"📋 Columns available: {list(df.columns)}")
        
        results = []
        errors = []
        start_time = time.time()
        
        # Enhanced progress bar
        with tqdm(
            total=total_rows,
            desc="🤖 AI Processing",
            unit="prompt",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        ) as pbar:
            
            for index, row in df.iterrows():
                try:
                    # Calculate speed
                    elapsed = time.time() - start_time
                    speed = (index + 1) / elapsed if elapsed > 0 else 0
                    
                    pbar.set_postfix({
                        'Row': f"{index + 1}",
                        'Errors': len(errors),
                        'Speed': f"{speed:.1f}/s",
                        'Model': '🤖 HF'
                    }, refresh=True)
                    
                    # Get prompt
                    raw_prompt = None
                    for col in ['prompt', 'text', 'description', 'input', 'query']:
                        if col in row and pd.notna(row[col]):
                            raw_prompt = str(row[col]).strip()
                            break
                    
                    if not raw_prompt:
                        raw_prompt = str(row.iloc[0]).strip()
                    
                    # Check if empty
                    if not raw_prompt or raw_prompt.lower() in ['nan', 'null', '']:
                        errors.append(f"Row {index + 1}: Empty or invalid prompt")
                        pbar.update(1)
                        continue
                    
                    # Process prompt for Activity Diagram
                    standardized = create_standard_activity_prompt(raw_prompt, models)
                    analysis_info = analyze_with_huggingface(raw_prompt, models)
                    
                    results.append({
                        'row_number': index + 1,
                        'original_prompt': raw_prompt,
                        'ai_enhanced_activity_prompt': standardized,
                        'ai_analysis_method': analysis_info['analysis_method'],
                        'detected_entities': ', '.join(analysis_info['entities'][:5]),
                        'detected_actions': ', '.join(analysis_info['verbs'][:3]),
                        'character_count': len(raw_prompt),
                        'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'processing_status': 'success'
                    })
                    
                except Exception as e:
                    error_msg = f"Row {index + 1}: {str(e)}"
                    errors.append(error_msg)
                    
                    results.append({
                        'row_number': index + 1,
                        'original_prompt': raw_prompt if 'raw_prompt' in locals() else 'N/A',
                        'ai_enhanced_activity_prompt': f"ERROR: {str(e)}",
                        'ai_analysis_method': 'Error',
                        'detected_entities': 'Error',
                        'detected_actions': 'Error',
                        'character_count': 0,
                        'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'processing_status': 'error'
                    })
                
                finally:
                    pbar.update(1)
                
                # Progressive save
                # Progressive save
                if (index + 1) % batch_size == 0:
                 temp_df = pd.DataFrame(results)
                 # Create directory if it doesn't exist
                 os.makedirs('report', exist_ok=True)
                 temp_output = f"report/backup_{output_csv}"
                 temp_df.to_csv(temp_output, index=False)
        
        # Save final results
        print(f"\n💾 Saving AI-enhanced activity diagram prompts to {output_csv}...")
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv, index=False)
        
        # Detailed summary
        successful = len([r for r in results if r['processing_status'] == 'success'])
        failed = len(errors)
        total_time = time.time() - start_time
        avg_speed = total_rows / total_time
        
        print(f"\n🎉 AI Activity Diagram Processing Complete!")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({avg_speed:.1f} prompts/sec)")
        print(f"✅ Successfully processed: {successful}/{total_rows} ({successful/total_rows*100:.1f}%)")
        print(f"❌ Failed: {failed}/{total_rows}")
        print(f"📁 Output saved to: {output_csv}")
        print(f"🤖 AI Enhancement: Hugging Face Transformers for Activity Diagrams")
        
        if errors:
            print(f"\n⚠️  First 5 errors:")
            for error in errors[:5]:
                print(f"   • {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")
        
        # Clean up backup file
        backup_file = f"backup_{output_csv}"
        if os.path.exists(backup_file):
            os.remove(backup_file)
            
    except Exception as e:
        print(f"💥 Fatal error processing CSV: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function with Hugging Face support for Activity Diagrams"""
    input_file = "data/row_promt.csv"
    output_file = "report/Modified_prompt.csv"
    models = None
    
    
    try:
        print("🤖 PlantUML Activity Diagram AI Enhancer v3.0 (Powered by Hugging Face)")
        print("=" * 75)
        
        # Check input file
        if not os.path.exists(input_file):
            print(f"📝 Creating sample {input_file} file...")
            sample_data = pd.DataFrame({
                'prompt': [
                    'User login process with authentication and validation',
                    'Online order processing workflow from cart to delivery',
                    'Document approval workflow with multiple reviewers',
                    'Customer registration process with email verification'
                ]
            })
            sample_data.to_csv(input_file, index=False)
            print(f"✅ Sample file created with activity workflow examples")
            print(f"📝 Please edit {input_file} with your activity prompts and run again.")
            return
        
        # Setup Hugging Face models
        print("\n🔧 Setting up AI models for activity analysis...")
        models = initialize_huggingface_models('english')  # Change to 'persian' for Persian
        
        if models is None:
            print("\n⚠️  Warning: AI models unavailable - using basic pattern matching")
            print("📈 Results will be less detailed but still functional")
            response = input("\n🤔 Continue with basic analysis? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("👋 Exiting. Please check internet connection and try again.")
                return
        else:
            print("🎯 AI models ready for activity diagram analysis!")
        
        # Process CSV
        process_csv(input_file, output_file, models)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean GPU memory (if used)
        if models:
            try:
                print("\n🧹 Cleaning up AI models...")
                # Free GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("✅ AI models cleaned up successfully")
            except Exception as e:
                print(f"⚠️  Warning: Error cleaning up models: {e}")

if __name__ == "__main__":
    main()
