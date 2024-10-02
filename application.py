
import pdfplumber
from transformers import pipeline

# Load the transformer model for NER
nlp = pipeline('ner', model='bert-base-cased', tokenizer='bert-base-cased')

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_entities_from_text(text):
    """
    Extract entities from text using the transformer model.

    Args:
        text (str): Extracted text.

    Returns:
        dict: Extracted entities.
    """
    entities = nlp(text)
    entity_dict = {}
    for ent in entities:
        label = ent['entity']
        text = ent['word']
        entity_dict[label] = entity_dict.get(label, []) + [text]
    return entity_dict

def analyze_sentences(text):
    """
    Analyze sentences for impact, numerical results, and action verbs.

    Args:
        text (str): Extracted text.

    Returns:
        dict: Analysis results.
    """
    doc = nlp(text)
    impact_count = 0
    numerical_impact_count = 0
    action_verb_count = 0

    action_verbs = ["managed", "developed", "led", "created"]
    numerical_indicators = ["%", "$", "million", "billion"]

    sentences = text.split('.')
    for sent in sentences:
        if any(verb in sent.lower() for verb in action_verbs):
            action_verb_count += 1
        if any(indicator in sent for indicator in numerical_indicators):
            numerical_impact_count += 1
        if "impact" in sent.lower() or "result" in sent.lower():
            impact_count += 1

    return {
        "impact_count": impact_count,
        "numerical_impact_count": numerical_impact_count,
        "action_verb_count": action_verb_count,
        "total_sentences": len(sentences)
    }

def calculate_final_score(analysis):
    """
    Calculate a final score out of 100 based on the analysis.

    Args:
        analysis (dict): Dictionary containing analysis results.

    Returns:
        dict: Final scores.
    """
    impact_score = (analysis["impact_count"] * 7.5) / analysis["total_sentences"]
    numerical_impact_score = (analysis["numerical_impact_count"] * 12.5) / analysis["total_sentences"]
    action_verb_score = (analysis["action_verb_count"] * 5) / analysis["total_sentences"]

    final_score = impact_score + numerical_impact_score + action_verb_score
    final_score = min(final_score, 100)

    return {
        "impact_score": impact_score,
        "numerical_impact_score": numerical_impact_score,
        "action_verb_score": action_verb_score,
        "final_score": final_score
    }

if __name__ == "__main__":
    pdf_path = 'data/21cs3016_DivyanshSharma_cv.pdf'
    pdf_text = extract_text_from_pdf(pdf_path)
    entities = extract_entities_from_text(pdf_text)
    analysis = analyze_sentences(pdf_text)
    final_score = calculate_final_score(analysis)
    
    print("Extracted Entities:", entities)
    print("Analysis Results:", analysis)
    print("Final Score:", final_score)