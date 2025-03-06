import modal
from flask import Flask, request, render_template
import pandas as pd
import re
import nltk
import spacy
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from gramformer import Gramformer

# Initialize Modal stub
stub = modal.Stub("writing-feedback-app")

# Define custom image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "flask",
        "pandas",
        "nltk",
        "spacy",
        "torch",
        "spellchecker",
        "gramformer",
        "python-dotenv",
        "transformers"
    )
    .run_commands(
        "python -m spacy download en_core_web_sm",
        "python -m nltk.downloader punkt wordnet omw-1.4"
    )
)

# Store CSV in Modal's persistent storage
cefr_volume = modal.Volume.persisted("cefr-volume")
CEFR_PATH = "/data/cefr.csv"

class NLPInitializer:
    def __init__(self):
        self._download_nltk_resources()
        self.spacy_nlp = spacy.load("en_core_web_sm")
    
    def _download_nltk_resources(self):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

class GrammarChecker:
    def __init__(self):
        self.grammar_fullforms = {
            'ADV': 'Adverb', 'PREP': 'Prepositions', 'PRON': 'Pronoun', 
            'WO': 'Wrong Order', 'VERB': 'Verbs', 'VERB:SVA': 'Singular-Plural',
            'VERB:TENSE': 'Verb Tenses', 'VERB:FORM': 'Verb Forms', 
            'VERB:INFL': 'Verbs', 'SPELL': 'Spelling', 'OTHER': 'Other',
            'NOUN': 'Other', 'NOUN:NUM': 'Singular-Plural', 'DET': 'Articles',
            'MORPH': 'Other', 'ADJ': 'Adjectives', 'PART': 'Other',
            'ORTH': 'Other', 'CONJ': 'Conjugations', 'PUNCT': 'Punctuation'
        }
        self._initialize_gramformer()
        
    def _initialize_gramformer(self):
        torch.manual_seed(1212)
        self.gf = Gramformer(models=1, use_gpu=False, token=False)  # Added token=False
    
    @staticmethod
    def _strikethrough(text):
        return ''.join([f'{c}\u0336' for c in text])
    
    def correct_grammar(self, text):
        corrected_text = []
        colored_text = []
        all_edits = []

        for sentence in sent_tokenize(text):
            corrected_set = self.gf.correct(sentence, max_candidates=1)
            if corrected_set:
                corrected = list(corrected_set)[0]
                edits = self.gf.get_edits(sentence, corrected)

                if edits:
                    all_edits += [e[0] for e in edits]
                    colored = self._highlight_edits(sentence, edits)
                    colored_text.append(colored)
                    corrected_text.append(corrected)
                else:
                    colored_text.append(sentence)
                    corrected_text.append(sentence)
            else:
                corrected_text.append(sentence)

        return (
            ' '.join(corrected_text),
            ' '.join(colored_text),
            pd.Series([self.grammar_fullforms[e] for e in all_edits]).value_counts()
        )
    
    def _highlight_edits(self, original, edits):
        parts = []
        last_pos = 0
        tokens = original.split()
        
        for edit in edits:
            start, end = edit[2], edit[3]
            parts.extend(tokens[last_pos:start])
            
            if edit[1]:
                parts.append(f'<span style="color:#ff3f33">{self._strikethrough(edit[1])}</span>')
            if edit[4]:
                parts.append(f'<span style="color:#07b81a">{edit[4]}</span>')
            
            last_pos = end
        
        parts.extend(tokens[last_pos:])
        return ' '.join(parts)

class VocabularyAnalyzer:
    def __init__(self, cefr_path):
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self._load_cefr_data(cefr_path)
    
    def _load_cefr_data(self, path):
        cefr_df = pd.read_csv(path)
        self.cefr_mapping = dict(cefr_df[['headword', 'CEFR']].values)
        self.cefr_words = set(cefr_df['headword'])
    
    def analyze_vocabulary(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\d+', '', text)
        words = word_tokenize(text)
        
        cefr_levels = []
        uncategorized = []
        
        for word in words:
            lemma = self._get_cefr_lemma(word)
            level = self.cefr_mapping.get(lemma, 'uncategorized')
            if level == 'uncategorized':
                uncategorized.append(word)
            cefr_levels.append(level)
        
        return (
            pd.Series(cefr_levels).value_counts(),
            uncategorized
        )
    
    def _get_cefr_lemma(self, word):
        lemma = self.lemmatizer.lemmatize(word)
        if lemma in self.cefr_words:
            return lemma
        
        for pos in ['v', 'a', 'r', 's']:
            pos_lemma = self.lemmatizer.lemmatize(word, pos=pos)
            if pos_lemma in self.cefr_words:
                return pos_lemma
        
        return lemma

class WritingFeedback:
    def __init__(self, cefr_csv_path):
        self.nlp = NLPInitializer()
        self.grammar_checker = GrammarChecker()
        self.vocab_analyzer = VocabularyAnalyzer(cefr_csv_path)
    
    def analyze_text(self, text):
        corrected_text, colored_text, grammar_stats = self.grammar_checker.correct_grammar(text)
        vocab_stats, uncategorized = self.vocab_analyzer.analyze_vocabulary(corrected_text)

        return {
            'original': text,
            'corrected': corrected_text,
            'colored_html': colored_text,
            'grammar_errors': grammar_stats.to_dict(),
            'vocab_levels': vocab_stats.to_dict(),
            'uncategorized_words': uncategorized
        }

@stub.function(
    image=image,
    volumes={CEFR_PATH: cefr_volume},
    timeout=600,
    concurrency_limit=1
)
@modal.asgi_app()
def create_app():
    app = Flask(__name__)
    feedback_system = WritingFeedback(CEFR_PATH)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            user_text = request.form.get('text', '').strip()
            if not user_text:
                return render_template('index.html', error="Please enter some text.")

            analysis = feedback_system.analyze_text(user_text)
            return render_template(
                "results.html",
                colored_html=analysis.get('colored_html', ""),
                grammar_errors=analysis.get('grammar_errors', {}),
                vocab_levels=analysis.get('vocab_levels', {}),
                uncategorized_words=analysis.get('uncategorized_words', [])
            )
        return render_template("index.html")

    return app
