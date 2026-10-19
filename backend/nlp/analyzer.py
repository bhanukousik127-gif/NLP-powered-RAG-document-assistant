"""
NLP Document Assistant - Document NLP Analyzer
Performs entity extraction, TF-IDF keyphrase discovery, readability scoring,
extractive summarization, and automated prompt/question generation.
"""

import re
import math
from collections import Counter
from typing import Dict, Any, List

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves", "also", "using", "within", "one", "two", "use", "used"
}


class DocumentAnalyzer:
    """Extracts NLP intelligence and linguistic analytics from ingested documents."""

    @classmethod
    def analyze(cls, text: str, filename: str = "") -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "keywords": [],
                "entities": {},
                "summary": "No text content found to summarize.",
                "readability": {"score": 0, "grade": "N/A", "label": "Empty"},
                "stats": {"words": 0, "sentences": 0, "avg_sentence_len": 0, "lexical_diversity": 0},
                "suggested_questions": []
            }

        sentences = cls._split_sentences(text)
        words = cls._tokenize_words(text)
        
        keywords = cls._extract_keyphrases(text, top_n=10)
        entities = cls._extract_entities(text)
        readability = cls._calculate_readability(sentences, words)
        summary = cls._generate_extractive_summary(sentences, top_k=3)
        suggested_questions = cls._generate_suggested_questions(keywords, entities, filename)

        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = round((unique_words / len(words)) * 100, 1) if words else 0

        return {
            "keywords": keywords,
            "entities": entities,
            "summary": summary,
            "readability": readability,
            "stats": {
                "words": len(words),
                "unique_words": unique_words,
                "sentences": len(sentences),
                "avg_sentence_len": round(len(words) / max(1, len(sentences)), 1),
                "lexical_diversity": lexical_diversity,
            },
            "suggested_questions": suggested_questions
        }

    @classmethod
    def _split_sentences(cls, text: str) -> List[str]:
        raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [s.strip() for s in raw_sents if len(s.strip()) > 15]

    @classmethod
    def _tokenize_words(cls, text: str) -> List[str]:
        return re.findall(r"\b[A-Za-z0-9_-]+\b", text)

    @classmethod
    def _extract_keyphrases(cls, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract high-relevance unigrams and bigrams using TF scoring with stopword filtering."""
        words = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", text)]
        filtered_words = [w for w in words if w not in STOPWORDS]

        # Count unigrams
        unigram_counts = Counter(filtered_words)

        # Count bigrams
        bigrams = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in STOPWORDS and w2 not in STOPWORDS:
                bigrams.append(f"{w1} {w2}")
        bigram_counts = Counter(bigrams)

        # Merge with boost for informative bigrams
        candidates = {}
        for w, c in unigram_counts.items():
            if c > 1:
                candidates[w] = c
        for b, c in bigram_counts.items():
            if c > 1:
                candidates[b] = c * 2.2  # Bigram significance boost

        sorted_phrases = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
        max_score = sorted_phrases[0][1] if sorted_phrases else 1

        return [
            {"phrase": phrase.title(), "count": int(score), "weight": round(score / max_score, 2)}
            for phrase, score in sorted_phrases
        ]

    @classmethod
    def _extract_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extract named entities and domain tokens."""
        entities = {
            "technologies": set(),
            "metrics": set(),
            "organizations": set(),
            "dates_years": set()
        }

        # Dates / Years
        years = re.findall(r"\b(19\d{2}|20\d{2}|Q[1-4]\s*20\d{2})\b", text)
        entities["dates_years"].update(years[:6])

        # Metrics / Quantities
        metrics = re.findall(r"\b\d+(?:\.\d+)?(?:\%|\s*billion|\s*million|\s*GB|\s*MB|\s*TB|\s*ms|\s*kW|\s*MW|\$|\€|\£)\b", text, re.IGNORECASE)
        entities["metrics"].update(metrics[:8])

        # Tech/Model keywords heuristics (CamelCase or Acronyms like RAG, LLM, BERT, API, GPU, CNN, Transformer)
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
        known_bad = {"THE", "AND", "FOR", "NOT", "CAN", "OUT", "ALL", "WAS", "NEW", "ONE", "TOP", "KEY"}
        for ac in acronyms:
            if ac not in known_bad and ac.lower() not in STOPWORDS:
                entities["technologies"].add(ac)

        # Capitalized multi-word noun phrases (Organizations or Concepts)
        org_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text)
        common_false_positives = {"This Section", "In Addition", "On The Other Hand", "For Example", "As Shown"}
        for om in org_matches:
            if om not in common_false_positives and len(om) > 5:
                entities["organizations"].add(om)

        return {k: sorted(list(v))[:8] for k, v in entities.items()}

    @classmethod
    def _calculate_readability(cls, sentences: List[str], words: List[str]) -> Dict[str, Any]:
        """Calculates Flesch Reading Ease and grade level."""
        if not sentences or not words:
            return {"score": 50.0, "grade": "Standard", "label": "Standard (High School)"}

        num_words = len(words)
        num_sentences = max(1, len(sentences))
        
        # Estimate syllables (vowel cluster heuristic)
        syllables = 0
        for w in words:
            w = w.lower()
            syl_count = len(re.findall(r"[aeiouy]+", w))
            if w.endswith("e") and not w.endswith("le") and syl_count > 1:
                syl_count -= 1
            syllables += max(1, syl_count)

        # Flesch-Kincaid formulas
        asl = num_words / num_sentences
        asw = syllables / max(1, num_words)
        fre = 206.835 - (1.015 * asl) - (84.6 * asw)
        score = round(max(0, min(100, fre)), 1)

        if score >= 80:
            label = "Easy (Conversational / Middle School)"
            grade = "Grade 6-7"
        elif score >= 60:
            label = "Standard (Plain English / High School)"
            grade = "Grade 8-10"
        elif score >= 40:
            label = "Technical (College Undergraduate)"
            grade = "College"
        else:
            label = "Academic / Highly Complex (Professional / Graduate)"
            grade = "Graduate"

        return {"score": score, "grade": grade, "label": label}

    @classmethod
    def _generate_extractive_summary(cls, sentences: List[str], top_k: int = 3) -> str:
        """TextRank-style sentence importance scoring based on keyword overlap."""
        if not sentences:
            return ""
        if len(sentences) <= top_k:
            return " ".join(sentences)

        # Build vocabulary frequency
        word_freq = Counter()
        for s in sentences:
            tokens = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", s) if w.lower() not in STOPWORDS]
            word_freq.update(tokens)

        # Score sentences
        scores = []
        for idx, s in enumerate(sentences):
            tokens = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", s) if w.lower() not in STOPWORDS]
            if not tokens:
                continue
            # Salience: sum of word frequencies normalized by length with lead bias
            salience = sum(word_freq[t] for t in tokens) / (math.sqrt(len(tokens)) + 1)
            # Give slight boost to early introductory sentences
            position_multiplier = 1.25 if idx < 3 else 1.0
            scores.append((idx, salience * position_multiplier, s))

        # Pick top sentences and sort by original narrative appearance
        top_selected = sorted(sorted(scores, key=lambda x: x[1], reverse=True)[:top_k], key=lambda x: x[0])
        return " ".join(item[2] for item in top_selected)

    @classmethod
    def _generate_suggested_questions(
        cls, 
        keywords: List[Dict[str, Any]], 
        entities: Dict[str, List[str]], 
        filename: str = ""
    ) -> List[str]:
        """Generates dynamic, clickable discovery questions for RAG testing."""
        questions = []
        
        phrases = [k["phrase"] for k in keywords[:4]]
        techs = entities.get("technologies", [])[:3]
        metrics = entities.get("metrics", [])[:2]

        if phrases:
            questions.append(f"What is the main finding regarding {phrases[0]}?")
            if len(phrases) > 1:
                questions.append(f"How does the document explain the role of {phrases[1]}?")
        if techs:
            questions.append(f"What advantages and architecture details are mentioned for {techs[0]}?")
        if metrics:
            questions.append(f"What key metrics or quantitative results (e.g. {metrics[0]}) are reported?")
        if len(questions) < 4:
            questions.append("What are the primary conclusions and recommendations outlined in the document?")

        return questions[:4]
