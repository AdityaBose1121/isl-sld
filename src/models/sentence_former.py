"""
Sentence Former — Converts recognized ISL glosses + facial emotion into natural English sentences.

ISL follows SOV (Subject-Object-Verb) order whereas English uses SVO.
This module handles the grammatical conversion and enriches output with emotional context.

Two modes:
    1. Rule-based (offline): Template-based ISL→English conversion
    2. Gemini API (online): Uses Google Gemini for complex sentence formation
"""

import os
import re
from collections import Counter

# ISL grammar rules for SOV → SVO conversion
ISL_GRAMMAR_RULES = {
    # Question markers (facial expression: raised eyebrows)
    "question_words": {"what", "where", "when", "how", "who", "why", "which"},

    # Negation (facial expression: head shake / furrowed brows)
    "negation_words": {"no", "not", "never", "nothing", "none"},

    # Time markers (typically come first in ISL)
    "time_words": {
        "today", "tomorrow", "yesterday", "morning", "evening",
        "night", "now", "later", "before", "after", "always"
    },

    # Pronouns
    "pronouns": {"i", "you", "he", "she", "we", "they", "it"},

    # Common ISL verb forms
    "verbs": {
        "go", "come", "eat", "drink", "sleep", "work", "study",
        "play", "help", "want", "need", "like", "love", "know",
        "understand", "see", "hear", "feel", "think", "say", "tell",
        "give", "take", "make", "do", "have", "be"
    },
}

# Emotion → sentence modifiers
EMOTION_MODIFIERS = {
    "happy": {
        "prefix": "",
        "suffix": "",
        "adjectives": ["happy", "glad", "joyful", "pleased"],
        "tone": "positive",
    },
    "sad": {
        "prefix": "Unfortunately, ",
        "suffix": "",
        "adjectives": ["sad", "upset", "unhappy"],
        "tone": "negative",
    },
    "angry": {
        "prefix": "",
        "suffix": "",
        "adjectives": ["angry", "frustrated", "annoyed"],
        "tone": "negative",
    },
    "fear": {
        "prefix": "",
        "suffix": "",
        "adjectives": ["scared", "afraid", "worried"],
        "tone": "negative",
    },
    "surprise": {
        "prefix": "",
        "suffix": "!",
        "adjectives": ["surprised", "amazed", "shocked"],
        "tone": "neutral",
    },
    "disgust": {
        "prefix": "",
        "suffix": "",
        "adjectives": ["disgusted", "repulsed"],
        "tone": "negative",
    },
    "neutral": {
        "prefix": "",
        "suffix": "",
        "adjectives": [],
        "tone": "neutral",
    },
}


class SentenceFormer:
    """
    Converts ISL glosses and detected emotions into natural English sentences.
    """

    def __init__(self, use_gemini=False, gemini_api_key=None):
        self.use_gemini = use_gemini
        self.gemini_model = None

        if use_gemini and gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                print("Gemini API initialized for sentence formation")
            except Exception as e:
                print(f"Failed to initialize Gemini API: {e}")
                self.use_gemini = False

    def form_sentence(self, glosses, emotion="neutral", emotion_confidence=0.5):
        """
        Convert a sequence of ISL glosses + emotion into a natural English sentence.

        Args:
            glosses: list of recognized sign glosses, e.g. ["i", "school", "go"]
            emotion: detected facial emotion string
            emotion_confidence: confidence of emotion detection

        Returns:
            sentence: natural English sentence string
            metadata: dict with processing details
        """
        if not glosses:
            return "", {"method": "empty", "raw_glosses": []}

        # Clean and normalize glosses
        cleaned = [g.lower().strip().replace("_", " ") for g in glosses]

        # Remove duplicates (repeated signs)
        deduped = self._remove_consecutive_duplicates(cleaned)

        metadata = {
            "raw_glosses": glosses,
            "cleaned_glosses": deduped,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
        }

        # Try Gemini first if available
        if self.use_gemini and self.gemini_model and len(deduped) >= 2:
            try:
                sentence = self._gemini_form_sentence(deduped, emotion)
                metadata["method"] = "gemini"
                return sentence, metadata
            except Exception as e:
                metadata["gemini_error"] = str(e)

        # Fallback to rule-based
        sentence = self._rule_based_form_sentence(deduped, emotion, emotion_confidence)
        metadata["method"] = "rule_based"
        return sentence, metadata

    def _remove_consecutive_duplicates(self, glosses):
        """Remove consecutive duplicate glosses (repeated signs)."""
        if not glosses:
            return glosses
        result = [glosses[0]]
        for g in glosses[1:]:
            if g != result[-1]:
                result.append(g)
        return result

    def _get_be_verb(self, subject_word):
        """Return the correct form of 'to be' for a given subject."""
        if subject_word.lower() in ("i",):
            return "am"
        elif subject_word.lower() in ("he", "she", "it"):
            return "is"
        elif subject_word.lower() in ("you", "we", "they"):
            return "are"
        return "is"

    def _rule_based_form_sentence(self, glosses, emotion, emotion_confidence):
        """
        Rule-based ISL to English conversion.

        ISL grammar: TIME + SUBJECT + OBJECT + VERB (SOV)
        English:     SUBJECT + VERB + OBJECT (SVO)
        """
        if len(glosses) == 1:
            return self._single_gloss_sentence(glosses[0], emotion, emotion_confidence)

        # Categorize glosses
        time_words = []
        subject = []
        objects = []
        verbs = []
        question_words = []
        negation = []

        for gloss in glosses:
            if gloss in ISL_GRAMMAR_RULES["time_words"]:
                time_words.append(gloss)
            elif gloss in ISL_GRAMMAR_RULES["pronouns"]:
                subject.append(gloss)
            elif gloss in ISL_GRAMMAR_RULES["verbs"]:
                verbs.append(gloss)
            elif gloss in ISL_GRAMMAR_RULES["question_words"]:
                question_words.append(gloss)
            elif gloss in ISL_GRAMMAR_RULES["negation_words"]:
                negation.append(gloss)
            else:
                objects.append(gloss)

        # Determine the main subject for verb agreement
        if not subject:
            subject = ["I"]  # Default subject in ISL
        main_subject = subject[0]
        be_verb = self._get_be_verb(main_subject)

        # Build sentence in English SVO order
        sentence_parts = []

        # 1. Time marker (comma-separated prefix)
        time_prefix = ""
        if time_words:
            time_prefix = " ".join(w.capitalize() for w in time_words) + ", "

        # 2. Subject
        subject_str = " ".join(s if s == "I" else s for s in subject)

        # 3. Determine if emotion should be the main predicate or a modifier
        emotion_clause = ""
        has_strong_emotion = (emotion_confidence > 0.6 and emotion != "neutral")

        if has_strong_emotion:
            modifier = EMOTION_MODIFIERS.get(emotion, {})
            adj_list = modifier.get("adjectives", [])
            if adj_list:
                if verbs or objects:
                    # Emotion as a feeling prefix: "I am happy to go to school"
                    emotion_clause = f"{be_verb} feeling {adj_list[0]}"
                else:
                    # Emotion is the main content
                    emotion_clause = f"{be_verb} {adj_list[0]}"

        # Movement verbs take 'to' before the object; action verbs take it directly
        MOVEMENT_VERBS = {"go", "come", "travel", "walk", "run"}
        PREPOSITION_VERBS = {"study", "work", "play"}  # also use 'at'

        # 4. Build the core sentence
        verb_phrase = " ".join(verbs)
        object_phrase = " ".join(objects)

        # Determine the right preposition for verb+object
        def verb_object_phrase(vp, op):
            if not vp:
                return op
            if not op:
                return vp
            first_verb = verbs[0] if verbs else ""
            if first_verb in MOVEMENT_VERBS or first_verb in PREPOSITION_VERBS:
                return f"{vp} to {op}"
            return f"{vp} {op}"

        if emotion_clause and (verbs or objects):
            # "I am feeling happy and go to school" / "I am feeling sad and eat food"
            core = f"{subject_str} {emotion_clause}"
            if verb_phrase and object_phrase:
                core += f" and {verb_object_phrase(verb_phrase, object_phrase)}"
            elif verb_phrase:
                core += f" and want to {verb_phrase}"
            elif object_phrase:
                core += f" about {object_phrase}"
        elif emotion_clause:
            # Just emotion: "I am happy"
            core = f"{subject_str} {emotion_clause}"
        else:
            # Standard SVO
            if verb_phrase and object_phrase:
                core = f"{subject_str} {verb_object_phrase(verb_phrase, object_phrase)}"
            elif verb_phrase:
                core = f"{subject_str} {verb_phrase}"
            elif object_phrase:
                core = f"{subject_str} want {object_phrase}"
            else:
                core = subject_str

        # 5. Add negation
        if negation:
            # Insert "not" or "don't" appropriately
            if be_verb in core:
                core = core.replace(be_verb, f"{be_verb} not", 1)
            else:
                core = core.replace(subject_str + " ", f"{subject_str} do not ", 1)

        # 6. Handle question words
        if question_words:
            q = question_words[0]
            # Restructure as a question: "What is your name?"
            if main_subject.lower() == "you":
                # "you name" -> "What is your name?"
                core = f"{q} is your {object_phrase}" if object_phrase else f"{q} do you {verb_phrase}"
            else:
                core = f"{q} do {subject_str} {verb_phrase} {object_phrase}".strip()

        # 7. Assemble final sentence
        sentence = time_prefix + core

        # Apply emotion prefix/suffix
        modifier = EMOTION_MODIFIERS.get(emotion, EMOTION_MODIFIERS["neutral"])
        sentence = modifier["prefix"] + sentence + modifier["suffix"]

        # Capitalize and punctuate
        sentence = self._finalize_sentence(sentence, glosses, emotion)

        return sentence

    def _single_gloss_sentence(self, gloss, emotion, confidence):
        """Handle single-gloss input."""
        # Common single-word responses
        single_word_sentences = {
            "hello": "Hello!",
            "thank_you": "Thank you!",
            "thanks": "Thank you!",
            "please": "Please.",
            "sorry": "I'm sorry.",
            "help": "I need help!",
            "yes": "Yes.",
            "no": "No.",
            "water": "I want water.",
            "food": "I want food.",
            "pain": "I'm in pain!",
            "happy": "I'm happy!",
            "sad": "I'm sad.",
            "good": "That's good!",
            "bad": "That's bad.",
        }

        if gloss in single_word_sentences:
            return single_word_sentences[gloss]

        # Add emotion context
        if emotion != "neutral" and confidence > 0.6:
            adj = EMOTION_MODIFIERS.get(emotion, {}).get("adjectives", [])
            if adj:
                return f"I feel {adj[0]} about {gloss}."

        return gloss.capitalize() + "."

    def _finalize_sentence(self, sentence, glosses, emotion):
        """Capitalize first letter, add proper punctuation."""
        sentence = sentence.strip()
        if not sentence:
            return ""

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]

        # Fix "i" → "I"
        sentence = re.sub(r'\bi\b', 'I', sentence)

        # Add punctuation
        has_question = any(g in ISL_GRAMMAR_RULES["question_words"] for g in glosses)
        if has_question and not sentence.endswith('?'):
            sentence = sentence.rstrip('.!') + '?'
        elif emotion == "surprise" and not sentence.endswith('!'):
            sentence = sentence.rstrip('.') + '!'
        elif not sentence[-1] in '.!?':
            sentence += '.'

        return sentence

    def _gemini_form_sentence(self, glosses, emotion):
        """Use Gemini API for advanced sentence formation."""
        prompt = f"""You are an Indian Sign Language (ISL) to English translator.

Convert the following sequence of ISL glosses into a natural, grammatically correct English sentence.

ISL Glosses (in order): {' '.join(glosses)}
Detected facial emotion: {emotion}

Rules:
1. ISL uses SOV (Subject-Object-Verb) order. Convert to English SVO order.
2. The facial emotion provides context for the tone of the sentence.
3. If the emotion is strong (not neutral), incorporate it naturally into the meaning.
4. Keep the sentence concise and natural.
5. Return ONLY the English sentence, nothing else.

English sentence:"""

        response = self.gemini_model.generate_content(prompt)
        sentence = response.text.strip()

        # Clean up any quotes or extra formatting
        sentence = sentence.strip('"\'')
        if not sentence[-1] in '.!?':
            sentence += '.'

        return sentence


def build_sentence_former():
    """Factory function to create SentenceFormer with config."""
    from src.utils.config import GEMINI_API_KEY, USE_GEMINI
    return SentenceFormer(use_gemini=USE_GEMINI, gemini_api_key=GEMINI_API_KEY)
