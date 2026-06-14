"""
Text Emotion Extraction Module

Uses:
- Groq LLM for transcript cleaning
- DistilRoBERTa for emotion classification

Output:
7-dimensional emotion probability vector
"""

from transformers import pipeline


class TextEmotionExtractor:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )

    def clean_text(self, text):
        """
        Placeholder for transcript cleaning.
        """
        return text.strip()

    def extract_emotions(self, text):
        text = self.clean_text(text)
        emotions = self.classifier(text)
        return emotions


if __name__ == "__main__":
    extractor = TextEmotionExtractor()
    sample = "I feel tired and emotionally exhausted."
    print(extractor.extract_emotions(sample))
