"""
RAG MVP
INTENT TRACKER

Responsible for

1. Loading the trained Logistic Regression model
2. Predicting intents
3. Tracking intent changes
4. Returning intent history
"""

import re
import joblib


MODEL_PATH = "models/intent_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


class IntentTracker:

    def __init__(self):

        print("Loading Intent Model...")

        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

        self.history = []

        print("Intent Model Loaded.")

    ####################################################################
    # CLEAN
    ####################################################################

    def clean(self, text):

        text = text.lower()

        text = re.sub(r"[^a-zA-Z ]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    ####################################################################
    # PREDICT ONE SEGMENT
    ####################################################################

    def predict(self, text):

        cleaned = self.clean(text)

        vector = self.vectorizer.transform([cleaned])

        intent = self.model.predict(vector)[0]

        confidence = float(self.model.predict_proba(vector).max())

        return intent, confidence

    ####################################################################
    # TRACK ENTIRE CONVERSATION
    ####################################################################

    def track(self, segments):

        self.history = []

        current_intent = None

        for segment in segments:

            text = segment["text"]

            if len(text.split()) < 3:
                continue

            intent, confidence = self.predict(text)

            if current_intent != intent:

                current_intent = intent

                self.history.append({

                    "intent": intent,

                    "confidence": confidence,

                    "segment": text

                })

        return self.history

    ####################################################################
    # CURRENT ACTIVE INTENT
    ####################################################################

    def current_intent(self):

        if len(self.history) == 0:
            return None

        return self.history[-1]



########################################################################
# TEST
########################################################################

if __name__ == "__main__":

    tracker = IntentTracker()

    segments = [

        {
            "original":"Hello"
        },

        {
            "original":"I want to claim for my washing machine."
        },

        {
            "original":"The repair has taken too long."
        },

        {
            "original":"Can I get a refund instead?"
        }

    ]

    history = tracker.track(segments)

    print()

    print("="*70)

    print("INTENT HISTORY")

    print("="*70)

    for item in history:

        print()

        print("Intent      :", item["intent"])

        print("Confidence  :", round(item["confidence"],3))

    print()

    print("="*70)

    print("CURRENT")

    print("="*70)

    print(tracker.current_intent())