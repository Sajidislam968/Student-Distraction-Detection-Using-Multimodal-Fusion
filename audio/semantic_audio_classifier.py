from transformers import pipeline


class SemanticAudioClassifier:

    def __init__(self):

        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        self.labels = [
            "study-related academic speech",
            "non-study casual conversation",
            "uncertain or unrelated speech"
        ]


    def classify(self, text):

        if text is None or text.strip() == "":
            return {
                "text": "",
                "label": "Uncertain Audio",
                "study_probability": 50,
                "confidence": 0
            }

        result = self.classifier(
            text,
            candidate_labels=self.labels
        )

        top_label = result["labels"][0]
        confidence = round(result["scores"][0] * 100, 2)

        study_score = 0

        for label, score in zip(result["labels"], result["scores"]):
            if label == "study-related academic speech":
                study_score = round(score * 100, 2)


        if top_label == "study-related academic speech":
            final_label = "Study Related Audio"

        elif top_label == "non-study casual conversation":
            final_label = "Non-Study Conversation"

        else:
            final_label = "Uncertain Audio"


        return {
            "text": text,
            "label": final_label,
            "study_probability": study_score,
            "confidence": confidence,
            "raw_label": top_label
        }