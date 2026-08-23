class StudyAudioClassifier:

    def __init__(self):

        self.study_keywords = [
            "study", "studying", "learn", "learning", "lesson",
            "chapter", "lecture", "class", "course", "teacher",
            "student", "university", "college", "school",
            "assignment", "homework", "exam", "test", "quiz",
            "presentation", "project", "research", "report",
            "thesis", "paper", "notes", "book", "reading",
            "prepare", "practice", "script",

            "python", "java", "c++", "html", "css", "javascript",
            "code", "coding", "programming", "function", "variable",
            "loop", "array", "database", "sql", "algorithm",
            "data", "model", "machine learning", "artificial intelligence",
            "ai", "training", "dataset", "accuracy", "classification",

            "math", "calculation", "equation", "formula",
            "physics", "chemistry", "biology", "english",
            "grammar", "paragraph", "definition", "explanation",
            "solve", "solution", "question", "answer"
        ]

        self.non_study_keywords = [
            "movie", "film", "song", "music", "game", "gaming",
            "facebook", "instagram", "tiktok", "youtube",
            "shopping", "food", "restaurant", "party",
            "gossip", "chat", "call", "friend", "friends",
            "football", "cricket", "drama", "series",
            "anime", "cartoon", "funny", "joke",
            "sleep", "nap", "travel", "trip"
        ]


    def classify(self, text):

        text = text.lower()

        study_count = 0
        non_study_count = 0

        for word in self.study_keywords:
            if word in text:
                study_count += 1

        for word in self.non_study_keywords:
            if word in text:
                non_study_count += 1

        if study_count == 0 and non_study_count == 0:
            study_probability = 50
            label = "Uncertain Audio"

        elif study_count > non_study_count:
            study_probability = int(
                (study_count / (study_count + non_study_count)) * 100
            )
            label = "Study Related Audio"

        elif non_study_count > study_count:
            study_probability = int(
                (study_count / (study_count + non_study_count)) * 100
            )
            label = "Non-Study Conversation"

        else:
            study_probability = 50
            label = "Uncertain Audio"

        return {
            "text": text,
            "study_keywords": study_count,
            "non_study_keywords": non_study_count,
            "study_probability": study_probability,
            "label": label
        }