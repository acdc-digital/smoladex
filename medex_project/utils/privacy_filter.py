import re

class PrivacyFilter:
    privacyKeywords = [
        "name",
        "address",
        "phone",
        "email",
        "birthdate",
        "social security",
        "passport",
        "driver's license",
        "insurance",
    ]

    def apply_filter(self, text):
        for keyword in self.privacyKeywords:
            pattern = re.compile(r'\b' + keyword + r'\b', re.IGNORECASE)
            text = pattern.sub("[REDACTED]", text)
        return text