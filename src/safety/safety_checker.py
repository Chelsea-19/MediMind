import re

class SafetyChecker:
    def __init__(self):
        # A simple list of red flags. In a real system, this could be backed by a specialized model or a more comprehensive KB.
        self.red_flags = [
            "剧烈头痛", "胸痛", "呼吸困难", "大量出血", "意识模糊", "昏迷", "偏瘫", "心脏骤停", "自杀"
        ]

    def analyze_risk(self, text: str) -> str:
        """
        Analyze the query text to determine the triage risk level.
        Returns: "Emergency", "Urgent", "Routine", or "Self-care / Education"
        """
        for flag in self.red_flags:
            if flag in text:
                return "Emergency"
        
        # Additional urgent heuristics
        if "持续发热" in text or "骨折" in text:
            return "Urgent"

        return "Routine"
        
    def check_unsafe_request(self, text: str) -> bool:
        """
        Check if the request asks for things outside the system's scope, like prescribing medication directly.
        """
        unsafe_keywords = ["开药", "处方", "给我买", "安乐死"]
        for keyword in unsafe_keywords:
            if keyword in text:
                return True
        return False

    def build_safety_context(self, text: str) -> dict:
        risk_level = self.analyze_risk(text)
        is_unsafe = self.check_unsafe_request(text)
        
        return {
            "risk_level": risk_level,
            "is_unsafe": is_unsafe,
            "escalate": risk_level == "Emergency",
            "message": "⚠️ 系统监测到高风险红旗症状，请停止使用本系统并**立即就医**或拨打急救电话！" if risk_level == "Emergency" else None
        }
