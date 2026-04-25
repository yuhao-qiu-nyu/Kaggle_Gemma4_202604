"""
LLM coaching layer — calls Gemma 4 via Google AI Studio (Gemini API).

Swap this file to use Ollama, OpenAI, or any other provider.
"""

import json
import os
from typing import Any, Dict, List, Optional

from google import genai

SYSTEM_PROMPT = """
你是一名友好、耐心、鼓励式的手语学习教练。

你的任务是根据手语识别模型的结果，为学习者生成简短、清晰、有帮助的反馈。

请始终输出以下四部分：
1. 识别总结
2. 纠错或解释
3. 鼓励反馈
4. 下一步练习建议

要求：
- 语言简洁
- 语气温和
- 有教学感
- 不要使用技术术语
- 如果置信度低，要明确说明模型可能不确定
- 如果 top-k 里有容易混淆的词，要顺带指出
""".strip()


def _build_user_prompt(payload: dict) -> str:
    return (
        "下面是手语识别模型对用户一次手势尝试的识别结果：\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "请你基于这些结果，生成一段适合学习者阅读的反馈，包含：\n"
        "- 识别总结\n"
        "- 纠错或解释\n"
        "- 鼓励反馈\n"
        "- 下一步练习建议"
    )


def build_gemma_payload(
    pred_result: dict,
    user_goal: Optional[str] = None,
    history_errors: Optional[List[str]] = None,
) -> dict:
    return {
        "predicted_label": pred_result["predicted_label"],
        "predicted_prob": pred_result["predicted_prob"],
        "confidence": pred_result["confidence"],
        "topk_predictions": pred_result["topk_predictions"],
        "user_goal": user_goal or "practice sign language",
        "history_errors": history_errors or [],
    }


class CoachLLM:
    """Thin wrapper around the Gemini / Gemma 4 API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemma-4-31b-it"):
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=key)
        self.model = model

    def get_coach_feedback(
        self,
        pred_result: dict,
        user_goal: Optional[str] = None,
        history_errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build the prompt from pred_result, call Gemma 4, return
        {"coach_feedback": str, "gemma_payload": dict}.
        """
        payload = build_gemma_payload(pred_result, user_goal, history_errors)
        user_prompt = _build_user_prompt(payload)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]},
            ],
        )

        return {
            "coach_feedback": response.text,
            "gemma_payload": payload,
        }
