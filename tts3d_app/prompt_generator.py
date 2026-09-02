"""语音提示词生成器 - 基于 Ollama 本地 LLM"""
from __future__ import annotations

import ollama
from typing import Optional


# Ollama 配置 - 使用 gemma4 2B 模型
DEFAULT_MODEL = "gemma4:2b"
OLLAMA_HOST = "http://localhost:11434"


class PromptGenerator:
    """根据中文描述生成 TTS 英文提示词"""

    SYSTEM_PROMPT = """You are a TTS prompt generator. Translate the Chinese description to a concise English TTS prompt.
IMPORTANT: Output ONLY the English prompt text, nothing else. No explanations, no thinking, no analysis.
Examples:
Input: 温柔女声
Output: A warm, gentle female voice

Input: 活泼可爱的儿童声音
Output: A lively, cute children's voice

Input: 磁性的中年男声
Output: A deep, magnetic middle-aged male voice"""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = ollama.Client(host=OLLAMA_HOST)
        return self._client

    def generate(self, description: str) -> str:
        """
        根据中文描述生成英文提示词

        Args:
            description: 中文声音描述，如"温柔的客服女声"

        Returns:
            英文提示词
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": description}
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 300,
                }
            )
            raw_content = response["message"]["content"]
            # Gemma 4 outputs thinking process + final answer; extract answer after "...done thinking."
            if "...done thinking." in raw_content:
                answer = raw_content.split("...done thinking.")[-1].strip()
            else:
                answer = raw_content.strip()
            return answer
        except Exception as e:
            raise PromptGenerationError(f"生成提示词失败: {e}") from e

    def test_connection(self) -> bool:
        """测试 Ollama 连接"""
        try:
            self.client.list()
            return True
        except Exception:
            return False


class PromptGenerationError(Exception):
    """提示词生成错误"""
    pass


# 全局实例
_generator: Optional[PromptGenerator] = None


def get_generator(model: str = DEFAULT_MODEL) -> PromptGenerator:
    """获取生成器实例（单例）"""
    global _generator
    if _generator is None or _generator.model != model:
        _generator = PromptGenerator(model)
    return _generator


def generate_voice_prompt(description: str, model: str = DEFAULT_MODEL) -> str:
    """快捷函数：根据描述生成提示词"""
    gen = get_generator(model)
    return gen.generate(description)
