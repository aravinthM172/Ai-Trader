import os
import json
from dotenv import load_dotenv

load_dotenv()

class ClaudeAnalyzer:

    def __init__(self):

        self.api_key = os.getenv("ANTHROPIC_API_KEY")

        self.client = None

        if self.api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(
                    api_key=self.api_key
                )
            except Exception as e:
                print("Claude initialization error:", e)

    def analyze(self, market):

        if self.client is None:

            return {
                "decision": "HOLD",
                "confidence": 0,
                "reason": "Claude API key not configured."
            }

        prompt = f"""
You are a market-analysis component for an XAUUSD
algorithmic trading system.

Analyze the following market state.

IMPORTANT:
Return HOLD when the setup is unclear.
Do not invent missing information.
Do not guarantee profitability.

Market data:

{json.dumps(market, indent=2)}

Return ONLY valid JSON:

{{
  "decision": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0,
  "reason": "short explanation",
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}}
"""

        try:

            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            text = response.content[0].text

            return json.loads(text)

        except Exception as e:

            return {
                "decision": "HOLD",
                "confidence": 0,
                "reason": f"Claude error: {e}",
                "risk_level": "HIGH"
            }
