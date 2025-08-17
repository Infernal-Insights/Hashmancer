#!/usr/bin/env python3
"""
Basic Opus client for automated code analysis
"""

import anthropic
import os

class OpusClient:
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def analyze_security_issues(self, file_contents: dict):
        """Analyze files for security issues"""
        # Basic implementation - see OPUS_INTEGRATION_WORKFLOW.md for full version
        prompt = "Analyze these files for security vulnerabilities:\n\n"
        for filepath, content in file_contents.items():
            prompt += f"\n## {filepath}\n```python\n{content}\n```\n"
        
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content

if __name__ == "__main__":
    print("Opus client ready - see OPUS_INTEGRATION_WORKFLOW.md for full setup")
