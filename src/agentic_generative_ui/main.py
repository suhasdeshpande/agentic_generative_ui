#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

from crewai.flow import Flow, start

from crewai import LLM
from pprint import pprint
import logging
import json

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

llm = LLM(
    model="gpt-4o"
)

class AgenticGenerativeUIFlow(Flow):

    @start()
    def chat(self):
        logger.info(f"Initial input state: {self.state}")
        logger.info(f"Raw state dump: {json.dumps(self.state, indent=2)}")
        logger.info(f"Messages in state: {self.state.get('messages', [])}")

        logger.info(f"####STATE####\n{json.dumps(self.state, indent=2)}")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

        if self.state.get('messages'):
            messages.append(self.state['messages'][-1])

        response = llm.call(messages)

        if 'messages' not in self.state:
            self.state['messages'] = []

        self.state['messages'].append({
            "role": "assistant",
            "content": response
        })
        return response

    def __repr__(self):
        pprint(vars(self), width=120, depth=3)

def kickoff():
    agentic_generative_ui_flow = AgenticGenerativeUIFlow()
    agentic_generative_ui_flow.kickoff()


if __name__ == "__main__":
    kickoff()
