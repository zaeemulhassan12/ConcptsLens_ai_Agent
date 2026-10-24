from pathlib import Path
from langchain.agents import create_agent
from pydantic import BaseModel

BASE_DIR = Path(__file__)
GLOSSARY_FILE = BASE_DIR.parent / "glossary.txt"

def load_glossary() -> str:
    pass


GLOSSARY = load_glossary()

class termExplainations(BaseModel):

#AGENT tool

def lookup_term():
    pass
agent = create_agent(
    model="ollama:gemma4:31b-cloud",
    tools=[lookup_term],
    system_prompt=
