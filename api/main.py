import json
from pathlib import Path
from typing import TypedDict

from langchain.agents import create_agent
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
GLOSSARY_FILE = BASE_DIR / "data" / "glossary.json"


class GlossaryEntry(TypedDict):
    id: str
    term: str
    definition: str


def load_glossary() -> dict[str, GlossaryEntry]:
    """Load and validate the glossary entries used by the lookup tool."""
    try:
        with GLOSSARY_FILE.open(encoding="utf-8") as file:
            document = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to load glossary from {GLOSSARY_FILE}") from error

    if isinstance(document, dict) and isinstance(document.get("terms"), list):
        entries = document["terms"]
    elif isinstance(document, dict) and all(
        isinstance(term, str) and isinstance(definition, str)
        for term, definition in document.items()
    ):
        entries = [
            {"id": term, "term": term, "definition": definition}
            for term, definition in document.items()
        ]
    else:
        raise ValueError(
            "Glossary must contain a 'terms' list or a term-to-definition mapping"
        )

    glossary: dict[str, GlossaryEntry] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term")
        definition = entry.get("definition")
        entry_id = entry.get("id")
        if isinstance(term, str) and isinstance(definition, str) and isinstance(entry_id, str):
            glossary[term.casefold()] = {
                "id": entry_id,
                "term": term,
                "definition": definition,
            }
    return glossary


GLOSSARY = load_glossary()


class TermExplanation(BaseModel):
    term: str = Field(description="The AI term being explained")
    simple: str = Field(description="A plain-language explanation")
    mental_model: str = Field(description="An intuitive analogy or mental model")
    example: str = Field(description="A concrete example")
    quiz: str = Field(description="One short question to test understanding")
    from_glossary: bool = Field(
        description="Whether the explanation is based on an entry in the glossary"
    )


def lookup_term(term: str) -> str:
    """Look up an AI term in the local glossary by name or ID."""
    query = term.strip().casefold()
    if not query:
        return "Please provide an AI term to look up."

    entry = GLOSSARY.get(query)
    if entry is None:
        entry = next(
            (
                candidate
                for candidate in GLOSSARY.values()
                if query in candidate["term"].casefold()
                or query in candidate["id"].casefold()
            ),
            None,
        )

    if entry is None:
        return f"No glossary entry found for '{term}'."
    return f"{entry['term']}: {entry['definition']}"


SYSTEM = """You explain artificial-intelligence concepts clearly for learners.
Use the lookup_term tool when the requested concept may be in the glossary.
Return a concise explanation with a simple definition, mental model, example,
and one quiz question. Set from_glossary to true only when the glossary
provided the term's definition. If the term is not in the glossary, explain
that it is not in the glossary but still answer from your general knowledge.
"""


def create_explanation_agent():
    """Create the explanation agent without contacting the model at import time."""
    return create_agent(
        model="ollama:gemma4:31b-cloud",
        tools=[lookup_term],
        system_prompt=SYSTEM,
        response_format=TermExplanation,
    )


def explain(term: str) -> TermExplanation:
    """Generate a structured explanation for an AI term."""
    result = create_explanation_agent().invoke(
        {"messages": [{"role": "user", "content": term}]}
    )
    return result["structured_response"]


def format_explanation(answer: TermExplanation) -> str:
    """Convert the structured explanation into readable sentences."""
    glossary_note = (
        "This explanation is based on the local glossary."
        if answer.from_glossary
        else "This term was not found in the local glossary."
    )
    return (
        f"{answer.term} means {answer.simple} {glossary_note}\n\n"
        f"You can think of it like this: {answer.mental_model}\n\n"
        f"For example, {answer.example}\n\n"
        f"Check your understanding: {answer.quiz}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Explain AI glossary terms.")
    parser.add_argument(
        "term",
        nargs="?",
        help="The AI term or question to explain",
    )
    args = parser.parse_args()

    term = args.term
    if not term:
        try:
            term = input("What AI term would you like explained? ").strip()
        except EOFError:
            parser.error("provide a term as an argument or enter one when prompted")
    if not term:
        parser.error("the term cannot be empty")

    answer = explain(term)
    print(format_explanation(answer))


if __name__ == "__main__":
    main()
