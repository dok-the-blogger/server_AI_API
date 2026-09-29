"""Jev's typed decisions; caller text never grants authority to perform an action."""

import json
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


def _text(value: str) -> str:
    if not value.strip():
        raise ValueError("Blank text")
    value.encode("utf-8")
    return value


Name = Annotated[str, Field(min_length=1, max_length=128), AfterValidator(_text)]
Description = Annotated[str, Field(min_length=1, max_length=4096), AfterValidator(_text)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
JevModel = Literal["typesafe-jev-1.13.0"]


class DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class ChoiceQuestion(DecisionModel):
    type: Literal["choice"]
    instructions: Description
    criteria: dict[Name, Description] = Field(min_length=2, max_length=32)


class ScoreQuestion(DecisionModel):
    type: Literal["score"]
    instructions: Description
    criteria: list[Description] = Field(min_length=2, max_length=32)


class NoulQuestion(DecisionModel):
    type: Literal["noul"]
    instructions: Description


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class SystemOneRequest(DecisionModel):
    model: JevModel = "typesafe-jev-1.13.0"
    state: str | dict[str, JsonValue] | list[str]
    questions: dict[Name, Question] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def bounded_payload(self) -> "SystemOneRequest":
        if isinstance(self.state, str):
            _text(self.state)
        elif isinstance(self.state, list):
            if not self.state:
                raise ValueError("Empty state")
            for text in self.state:
                _text(text)
        try:
            payload = json.dumps(
                self.model_dump(), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError("Invalid decision data") from None
        if len(payload) > 65536:
            raise ValueError("Decision request exceeds 64 KiB")
        return self


class ChoiceAnswer(DecisionModel):
    type: Literal["choice"]
    choice: Name
    probabilities: dict[Name, Probability] = Field(min_length=2, max_length=32)
    confidence: Probability


class ScoreAnswer(DecisionModel):
    type: Literal["score"]
    score: float = Field(ge=0, le=31, allow_inf_nan=False)
    legend: dict[str, Description] = Field(min_length=2, max_length=32)
    probabilities: dict[str, Probability] = Field(min_length=2, max_length=32)
    confidence: Probability


class NoulAnswer(DecisionModel):
    type: Literal["noul"]
    noul: Probability


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class DecisionUsage(DecisionModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class SystemOneResponse(DecisionModel):
    model: JevModel
    provider: Literal["digitalocean"]
    answers: dict[Name, Answer] = Field(min_length=1, max_length=16)
    usage: DecisionUsage
    elapsed_ms: int = Field(ge=0)

    def validate_for(self, request: SystemOneRequest) -> None:
        if self.model != request.model or self.answers.keys() != request.questions.keys():
            raise ValueError("Unexpected decision profile")
        for name, question in request.questions.items():
            answer = self.answers[name]
            if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
                if (
                    answer.probabilities.keys() != question.criteria.keys()
                    or answer.choice not in question.criteria
                ):
                    raise ValueError("Unexpected choice")
            elif isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
                legend = {str(i): level for i, level in enumerate(question.criteria)}
                if (
                    answer.legend != legend
                    or answer.probabilities.keys() != legend.keys()
                    or answer.score > len(legend) - 1
                ):
                    raise ValueError("Unexpected score scale")
            elif not (isinstance(question, NoulQuestion) and isinstance(answer, NoulAnswer)):
                raise ValueError("Unexpected answer type")
            if isinstance(answer, (ChoiceAnswer, ScoreAnswer)) and abs(
                sum(answer.probabilities.values()) - 1
            ) > 0.01:
                raise ValueError("Invalid probability distribution")
