from typing import List

from pydantic import BaseModel, Field

"""schema of the source of the agent response """


class Source(BaseModel):
    url: str = Field("Url of the source of the agent response")


class AgentResponse(BaseModel):
    answer: str = Field("Answer of the agent to the query")
    sources: List[Source] = Field("sources used by agent to answer the query")
