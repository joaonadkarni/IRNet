from typing import NamedTuple, Dict, List


class ModelInput(NamedTuple):
    question: List[Dict]
    tables: List[Dict]


class ServerResponse(NamedTuple):
    sql: str
    out_ents: List[str]
