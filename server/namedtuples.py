from typing import NamedTuple, Dict, List


class ServerModelInput(NamedTuple):
    question: List[Dict]
    tables: List[Dict]


class ServerResponse(NamedTuple):
    sql: str
    out_attrs: List[str]
    input_params: List[Dict[str, str]]
