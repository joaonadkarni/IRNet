DUMMY_INPUT = {
    "db_id": "customer",
    "query": "SELECT count(*) FROM customer",
    "query_toks": [
        "SELECT",
        "count",
        "(",
        "*",
        ")",
        "FROM",
        "customer"
    ],
    "query_toks_no_value": [
        "select",
        "count",
        "(",
        "*",
        ")",
        "from",
        "customer"
    ],
    "question": "How many customers have their birthday in May?",
    "question_toks": [
        "How",
        "many",
        "customers",
        "have",
        "their",
        "birthday",
        "in",
        "May",
        "?"
    ],
    "sql": {
        "except": None,
        "from": {
            "conds": [],
            "table_units": [
                [
                    "table_unit",
                    0
                ]
            ]
        },
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [
            False,
            [
                [
                    3,
                    [
                        0,
                        [
                            0,
                            0,
                            False
                        ],
                        None
                    ]
                ]
            ]
        ],
        "union": None,
        "where": []
    }
}