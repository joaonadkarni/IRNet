from flask import Flask, make_response


class BaseFlaskApp(Flask):

    """
    Base class that encapsulates commonly used logic when implementing Flask apps
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_apis = {}
        self.production_apis = []
        # used when running tests
        self.add_url_rule("/status", endpoint="root_status", view_func=self.status, methods=['GET'])

    def status(self):
        """status view"""
        response = make_response("OK")
        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        return response

    def register_pipelines(self):
        for url, pipeline in self.known_apis.items():
            pipeline.register(self, url)

    def preload(self, *args, **kwargs):
        """
        Preloads the production APIs.
        Call the preload method of the APIs.
        If they don't implement this method it's assumed that they don't need to be preloaded and so this method will do
        nothing in that API
        """
        for url in self.production_apis:
            api = self.known_apis[url]
            self.logger.info(f'Preloading API {url}: class {api.__class__.__name__}')
            if hasattr(api, "preload") and callable(api.preload):
                api.preload(**kwargs)
