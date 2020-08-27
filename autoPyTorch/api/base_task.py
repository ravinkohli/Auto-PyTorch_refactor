class Task:

    def __init__(
        self,
        **pipeline_kwargs,
    ):
        self._pipeline = pipeline_kwargs['pipeline']
        self._pipeline_config = pipeline_kwargs['pipeline_config']
        self._optimizer = pipeline_kwargs['optimizer']
        self._resource_scheduler = ['resource_scheduler']
        self._backend = ['backend']

    def search(
        self,
        X,
        y,
        X_val,
        y_val,
        X_test,
        y_test,
        search_space,
    ):
        pass

    def fit(
        self,
        X,
        y,
        X_test,
        y_test,
        model_config,
    ):
        pass

    def predict(
        self,
        X_test,
        y_test,
    ):
        pass

    def score(
        self,
        X_test,
        y_test,
    ):
        pass

    def get_pipeline_config(
        self
    ):
        return self._pipeline_config

    def set_pipeline_config(
        self,
        new_pipeline_config,
    ):
        self._pipeline_config = new_pipeline_config

    def get_incumbent_results(
        self
    ):
        pass

    def get_incumbent_config(
        self
    ):
        pass

    def get_default_search_space(
        self
    ):
        pass
