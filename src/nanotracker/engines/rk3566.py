from ..exceptions import EngineUnavailableError, InferenceError, ModelNotFoundError


class RK3566Engine:
    def __init__(self, model_dir):
        names = ("backbone_template.rknn", "backbone_search.rknn", "head.rknn")
        paths = [model_dir / name for name in names]
        if any(not path.is_file() for path in paths):
            raise ModelNotFoundError(f"Expected RK3566 template, search and head models in {model_dir}")
        self.models = []
        try:
            from rknnlite.api import RKNNLite

            for path in paths:
                model = RKNNLite()
                self.models.append(model)
                if model.load_rknn(str(path)) != 0 or model.init_runtime() != 0:
                    raise RuntimeError(f"Cannot initialize {path.name}")
        except Exception as error:
            for model in self.models:
                model.release()
            raise EngineUnavailableError(f"Cannot initialize RK3566 runtime: {error}") from error

    def _run(self, model, inputs):
        try:
            outputs = model.inference(inputs=inputs, data_format="nchw")
            if outputs is None:
                raise RuntimeError("RKNN inference returned no outputs")
            return outputs
        except Exception as error:
            raise InferenceError(f"RK3566 inference failed: {error}") from error

    def backbone(self, image):
        model = self.models[0] if image.shape[-1] == 127 else self.models[1]
        return self._run(model, [image])[0]

    def head(self, template, search):
        return self._run(self.models[2], [template, search])
