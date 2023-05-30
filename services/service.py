# THIS IS JUST AN EXAMPLE!
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

model_runner = bentoml.sklearn.get("cc_model:latest").to_runner()
svc = bentoml.Service("cc_model", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    return result