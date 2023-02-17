from hypothesis import given

from zodipy_local.zodipy_local import Zodipy

from ._strategies import model


@given(model())
def test_ipd_model_repr(model: Zodipy) -> None:
    repr(model)
    repr(model._ipd_model)
