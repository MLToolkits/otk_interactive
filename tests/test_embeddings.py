import pytest
from unittest.mock import patch
import plotly.graph_objects as go
from sklearn.datasets import load_digits
from otk_interactive.otk import interactive_embeddings


def mock_get_embeddings(X, y, dims, embedding_type):
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode="markers"))
    return None, fig


@pytest.fixture
def mock_get_embeddings_fixture():
    with patch(
        "otk_interactive.otk.interactive_embeddings", side_effect=mock_get_embeddings
    ):
        yield


def test_interactive_embeddings(mock_get_embeddings_fixture):
    digits = load_digits()
    X, y = digits.images, digits.target
    dims = 2
    embedding_type = "t-SNE embedding"
    fig = interactive_embeddings(X, y, dims, embedding_type)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].mode == "markers"
