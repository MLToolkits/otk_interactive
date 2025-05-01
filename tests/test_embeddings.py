import pytest
import numpy as np
from unittest.mock import patch
import plotly.graph_objects as go
from otk_interactive.otk import interactive_embeddings


def mock_get_embeddings(X, y, dims, embedding_type):
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode="markers"))
    return None, fig


@pytest.fixture
def mock_get_embeddings_fixture():
    with patch(
        "otk_interactive.otk.interactive_embeddings.get_embeddings",
        side_effect=mock_get_embeddings,
    ):
        yield


def generate_correlated_data(n_samples=100, n_features=10, correlation_strength=0.9):
    np.random.seed(42)
    X = np.random.randn(n_samples, 1)
    for i in range(1, n_features):
        X = np.hstack(
            (
                X,
                X[:, [0]] * correlation_strength
                + np.random.randn(n_samples, 1) * (1 - correlation_strength),
            )
        )
    y = (np.sum(X, axis=1) > 0).astype(int)
    return X, y


def test_interactive_embeddings(mock_get_embeddings_fixture):
    X, y = generate_correlated_data(n_samples=100, n_features=20)
    dims = 2
    embedding_type = "tsne"
    _, fig = interactive_embeddings(X, y, dims, embedding_type)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].mode == "markers"
