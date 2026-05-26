import matplotlib.pyplot as plt


def draw_quantum_feature_map(
    feature_map,
    output_path=None,
):
    """
    Draw and optionally save a quantum feature map circuit.

    Parameters
    ----------
    feature_map :
        Qiskit feature map circuit.

    output_path : str, optional
        Path to save PDF figure.
    """

    figure = feature_map.draw(
        output="mpl"
    )

    if output_path is not None:
        figure.savefig(
            output_path,
            bbox_inches="tight",
        )

    return figure