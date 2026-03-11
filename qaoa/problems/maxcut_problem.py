import networkx as nx

from .maxkcut_binary_powertwo import MaxKCutBinaryPowerOfTwo


class MaxCut(MaxKCutBinaryPowerOfTwo):
    """
    Standard Max Cut problem.

    Convenience subclass of :class:`MaxKCutBinaryPowerOfTwo` that fixes
    ``k_cuts=2``, so callers only need to supply the graph.

    Attributes:
        G (nx.Graph): The input graph.
        method (str): Circuit construction method (``"Diffusion"`` or
            ``"PauliBasis"``).
        fix_one_node (bool): Whether to fix the last node to colour 1.

    Methods:
        Inherits all methods from :class:`MaxKCutBinaryPowerOfTwo`.
    """

    def __init__(
        self,
        G: nx.Graph,
        method: str = "Diffusion",
        fix_one_node: bool = False,
    ) -> None:
        """
        Initialises the MaxCut problem.

        Args:
            G (nx.Graph): The input graph on which Max Cut is defined.
            method (str): Circuit construction method.  Defaults to
                ``"Diffusion"``.
            fix_one_node (bool): If ``True``, the last node is fixed to
                colour 1, removing one qubit.  Defaults to ``False``.
        """
        super().__init__(G, k_cuts=2, method=method, fix_one_node=fix_one_node)
