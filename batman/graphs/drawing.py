"""Tools for drawing Decay and Reaction graphs in a way that makes physical
sense.

"""
from collections.abc import Hashable

import networkx as nx
import numpy as np

from .decay import DecayGraph
from .reaction import ReactionGraph


def draw_multigraph_edge_labels(g: nx.MultiGraph, pos, edge_labels=None,
                                label_pos=0.5, font_size=10, font_color='k',
                                font_family='sans-serif', font_weight='normal',
                                alpha=1.0, bbox=None, ax=None, rotate=True,
                                **kwds):
    """Draw edge labels for a MultiGraph.
    The way this works is that labels are written for each key under the
    arrow, because I don't feel like drawing multiple arrows.

    Most of this code is taken as is from the networkx package, and was
    slightly edited to allow multigraphs to be used.

    Parameters
    ----------
    g : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    alpha : float
       The text transparency (default=1.0)

    edge_labels : dictionary
       Edge labels in a dictionary keyed by edge two-tuple of text
       labels (default=None). Only labels for the keys in the dictionary
       are drawn.

    label_pos : float
       Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int
       Font size for text labels (default=12)

    font_color : string
       Font color string (default='k' black)

    font_weight : string
       Font weight (default='normal')

    font_family : string
       Font family (default='sans-serif')

    bbox : Matplotlib bbox
       Specify text box shape and colors.

    rotate : bool
       Turn on rotation according to screen layout.

    Returns
    -------
    dict
        `dict` of labels keyed on the edges

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()

    In the networkx package.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rc
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    rc('text', usetex=True)

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v, k): d for u, v, k, d in g.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    used = {}
    for (n1, n2, k), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        x, y = (x1 * label_pos + x2 * (1.0 - label_pos),
                y1 * label_pos + y2 * (1.0 - label_pos))

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < - 90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(np.array((angle,)),
                                                        xy.reshape((1, 2)))[0]
        else:
            trans_angle = 0.0

        cnt = used.setdefault((n1, n2), 0)
        trans_mat = np.array([(np.cos(trans_angle), -np.sin(trans_angle)),
                              (np.sin(trans_angle), np.cos(trans_angle))])
        offset = np.array((0, 0.1 * cnt + (0.1 if n1 == n2 else 0))).T
        x, y = np.array((x, y)).T - trans_mat @ offset
        used[(n1, n2)] = cnt + 1
        # use default box of white with white border
        if bbox is None:
            bbox = {'boxstyle': 'round', 'ec': (1.0, 1.0, 1.0),
                        'fc': (1.0, 1.0, 1.0), }
        label = str(label)  # this makes "1" and 1 labeled the same

        # set optional alignment
        horizontalalignment = kwds.get('horizontalalignment', 'center')
        verticalalignment = kwds.get('verticalalignment', 'center')

        t = ax.text(x, y, label, size=font_size, color=font_color,
                    family=font_family, weight=font_weight, alpha=alpha,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment, rotation=trans_angle,
                    transform=ax.transData, bbox=bbox, zorder=1)
        text_items[(n1, n2, k)] = t

    return text_items


def draw(g: DecayGraph | ReactionGraph,
         pos: dict[Hashable, tuple[float, float]] | None = None, ax=None,
         delimiter: str = r'$\rightarrow$', with_labels: bool = True, **kwds):
    """A networkx.draw wrapper that gives a physics-like graph structure.

    Most of the code is taken from networkx's draw, but it isn't good for us
    because it sets the axis as off, and we need it on.

    Parameters
    ----------
    g - Graph to draw
    pos - Position of each isotope(node) in the graph. Defaults to
          physics-like structure.
    ax - Matplotlib Axes object, optional
         Draw the graph in specified Matplotlib axes.
    delimiter - Delimiter between subsequent modes of decay/reaction
    with_labels - Flag to display node names. Defaults True
    kwargs - Any other parameter, sent as is to networkx.draw

    Returns
    -------

    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rc
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    rc('text', usetex=True)

    pos = pos or {iso: (3 * iso.A + iso.m, iso.Z) for iso in g}

    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor('w')
    if ax is None:
        # noinspection PyProtectedMember
        if cf._axstack() is None:
            ax = cf.add_axes((0, 0, 1, 1))
        else:
            ax = cf.gca()

    labels = {(u, v, k): delimiter.join(k) for u, v, k in g.edges}
    draw_multigraph_edge_labels(g, pos=pos, ax=ax, label_pos=0.8,
                                edge_labels=labels, **kwds)
    nx.draw_networkx(g, pos=pos, with_labels=with_labels, ax=ax, **kwds)
