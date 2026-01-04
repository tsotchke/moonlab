"""
Feynman Diagram Visualization

Publication-quality Feynman diagram rendering in multiple formats:
- ASCII (terminal output)
- Matplotlib figure (interactive/saveable)
- SVG (web/publication)
- LaTeX/TikZ-Feynman (papers)

Example:
--------
    >>> from moonlab.visualization import FeynmanDiagram
    >>>
    >>> # Create e+ e- -> mu+ mu- diagram
    >>> fd = FeynmanDiagram.ee_to_mumu()
    >>> fd.render()  # Returns matplotlib figure
    >>> fd.to_ascii()  # Returns ASCII string
    >>> fd.to_latex()  # Returns TikZ-Feynman code
    >>>
    >>> # Custom diagram
    >>> fd = FeynmanDiagram("Custom process")
    >>> v1 = fd.vertex(-2, 0, "e-")
    >>> v2 = fd.vertex(0, 0)
    >>> v3 = fd.vertex(2, 0, "e-")
    >>> fd.fermion(v1, v2, "e-").fermion(v2, v3, "e-")
    >>> fd.photon(v2, v2, "gamma")
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle, FancyArrowPatch
    from matplotlib.path import Path
    import matplotlib.patches as patches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ParticleType(Enum):
    """Particle types for propagators"""
    FERMION = "fermion"           # e-, mu-, quark
    ANTIFERMION = "antifermion"   # e+, mu+, antiquark
    PHOTON = "photon"             # gamma
    GLUON = "gluon"               # g
    W_BOSON = "w_boson"           # W+/W-
    Z_BOSON = "z_boson"           # Z0
    HIGGS = "higgs"               # H
    SCALAR = "scalar"             # Generic scalar


@dataclass
class Vertex:
    """Vertex (interaction point)"""
    x: float
    y: float
    id: int
    label: str = ""
    is_external: bool = False


@dataclass
class Propagator:
    """Propagator (line between vertices)"""
    from_vertex: int
    to_vertex: int
    type: ParticleType
    label: str = ""
    arrow_forward: bool = True
    is_external: bool = False


class FeynmanDiagram:
    """
    Feynman diagram builder and renderer.

    Particle types and line styles:
    - Fermion: solid line with forward arrow
    - Antifermion: solid line with backward arrow
    - Photon: wavy line
    - Gluon: curly line
    - W/Z boson: dashed line
    - Higgs: dotted line
    - Scalar: dashed line

    Example:
    --------
        # Standard diagram
        fd = FeynmanDiagram.compton_scattering()
        fd.render()

        # Custom diagram
        fd = FeynmanDiagram("My process")
        v1 = fd.vertex(-2, 0, "e-")
        v2 = fd.vertex(0, 0)
        v3 = fd.vertex(2, 0, "e-")
        fd.fermion(v1, v2).fermion(v2, v3)
        fd.photon(v2, fd.vertex(0, 1.5, "gamma"))
    """

    def __init__(self, process: str = ""):
        """
        Initialize Feynman diagram.

        Parameters:
        -----------
        process : str
            Process notation (e.g., "e+ e- -> mu+ mu-")
        """
        self.process = process
        self.title = ""
        self.vertices: List[Vertex] = []
        self.propagators: List[Propagator] = []
        self.loop_order = 0
        self._next_id = 0

    def set_title(self, title: str) -> 'FeynmanDiagram':
        """Set diagram title."""
        self.title = title
        return self

    def set_loop_order(self, order: int) -> 'FeynmanDiagram':
        """Set loop order (0=tree, 1=one-loop, etc.)."""
        self.loop_order = order
        return self

    def vertex(self, x: float, y: float, label: str = "") -> int:
        """
        Add vertex and return its ID.

        Parameters:
        -----------
        x, y : float
            Vertex position
        label : str
            Particle label for external vertices

        Returns:
        --------
        int : Vertex ID
        """
        vid = self._next_id
        is_external = bool(label)
        self.vertices.append(Vertex(x=x, y=y, id=vid, label=label, is_external=is_external))
        self._next_id += 1
        return vid

    def external_vertex(self, x: float, y: float, label: str) -> int:
        """Add external vertex with particle label."""
        vid = self._next_id
        self.vertices.append(Vertex(x=x, y=y, id=vid, label=label, is_external=True))
        self._next_id += 1
        return vid

    def _add_propagator(self, from_v: int, to_v: int, ptype: ParticleType,
                        label: str = "", forward: bool = True) -> 'FeynmanDiagram':
        """Internal: add propagator."""
        self.propagators.append(Propagator(
            from_vertex=from_v,
            to_vertex=to_v,
            type=ptype,
            label=label,
            arrow_forward=forward
        ))
        return self

    def fermion(self, from_v: int, to_v: int, label: str = "") -> 'FeynmanDiagram':
        """Add fermion propagator (solid line with arrow)."""
        return self._add_propagator(from_v, to_v, ParticleType.FERMION, label, True)

    def antifermion(self, from_v: int, to_v: int, label: str = "") -> 'FeynmanDiagram':
        """Add antifermion propagator (arrow reversed)."""
        return self._add_propagator(from_v, to_v, ParticleType.ANTIFERMION, label, False)

    def photon(self, from_v: int, to_v: int, label: str = "gamma") -> 'FeynmanDiagram':
        """Add photon propagator (wavy line)."""
        return self._add_propagator(from_v, to_v, ParticleType.PHOTON, label)

    def gluon(self, from_v: int, to_v: int, label: str = "g") -> 'FeynmanDiagram':
        """Add gluon propagator (curly line)."""
        return self._add_propagator(from_v, to_v, ParticleType.GLUON, label)

    def w_boson(self, from_v: int, to_v: int, label: str = "W") -> 'FeynmanDiagram':
        """Add W boson propagator (dashed line)."""
        return self._add_propagator(from_v, to_v, ParticleType.W_BOSON, label)

    def z_boson(self, from_v: int, to_v: int, label: str = "Z") -> 'FeynmanDiagram':
        """Add Z boson propagator (dashed line)."""
        return self._add_propagator(from_v, to_v, ParticleType.Z_BOSON, label)

    def higgs(self, from_v: int, to_v: int, label: str = "H") -> 'FeynmanDiagram':
        """Add Higgs propagator (dotted line)."""
        return self._add_propagator(from_v, to_v, ParticleType.HIGGS, label)

    def scalar(self, from_v: int, to_v: int, label: str = "") -> 'FeynmanDiagram':
        """Add scalar propagator (dashed line)."""
        return self._add_propagator(from_v, to_v, ParticleType.SCALAR, label)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, max_x, min_y, max_y)."""
        if not self.vertices:
            return (0, 1, 0, 1)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return (min(xs), max(xs), min(ys), max(ys))

    def _get_vertex(self, vid: int) -> Optional[Vertex]:
        """Get vertex by ID."""
        for v in self.vertices:
            if v.id == vid:
                return v
        return None

    # Standard diagrams
    @classmethod
    def qed_vertex(cls) -> 'FeynmanDiagram':
        """Create QED vertex (e- -> e- + gamma)."""
        fd = cls("QED vertex")
        v1 = fd.external_vertex(-2, 0, "e-")
        v2 = fd.vertex(0, 0)
        v3 = fd.external_vertex(2, 0, "e-")
        v4 = fd.external_vertex(0, 1.5, "gamma")
        fd.fermion(v1, v2, "e-")
        fd.fermion(v2, v3, "e-")
        fd.photon(v2, v4, "gamma")
        return fd

    @classmethod
    def ee_to_mumu(cls) -> 'FeynmanDiagram':
        """Create e+ e- -> mu+ mu- diagram (s-channel)."""
        fd = cls("e+ e- -> mu+ mu-")
        v1 = fd.external_vertex(-2, 1, "e-")
        v2 = fd.external_vertex(-2, -1, "e+")
        v3 = fd.vertex(0, 0)
        v4 = fd.external_vertex(2, 1, "mu-")
        v5 = fd.external_vertex(2, -1, "mu+")
        fd.fermion(v1, v3, "e-")
        fd.antifermion(v2, v3, "e+")
        fd.fermion(v3, v4, "mu-")
        fd.antifermion(v3, v5, "mu+")
        return fd

    @classmethod
    def compton_scattering(cls) -> 'FeynmanDiagram':
        """Create Compton scattering diagram."""
        fd = cls("Compton scattering")
        v1 = fd.external_vertex(-2, 0, "e-")
        v2 = fd.external_vertex(-1, 1.5, "gamma")
        v3 = fd.vertex(-0.5, 0)
        v4 = fd.vertex(0.5, 0)
        v5 = fd.external_vertex(2, 0, "e-")
        v6 = fd.external_vertex(1, 1.5, "gamma")
        fd.fermion(v1, v3, "e-")
        fd.photon(v2, v3, "gamma")
        fd.fermion(v3, v4, "e-")
        fd.fermion(v4, v5, "e-")
        fd.photon(v4, v6, "gamma")
        return fd

    @classmethod
    def pair_annihilation(cls) -> 'FeynmanDiagram':
        """Create e+ e- -> gamma gamma diagram."""
        fd = cls("Pair annihilation")
        v1 = fd.external_vertex(-2, 1, "e-")
        v2 = fd.external_vertex(-2, -1, "e+")
        v3 = fd.vertex(-0.5, 0)
        v4 = fd.vertex(0.5, 0)
        v5 = fd.external_vertex(2, 1, "gamma")
        v6 = fd.external_vertex(2, -1, "gamma")
        fd.fermion(v1, v3, "e-")
        fd.antifermion(v2, v3, "e+")
        fd.fermion(v3, v4)
        fd.photon(v4, v5, "gamma")
        fd.photon(v4, v6, "gamma")
        return fd

    @classmethod
    def electron_self_energy(cls) -> 'FeynmanDiagram':
        """Create electron self-energy (one-loop)."""
        fd = cls("Electron self-energy")
        fd.set_loop_order(1)
        v1 = fd.external_vertex(-2.5, 0, "e-")
        v2 = fd.vertex(-1, 0)
        v3 = fd.vertex(0, 0.8)
        v4 = fd.vertex(1, 0)
        v5 = fd.external_vertex(2.5, 0, "e-")
        fd.fermion(v1, v2, "e-")
        fd.fermion(v2, v3)
        fd.fermion(v3, v4)
        fd.fermion(v4, v5, "e-")
        fd.photon(v2, v4, "gamma")
        return fd

    @classmethod
    def vacuum_polarization(cls) -> 'FeynmanDiagram':
        """Create vacuum polarization (one-loop)."""
        fd = cls("Vacuum polarization")
        fd.set_loop_order(1)
        v1 = fd.external_vertex(-2.5, 0, "gamma")
        v2 = fd.vertex(-1, 0)
        v3 = fd.vertex(0, 0.8)
        v4 = fd.vertex(0, -0.8)
        v5 = fd.vertex(1, 0)
        v6 = fd.external_vertex(2.5, 0, "gamma")
        fd.photon(v1, v2, "gamma")
        fd.fermion(v2, v3, "e-")
        fd.fermion(v3, v5, "e-")
        fd.fermion(v5, v4, "e+")
        fd.fermion(v4, v2, "e+")
        fd.photon(v5, v6, "gamma")
        return fd

    @classmethod
    def moller_scattering(cls) -> 'FeynmanDiagram':
        """Create Moller scattering (e- e- -> e- e-)."""
        fd = cls("Moller scattering")
        v1 = fd.external_vertex(-2, 1.5, "e-")
        v2 = fd.external_vertex(-2, -1.5, "e-")
        v3 = fd.vertex(0, 1)
        v4 = fd.vertex(0, -1)
        v5 = fd.external_vertex(2, 1.5, "e-")
        v6 = fd.external_vertex(2, -1.5, "e-")
        fd.fermion(v1, v3, "e-")
        fd.fermion(v3, v5, "e-")
        fd.fermion(v2, v4, "e-")
        fd.fermion(v4, v6, "e-")
        fd.photon(v3, v4, "gamma")
        return fd

    @classmethod
    def bhabha_scattering(cls) -> 'FeynmanDiagram':
        """Create Bhabha scattering (e+ e- -> e+ e-)."""
        fd = cls("Bhabha scattering")
        v1 = fd.external_vertex(-2, 1, "e-")
        v2 = fd.external_vertex(-2, -1, "e+")
        v3 = fd.vertex(0, 0)
        v4 = fd.external_vertex(2, 1, "e-")
        v5 = fd.external_vertex(2, -1, "e+")
        fd.fermion(v1, v3, "e-")
        fd.antifermion(v2, v3, "e+")
        fd.fermion(v3, v4, "e-")
        fd.antifermion(v3, v5, "e+")
        return fd

    # Rendering methods
    def to_ascii(self, width: int = 60, height: int = 25) -> str:
        """
        Render diagram as ASCII string.

        Parameters:
        -----------
        width, height : int
            Output dimensions in characters

        Returns:
        --------
        str : ASCII representation
        """
        min_x, max_x, min_y, max_y = self.bounds
        x_range = max_x - min_x if max_x > min_x else 1
        y_range = max_y - min_y if max_y > min_y else 1

        margin = 8
        scale_x = (width - 2 * margin) / x_range
        scale_y = (height - 2 * margin) / y_range

        def map_x(x):
            return int(margin + (x - min_x) * scale_x)

        def map_y(y):
            return int(height - margin - 1 - (y - min_y) * scale_y)

        # Create grid
        grid = [[' '] * width for _ in range(height)]

        # Draw title
        if self.process:
            title_x = (width - len(self.process)) // 2
            for i, c in enumerate(self.process):
                if 0 <= title_x + i < width:
                    grid[1][title_x + i] = c

        # Draw propagators
        for prop in self.propagators:
            v1 = self._get_vertex(prop.from_vertex)
            v2 = self._get_vertex(prop.to_vertex)
            if not v1 or not v2:
                continue

            x1, y1 = map_x(v1.x), map_y(v1.y)
            x2, y2 = map_x(v2.x), map_y(v2.y)

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            steps = max(dx, dy, 1)

            for i in range(steps + 1):
                x = x1 + (x2 - x1) * i // steps
                y = y1 + (y2 - y1) * i // steps

                if 0 <= x < width and 0 <= y < height:
                    if prop.type == ParticleType.FERMION or prop.type == ParticleType.ANTIFERMION:
                        if dx > dy:
                            grid[y][x] = '>' if (x2 > x1) == prop.arrow_forward else '<'
                        else:
                            grid[y][x] = 'v' if (y2 > y1) == prop.arrow_forward else '^'
                        if i != steps // 2:
                            grid[y][x] = '-' if dx > dy else '|'
                    elif prop.type == ParticleType.PHOTON:
                        grid[y][x] = '~'
                    elif prop.type == ParticleType.GLUON:
                        grid[y][x] = '@'
                    elif prop.type in (ParticleType.W_BOSON, ParticleType.Z_BOSON, ParticleType.SCALAR):
                        grid[y][x] = '-' if i % 2 == 0 else ' '
                    elif prop.type == ParticleType.HIGGS:
                        grid[y][x] = '.' if i % 3 == 0 else ' '

            # Draw label
            if prop.label:
                mx = (x1 + x2) // 2
                my = (y1 + y2) // 2 - 1
                if my < 0:
                    my = 0
                for i, c in enumerate(prop.label):
                    if 0 <= mx + i - len(prop.label) // 2 < width:
                        grid[my][mx + i - len(prop.label) // 2] = c

        # Draw vertices
        for v in self.vertices:
            x, y = map_x(v.x), map_y(v.y)
            if 0 <= x < width and 0 <= y < height:
                if not v.is_external:
                    grid[y][x] = 'o'
                elif v.label:
                    for i, c in enumerate(v.label):
                        if 0 <= x + i - len(v.label) // 2 < width:
                            grid[y][x + i - len(v.label) // 2] = c

        lines = [''.join(row).rstrip() for row in grid]
        return '\n'.join(line for line in lines if line)

    def render(self, figsize: Tuple[float, float] = (8, 6),
               style: str = 'publication') -> 'plt.Figure':
        """
        Render diagram to matplotlib figure.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        style : str
            Style preset: 'publication', 'colorful'

        Returns:
        --------
        matplotlib.Figure : Rendered diagram
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required. Install: pip install matplotlib")

        styles = {
            'publication': {
                'fermion_color': 'black',
                'boson_color': 'black',
                'vertex_color': 'black',
                'font': 'serif',
                'fontsize': 12
            },
            'colorful': {
                'fermion_color': '#1976d2',
                'boson_color': '#d32f2f',
                'vertex_color': '#333333',
                'font': 'sans-serif',
                'fontsize': 11
            }
        }
        s = styles.get(style, styles['publication'])

        fig, ax = plt.subplots(figsize=figsize)

        min_x, max_x, min_y, max_y = self.bounds
        margin = 0.5

        # Draw propagators
        for prop in self.propagators:
            v1 = self._get_vertex(prop.from_vertex)
            v2 = self._get_vertex(prop.to_vertex)
            if not v1 or not v2:
                continue

            x1, y1 = v1.x, v1.y
            x2, y2 = v2.x, v2.y

            if prop.type in (ParticleType.FERMION, ParticleType.ANTIFERMION):
                self._draw_fermion(ax, x1, y1, x2, y2, prop.arrow_forward, s['fermion_color'])

            elif prop.type == ParticleType.PHOTON:
                self._draw_wavy(ax, x1, y1, x2, y2, s['boson_color'])

            elif prop.type == ParticleType.GLUON:
                self._draw_curly(ax, x1, y1, x2, y2, s['boson_color'])

            elif prop.type in (ParticleType.W_BOSON, ParticleType.Z_BOSON, ParticleType.SCALAR):
                ax.plot([x1, x2], [y1, y2], color=s['boson_color'], linewidth=1.5,
                        linestyle='--')

            elif prop.type == ParticleType.HIGGS:
                ax.plot([x1, x2], [y1, y2], color=s['boson_color'], linewidth=1.5,
                        linestyle=':')

            # Draw label
            if prop.label:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mx, my + 0.15, f"${prop.label}$", fontsize=s['fontsize'],
                        fontfamily=s['font'], ha='center', va='bottom', fontstyle='italic')

        # Draw vertices
        for v in self.vertices:
            if not v.is_external:
                ax.plot(v.x, v.y, 'ko', markersize=6)
            if v.label:
                offset_x = -0.3 if v.x < (min_x + max_x) / 2 else 0.3
                ha = 'right' if v.x < (min_x + max_x) / 2 else 'left'
                ax.text(v.x + offset_x, v.y, f"${v.label}$", fontsize=s['fontsize'],
                        fontfamily=s['font'], ha=ha, va='center', fontstyle='italic')

        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        ax.set_aspect('equal')
        ax.axis('off')

        if self.process:
            ax.set_title(f"${self.process}$", fontsize=s['fontsize'] + 2,
                         fontfamily=s['font'])

        return fig

    def _draw_fermion(self, ax, x1, y1, x2, y2, forward, color):
        """Draw fermion line with arrow."""
        if forward:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        else:
            ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    def _draw_wavy(self, ax, x1, y1, x2, y2, color, amplitude=0.08, frequency=8):
        """Draw wavy photon line."""
        if not HAS_MATPLOTLIB:
            return

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1)

        t = np.linspace(0, 1, 200)
        x_base = x1 + (x2 - x1) * t
        y_base = y1 + (y2 - y1) * t
        perp_x = -np.sin(angle) * amplitude * np.sin(2 * np.pi * frequency * t)
        perp_y = np.cos(angle) * amplitude * np.sin(2 * np.pi * frequency * t)

        ax.plot(x_base + perp_x, y_base + perp_y, color=color, lw=1.5)

    def _draw_curly(self, ax, x1, y1, x2, y2, color, amplitude=0.1, loops=6):
        """Draw curly gluon line."""
        if not HAS_MATPLOTLIB:
            return

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1)

        t = np.linspace(0, 1, 500)
        x_base = x1 + (x2 - x1) * t
        y_base = y1 + (y2 - y1) * t

        phase = 2 * np.pi * loops * t
        perp_x = -np.sin(angle) * amplitude * np.sin(phase) * (1 - np.cos(phase))
        perp_y = np.cos(angle) * amplitude * np.sin(phase) * (1 - np.cos(phase))

        ax.plot(x_base + perp_x, y_base + perp_y, color=color, lw=1.5)

    def to_latex(self) -> str:
        """
        Generate LaTeX/TikZ-Feynman representation.

        Returns:
        --------
        str : TikZ-Feynman code
        """
        lines = [
            "% Requires: \\usepackage{tikz-feynman}",
            "\\begin{tikzpicture}",
            "  \\begin{feynman}"
        ]

        # Vertices
        for v in self.vertices:
            if v.label and v.is_external:
                lines.append(f"    \\vertex (v{v.id}) at ({v.x:.1f}, {v.y:.1f}) {{${v.label}$}};")
            else:
                lines.append(f"    \\vertex (v{v.id}) at ({v.x:.1f}, {v.y:.1f});")

        lines.append("")
        lines.append("    \\diagram* {")

        # Propagators
        tikz_styles = {
            ParticleType.FERMION: "fermion",
            ParticleType.ANTIFERMION: "anti fermion",
            ParticleType.PHOTON: "photon",
            ParticleType.GLUON: "gluon",
            ParticleType.W_BOSON: "boson",
            ParticleType.Z_BOSON: "boson",
            ParticleType.HIGGS: "scalar",
            ParticleType.SCALAR: "scalar",
        }

        for prop in self.propagators:
            style = tikz_styles.get(prop.type, "plain")
            if prop.label:
                lines.append(f"      (v{prop.from_vertex}) -- [{style}, edge label=${prop.label}$] (v{prop.to_vertex}),")
            else:
                lines.append(f"      (v{prop.from_vertex}) -- [{style}] (v{prop.to_vertex}),")

        lines.append("    };")
        lines.append("  \\end{feynman}")
        lines.append("\\end{tikzpicture}")

        return "\n".join(lines)

    def save(self, filename: str, format: str = None, dpi: int = 300):
        """
        Save diagram to file.

        Parameters:
        -----------
        filename : str
            Output filename
        format : str, optional
            File format ('svg', 'png', 'pdf', 'tex')
        dpi : int
            Resolution for raster formats
        """
        if format is None:
            format = filename.split('.')[-1].lower()

        if format == 'tex' or format == 'latex':
            with open(filename, 'w') as f:
                f.write(self.to_latex())
        elif format == 'txt' or format == 'ascii':
            with open(filename, 'w') as f:
                f.write(self.to_ascii())
        else:
            if not HAS_MATPLOTLIB:
                raise ImportError("matplotlib required for graphical export")
            fig = self.render()
            fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

    def __repr__(self) -> str:
        return f"FeynmanDiagram('{self.process}', {len(self.vertices)} vertices, {len(self.propagators)} propagators)"
