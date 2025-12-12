"""Generate a synthetic semiconductor-style layout in OASIS format using gdstk.

The script emits a small hierarchical layout with CMOS-like standard cells,
an SRAM-inspired macro, and a bit of global routing. Run it directly to
produce an OASIS file:

    python generate_oasis.py --out synthetic_layout.oas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gdstk

# Layer / datatype assignments chosen to loosely mimic a CMOS stack.
LAYER = {
    "nwell": (10, 0),
    "active": (1, 0),
    "poly": (2, 0),
    "contact": (3, 0),
    "metal1": (4, 0),
    "via1": (30, 0),
    "metal2": (5, 0),
    "via2": (31, 0),
    "metal3": (6, 0),
    "text": (64, 0),
}


def _rect(x0, y0, x1, y1, layer_name: str) -> gdstk.Polygon:
    """Helper to build a rectangle on a named layer."""
    layer, datatype = LAYER[layer_name]
    return gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype)


def _contacts(origin, cols, rows, size, pitch, layer_name: str):
    """Create a small contact/via array starting at origin."""
    ox, oy = origin
    shapes = []
    for c in range(cols):
        for r in range(rows):
            x0 = ox + c * pitch
            y0 = oy + r * pitch
            shapes.append(_rect(x0, y0, x0 + size, y0 + size, layer_name))
    return shapes


def build_standard_cell(lib: gdstk.Library, name: str = "STD_INV") -> gdstk.Cell:
    """Construct a CMOS-style inverter cell with stacked PMOS/NMOS."""
    cell = lib.new_cell(name)

    gate_pitch = 1.4
    gates = 4
    diffusion_margin = 1.0
    active_width = 2.2
    poly_extension = 0.35
    contact_size = 0.25
    contact_pitch = 0.55

    active_length = diffusion_margin * 2 + (gates - 1) * gate_pitch
    pmos_y = active_width + 1.0

    # N-well encompassing PMOS.
    cell.add(
        _rect(
            -0.5,
            pmos_y - 0.6,
            active_length + 0.5,
            pmos_y + active_width + 0.9,
            "nwell",
        )
    )

    # Active regions.
    cell.add(_rect(0, 0, active_length, active_width, "active"))
    cell.add(_rect(0, pmos_y, active_length, pmos_y + active_width, "active"))

    # Poly gates running through both devices.
    for i in range(gates):
        x = diffusion_margin + i * gate_pitch
        cell.add(
            _rect(
                x - 0.09,
                -poly_extension,
                x + 0.09,
                pmos_y + active_width + poly_extension,
                "poly",
            )
        )

    # Source/drain contacts on NMOS and PMOS.
    n_rows = max(1, int(active_width // contact_pitch))
    p_rows = max(1, int(active_width // contact_pitch))
    left_block = (0.25, 0.35)
    right_block = (active_length - 0.25 - contact_size - (contact_pitch * (n_rows - 1)), 0.35)
    cell.add(*_contacts(left_block, 1, n_rows, contact_size, contact_pitch, "contact"))
    cell.add(*_contacts(right_block, 1, n_rows, contact_size, contact_pitch, "contact"))
    cell.add(
        *_contacts(
            (left_block[0], pmos_y + 0.35),
            1,
            p_rows,
            contact_size,
            contact_pitch,
            "contact",
        )
    )
    cell.add(
        *_contacts(
            (right_block[0], pmos_y + 0.35),
            1,
            p_rows,
            contact_size,
            contact_pitch,
            "contact",
        )
    )

    # Metal1 over diffusion plus gate straps at the edges.
    cell.add(_rect(-0.2, -0.2, 0.6, pmos_y + active_width + 0.2, "metal1"))
    cell.add(_rect(active_length - 0.6, -0.2, active_length + 0.2, pmos_y + active_width + 0.2, "metal1"))
    cell.add(_rect(-0.2, -0.2, active_length + 0.2, 0.6, "metal1"))  # shared ground rail
    cell.add(
        _rect(
            -0.2,
            pmos_y + active_width - 0.6,
            active_length + 0.2,
            pmos_y + active_width + 0.2,
            "metal1",
        )
    )  # VDD rail

    # Via stacks to metal2 for VDD/GND.
    via_cols = 2
    vdd_origin = (0.35, pmos_y + active_width - 0.55)
    gnd_origin = (active_length - 0.9, -0.55)
    cell.add(*_contacts(vdd_origin, via_cols, 1, 0.2, 0.35, "via1"))
    cell.add(*_contacts(gnd_origin, via_cols, 1, 0.2, 0.35, "via1"))

    # Text label for debugging.
    cell.add(
        gdstk.Label(
            name,
            (active_length / 2, pmos_y + active_width + 0.7),
            layer=LAYER["text"][0],
            texttype=LAYER["text"][1],
        )
    )
    return cell


def build_memory_bit(lib: gdstk.Library, name: str = "MEM_BIT") -> gdstk.Cell:
    """Small memory bit-like primitive."""
    cell = lib.new_cell(name)
    size = 1.2
    cell.add(_rect(0, 0, size, size, "active"))
    cell.add(_rect(-0.15, -0.15, size + 0.15, size + 0.15, "metal1"))
    cell.add(*_contacts((0.3, 0.3), 2, 2, 0.22, 0.32, "via1"))
    cell.add(_rect(-0.15, size + 0.3, size + 0.15, size + 0.6, "metal2"))
    return cell


def build_pattern_cell(lib: gdstk.Library, name: str = "PATTERNS") -> gdstk.Cell:
    """Cell containing canonical shapes: line ends, T-junctions, via corners, rectangular corners."""
    cell = lib.new_cell(name)
    y = 0.0

    # Line ends: three widths, flush termination.
    line_lengths = [3.0, 4.5, 6.0]
    for idx, length in enumerate(line_lengths):
        x0 = 0.0
        y0 = y + idx * 1.2
        cell.add(_rect(x0, y0, x0 + length, y0 + 0.4, "metal2"))

    y += len(line_lengths) * 1.2 + 0.8
    line_label_y = 0.6

    # T-junction: vertical trunk with horizontal bar.
    trunk_x = 0.8
    trunk_len = 3.5
    t_y = y
    cell.add(_rect(trunk_x, t_y, trunk_x + 0.5, t_y + trunk_len, "metal2"))
    bar_y = t_y + trunk_len - 0.5
    cell.add(_rect(trunk_x - 1.5, bar_y, trunk_x + 2.0, bar_y + 0.5, "metal2"))
    y += trunk_len + 0.8
    t_label_y = t_y + trunk_len * 0.4

    # Via corners: metal pad with vias near the four corners.
    pad_size = 3.0
    pad_origin = (0.0, y)
    px, py = pad_origin
    cell.add(_rect(px, py, px + pad_size, py + pad_size, "metal3"))
    via_off = 0.25
    cell.add(*_contacts((px + via_off, py + via_off), 2, 2, 0.18, 0.28, "via2"))
    cell.add(*_contacts((px + pad_size - via_off - 0.36, py + via_off), 2, 2, 0.18, 0.28, "via2"))
    cell.add(*_contacts((px + via_off, py + pad_size - via_off - 0.36), 2, 2, 0.18, 0.28, "via2"))
    cell.add(*_contacts((px + pad_size - via_off - 0.36, py + pad_size - via_off - 0.36), 2, 2, 0.18, 0.28, "via2"))
    y += pad_size + 0.8
    via_label_y = py + pad_size + 0.2

    # Rectangular corners: L-shape polygon plus acute corner notch.
    rect_y = y
    l_outer = gdstk.Polygon(
        [(0, rect_y), (4.0, rect_y), (4.0, rect_y + 0.8), (0.8, rect_y + 0.8), (0.8, rect_y + 4.0), (0, rect_y + 4.0)],
        layer=LAYER["metal1"][0],
        datatype=LAYER["metal1"][1],
    )
    cell.add(l_outer)
    notch = gdstk.Polygon(
        [(1.2, rect_y + 1.2), (3.2, rect_y + 1.2), (3.2, rect_y + 1.8), (1.8, rect_y + 1.8), (1.8, rect_y + 3.2), (1.2, rect_y + 3.2)],
        layer=LAYER["metal1"][0],
        datatype=LAYER["metal1"][1],
    )
    cell.add(notch)

    return cell


def build_layout(out_path: Path, compress: bool) -> None:
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    std_cell = build_standard_cell(lib)
    mem_bit = build_memory_bit(lib)
    patterns = build_pattern_cell(lib)

    # Standard-cell row.
    row = lib.new_cell("STD_ROW")
    pitch_x = 8.0
    columns = 6
    row.add(gdstk.Reference(std_cell, columns=columns, rows=1, spacing=(pitch_x, 0)))

    # Memory macro using the bit cell in a 12x8 grid.
    mem_array = lib.new_cell("MEM_MACRO")
    mem_array.add(gdstk.Reference(mem_bit, columns=12, rows=8, spacing=(1.7, 1.7)))
    mem_array.add(_rect(-1.0, -1.0, 1.7 * 12 + 1.0, 1.7 * 8 + 1.0, "metal2"))
    mem_array.add(
        gdstk.Label("MEMORY", (1.7 * 6, 1.7 * 8 + 0.6), layer=LAYER["text"][0], texttype=LAYER["text"][1])
    )

    core_width = columns * pitch_x + 18.0
    top = lib.new_cell("TOP")
    # Place three rows of standard cells.
    for r in range(3):
        y_offset = r * 6.0
        top.add(gdstk.Reference(row, origin=(0, y_offset)))

    # Place the memory macro off to the side.
    top.add(gdstk.Reference(mem_array, origin=(columns * pitch_x + 6.0, 2.0)))

    # Drop pattern snippets for line ends, T-junctions, via corners, and rectangular corners.
    top.add(gdstk.Reference(patterns, origin=(core_width + 8.0, -2.0)))

    # Global routing in metal2/metal3 with vias.
    m2 = gdstk.FlexPath(
        [(0, 3.0), (columns * pitch_x, 3.0)],
        width=0.6,
        layer=LAYER["metal2"][0],
        datatype=LAYER["metal2"][1],
        simple_path=True,
    )
    top.add(m2)

    serpent = gdstk.FlexPath(
        [(0, -2.0)],
        width=0.8,
        ends="round",
        layer=LAYER["metal3"][0],
        datatype=LAYER["metal3"][1],
    )
    step_x = 12.0
    step_y = 4.0
    current_y = -2.0
    for i in range(8):
        x = (i + 1) * step_x
        current_y += (-1) ** i * step_y
        serpent.segment((x, current_y), relative=False)
    top.add(serpent)

    # Stitch metal2 to metal3 at a few points.
    for idx in range(4):
        vx = (idx + 1) * 2 * step_x / 4
        vy = 3.0 + (idx % 2) * 2.0
        top.add(*_contacts((vx - 0.2, vy - 0.2), 2, 2, 0.18, 0.3, "via2"))
        top.add(_rect(vx - 0.3, vy - 0.3, vx + 0.3, vy + 0.3, "metal3"))

    # Frame around the design on metal3.
    height = 20.0
    top.add(_rect(-1.0, -3.0, core_width + 1.0, -2.0, "metal3"))  # bottom
    top.add(_rect(-1.0, height + 2.0, core_width + 1.0, height + 3.0, "metal3"))  # top
    top.add(_rect(-1.0, -3.0, 0.0, height + 3.0, "metal3"))  # left
    top.add(_rect(core_width, -3.0, core_width + 1.0, height + 3.0, "metal3"))  # right

    lib.write_oas(
        str(out_path),
        compression_level=9 if compress else 0,
        detect_rectangles=True,
        detect_trapezoids=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic OASIS layout with CMOS-like features.")
    parser.add_argument("--out", type=Path, default=Path("synthetic_layout.oas"), help="Output OASIS file path.")
    parser.add_argument("--no-compress", action="store_true", help="Disable OASIS CBLOCK compression.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_layout(args.out, compress=not args.no_compress)
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
