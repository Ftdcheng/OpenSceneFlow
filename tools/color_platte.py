from typing import Sequence, Tuple, Final
from enum import Enum

RGBColor = Tuple[float, float, float]

class ColorPlatte:
    def __init__(self, hex_colors:Sequence[str]):
        self._colors = [self._color_hex2float(hex_color) for hex_color in hex_colors]

    def _color_hex2float(self, hex_color:str) -> RGBColor:
        hex_color = hex_color.lstrip('#')
        return (
            int(hex_color[0:2], 16) / 255.0,
            int(hex_color[2:4], 16) / 255.0,
            int(hex_color[4:6], 16) / 255.0,
        )

    def get(self, idx):
        return self._colors[idx % len(self._colors)]

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return len(self._colors)


COLOR_MAP_HEX = [
    '#a6cee3', '#de2d26', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
    '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#8dd3c7',
    '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
    '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'
]

WHITE = RGBColor(1., 1., 1.)

color_map = ColorPlatte(COLOR_MAP_HEX)

if __name__ == '__main__':
    n = len(color_map)
    for i in range(2*n):
        print(color_map[i])