import math
def num_tiles(width, height, window, stride, include_partial=False):

    if include_partial:
        tiles_x = math.ceil((width - window) / stride) + 1 if width > window else 1
        tiles_y = math.ceil((height - window) / stride) + 1 if height > window else 1
    else:
        if width < window or height < window:
            return 0, 0, 0
        tiles_x = ((width - window) // stride) + 1
        tiles_y = ((height - window) // stride) + 1

    total_tiles = tiles_x * tiles_y
    print(f"Total of {total_tiles} ({window}x{window}) Tiles for ({width}x{height}) images")

configs = [
    [9504, 6336, 768, 512],
    [4752, 3168, 624, 416],
    [2376, 1584, 432, 288],
    [1188, 792, 225, 150]
]

for x, y, w, s in configs:
    num_tiles(x,y,w,s)