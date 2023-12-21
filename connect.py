from moldiff_zndraw.main import  DiffusionModelling
from zndraw import ZnDraw
import time
import sys

while True:
    vis = ZnDraw(url="https://zndraw.icp.uni-stuttgart.de/")
    vis.register_modifier(DiffusionModelling, default=True)
    while vis.socket.connected:
        time.sleep(5)
    print("Connection lost, stopping...")
    sys.exit(0)
