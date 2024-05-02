"""
# OOP Program 1
# Predicting Forest Fires Using PyTorch on codeprojects
import intel_extension_for_pytorch as ipex
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

num_physical_cores = psutil.cpu_count(logical=False)
data_dir = pathlib.Path("./data/output/")
TRAIN_DIR = data_dir / "train"
VALID_DIR = data_dir / "val"

img_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats)]
    )
}


class FireFinder(nn.Module):

    def __init__(self, backbone=18, simple=True, dropout= .4):
        super(FireFinder, self).__init__()
        backbones = {
            18: models.resnet18,
        }
        fc = nn.Sequential(
            nn.Linear(self.network.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2))


class Trainer:

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.ipx = ipx
        self.epochs = epochs
        if isinstance(optimizer, torch.optim.Adam):
            self.lr = 2e-3
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

        def train(self):
            self.model.train()
            t_epoch_loss, t_epoch_acc = 0.0, 0.0
            start = time.time()
            for inputs, labels in tqdm(train_dataloader, desc="tr loop"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.ipx:
                    inputs = inputs.to(memory_format=torch.channels_last)
                self.optimizer.zero_grad()
                loss, acc = self.forward_pass(inputs, labels)
                loss.backward()
                self.optimizer.step()
                t_epoch_loss += loss.item()
                t_epoch_acc += acc.item()
            return (t_epoch_loss, t_epoch_acc)

        def _to_ipx(self):
            self.model.train()
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model, self.optimizer = ipex.optimize(
                self.model, optimizer=self.optimizer, dtype=torch.float32
            )

epochs = 20
ipx = True
dropout = .33
lr = .02

torch.set_num_threads(num_physical_cores)
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

start = time.time()
model = FireFinder(simple=simple, dropout= dropout)

trainer = Trainer(model, lr = lr, epochs=epochs, ipx=ipx)
tft = trainer.fine_tune(train_dataloader, valid_dataloader)

class ImageFolderWithPaths(datasets.ImageFolder):

    def infer(model, data_path: str):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*imagenet_stats)]
        )

        data = ImageFolderWithPaths(data_path, transform=transform)

        dataloader = DataLoader(data, batch_size=4)


images, yhats, img_paths = infer(model, data_path="./data//test/")
"""

"""
# sample 2: The triangle classification problem
class Triangle:
    def __init__(self, angle_a, angle_b, angle_c):
        self.angle_a = angle_a
        self.angle_b = angle_b
        self.angle_c = angle_c

    def classify_triangle(self):
        total_degrees = self.angle_a + self.angle_b + self.angle_c

        if total_degrees == 180:
            return "Euclidean Triangle"
        elif total_degrees > 180:
            return "Elliptical Triangle"
        else:
            return "Hyperbolic Triangle"

    def triangle_type(self):
        if self.angle_a == self.angle_b == self.angle_c:
            return "Equilateral"
        elif self.angle_a == self.angle_b or self.angle_a == self.angle_c or self.angle_b == self.angle_c:
            return "Isosceles"
        else:
            return "Scalene"

# Example usage:
angle_a = 29
angle_b = 60
angle_c = 90

my_triangle = Triangle(angle_a, angle_b, angle_c)
print(f"That is a {my_triangle.classify_triangle()} {my_triangle.triangle_type()} Triangle.")

"""

"""
sample -3
##
Author: Parisa Arbab
Date: Feb 15 2024
Statement:“I have not given or received any unauthorized assistance on this assignment.”
YouTube Link: https://youtu.be/3rMbnwBFsfA

Explained in video:
1. Show how you extend sixSidedDie when writing TenSidedDie and TwentySidedDie.
2. Show how you compose the cups class with the die classes.

##
import random

class SixSidedDie:

    def __init__(self):
        self.faceValue = 1

    def roll(self):
        self.faceValue = random.randint(1, 6)
        return self.faceValue

    def getFaceValue(self):
        return self.faceValue

    def __repr__(self):
        return f"SixSidedDie({self.faceValue})"
class TenSidedDie(SixSidedDie):

    def roll(self):
        self.faceValue = random.randint(1, 10)
        return self.faceValue

    def __repr__ (self):
        return f"TenSidedDie({self.faceValue})"

class TwentySidedDie(SixSidedDie):

    def roll(self):
        self.faceValue = random.randint(1, 20)
        return self.faceValue

    def __repr__(self):
        return f"TwentySidedDie({self.faceValue})"


class Cup:

    #instantiate of six,ten,twenty side
    def __init__(self, six=1, ten=1, twenty=1):

        self.dice = [SixSidedDie() for _ in range(six)] + \
                    [TenSidedDie() for _ in range(ten)] + \
                    [TwentySidedDie() for _ in range(twenty)]


#Q1 Extend: override roll() to adjust the range of possible values to fit ten and twenty sides
    def roll(self):

        return sum(die.roll() for die in self.dice)

    def getSum(self):
        return sum(die.getFaceValue() for die in self.dice)

    def __repr__(self):
        return f"Cup({', '.join(repr(die) for die in self.dice)})"



# Example Usage
if __name__ == "__main__":
    # Demonstrate using a six-sided die
    d = SixSidedDie()
    print(d.roll())  # Roll the die and print the result
    print(d.getFaceValue())  # Print the current face value of the die
    print(d)  # Print the string representation of the die

    # Demonstrate using a ten-sided die
    t = TenSidedDie()
    print(t.roll())  # Roll the die and print the result
    print(t)  # Print the string representation of the die

    # Demonstrate using a twenty-sided die
    tw = TwentySidedDie()
    print(tw.roll())  # Roll the die and print the result
    print(tw)  # Print the string representation of the die

    # Demonstrate using a cup with dice
    cup = Cup(1, 2, 1)  # Initialize a cup with 1 six-sided, 2 ten-sided, and 1 twenty-sided dice
    print(cup.roll())  # Roll all dice in the cup and print the total sum
    print(cup.getSum())  # Print the sum of the current face values of all dice in the cup
    print(cup)  # Print the string representation of the cup and its contents

"""

"""
## mine sweeper algorithm by 
mzdluo123
github
sample = 4

from PIL import Image, ImageDraw, ImageColor, ImageFont
from enum import Enum
import random
from typing import Tuple
from time import time

COLUMN_NAME = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class GameState(Enum):
    PREPARE = 1
    GAMING = 2
    WIN = 3
    FAIL = 4


class Cell:
    def __init__(self, is_mine: bool, row: int = 0, column: int = 0, is_mined: bool = False, is_marked: bool = False):
        self.is_mine = is_mine
        self.is_mined = is_mined
        self.is_marked = is_marked
        self.row = row
        self.column = column
        self.is_checked = False

    def __str__(self):
        return f"[Cell] is_mine:{self.is_mine} is_marked:{self.is_marked} is_mined:{self.is_mined}"


class MineSweeper:
    def __init__(self, row: int, column: int, mines: int):
        if row > 26 or column > 26:
            raise ValueError("暂不支持这么大的游戏盘")
        if mines >= row * column or mines == 0:
            raise ValueError("非法操作")
        if mines < column - 1 or mines < row - 1:
            raise ValueError("就不能来点难的吗")
        self.row = row
        self.column = column
        self.mines = mines
        self.start_time = time()
        self.actions = 0
        self.font = ImageFont.truetype("00TT.TTF", 40)
        self.panel = [[Cell(False, row=r, column=c) for c in range(column)] for r in range(row)]
        self.state = GameState.PREPARE

    def __str__(self):
        return f"[MineSweeper] {self.mines} in {self.row}*{self.column}"


    def draw_panel(self) -> Image.Image:
        start = time()
        img = Image.new("RGB", (80 * self.column, 80 * self.row), (255, 255, 255))
        self.__draw_split_line(img)
        self.__draw_cell_cover(img)
        self.__draw_cell(img)
        print(f"draw spend {time()-start}ms at {str(self)}")
        return img

    def __draw_split_line(self, img: Image.Image):
        draw = ImageDraw.Draw(img)
        for i in range(0, self.row):
            draw.line((0, i * 80, img.size[0], i * 80), fill=ImageColor.getrgb("black"))
        for i in range(0, self.column):
            draw.line((i * 80, 0, i * 80, img.size[1]), fill=ImageColor.getrgb("black"))

    def __draw_cell_cover(self, img: Image.Image):
        draw = ImageDraw.Draw(img)
        for i in range(0, self.row):
            for j in range(0, self.column):
                cell = self.panel[i][j]
                if self.state == GameState.FAIL and cell.is_mine:
                    draw.rectangle((j * 80 + 1, i * 80 + 1, (j + 1) * 80 - 1, (i + 1) * 80 - 1),
                                   fill=ImageColor.getrgb("red"))
                    continue
                if cell.is_marked:
                    draw.rectangle((j * 80 + 1, i * 80 + 1, (j + 1) * 80 - 1, (i + 1) * 80 - 1),
                                   fill=ImageColor.getrgb("blue"))
                    continue
                if not cell.is_mined:
                    draw.rectangle((j * 80 + 1, i * 80 + 1, (j + 1) * 80 - 1, (i + 1) * 80 - 1),
                                   fill=ImageColor.getrgb("gray"))

    def __draw_cell(self, img: Image.Image):
        draw = ImageDraw.Draw(img)
        for i in range(0, self.row):
            for j in range(0, self.column):
                cell = self.panel[i][j]
                if not cell.is_mined:
                    font_size = self.font.getsize("AA")
                    index = f"{COLUMN_NAME[i]}{COLUMN_NAME[j]}"
                    center = (80 * (j + 1) - (font_size[0] / 2) - 40, 80 * (i + 1) - 40 - (font_size[1] / 2))
                    draw.text(center, index, fill=ImageColor.getrgb("black"), font=self.font)
                else:
                    count = self.count_around(i, j)
                    if count == 0:
                        continue
                    font_size = self.font.getsize(str(count))
                    center = (80 * (j + 1) - (font_size[0] / 2) - 40, 80 * (i + 1) - 40 - (font_size[1] / 2))
                    draw.text(center, str(count), fill=self.__get_count_text_color(count), font=self.font)

    @staticmethod
    def __get_count_text_color(count):
        if count == 1:
            return ImageColor.getrgb("green")
        if count == 2:
            return ImageColor.getrgb("orange")
        if count == 3:
            return ImageColor.getrgb("red")
        if count == 4:
            return ImageColor.getrgb("darkred")
        return ImageColor.getrgb("black")

    def mine(self, row: int, column: int):
        if not self.__is_valid_location(row, column):
            raise ValueError("非法操作")
        start = time()
        cell = self.panel[row][column]
        if cell.is_mined:
            raise ValueError("你已经挖过这里了")
        cell.is_mined = True
        if self.state == GameState.PREPARE:
            self.__gen_mine()
        if self.state != GameState.GAMING:
            raise ValueError("游戏已结束")
        self.actions += 1
        if cell.is_mine:
            self.state = GameState.FAIL
            return
        self.__reset_check()
        self.__spread_not_mine(row, column)
        self.__win_check()
        print(f"mine spend {time()-start}ms at {str(self)}")

    def tag(self, row: int, column: int):
        cell = self.panel[row][column]
        start = time()
        if cell.is_mined:
            raise ValueError("你不能标记一个你挖开的地方")
        if self.state != GameState.GAMING and self.state != GameState.PREPARE:
            raise ValueError("游戏已结束")
        self.actions += 1
        if cell.is_marked:
            cell.is_marked = False
        else:
            cell.is_marked = True
        print(f"tag spend {time()-start}ms at {str(self)}")

    def __gen_mine(self):
        count = 0
        while count < self.mines:
            row = random.randint(0, self.row - 1)
            column = random.randint(0, self.column - 1)
            if self.panel[row][column].is_mine or self.panel[row][column].is_mined:
                continue
            self.panel[row][column].is_mine = True
            count += 1
        self.state = GameState.GAMING

    def __spread_not_mine(self, row: int, column):
        if not self.__is_valid_location(row, column):
            return
        cell = self.panel[row][column]
        if cell.is_checked:
            return
        if cell.is_mine:
            return
        cell.is_mined = True
        cell.is_checked = True
        count = self.count_around(row, column)
        if count > 0:
            return
        self.__spread_not_mine(row + 1, column)
        self.__spread_not_mine(row - 1, column)
        self.__spread_not_mine(row, column + 1)
        self.__spread_not_mine(row, column - 1)
        if count == 0:
            self.__spread_not_mine(row + 1, column + 1)
            self.__spread_not_mine(row - 1, column - 1)
            self.__spread_not_mine(row + 1, column - 1)
            self.__spread_not_mine(row - 1, column + 1)

    def __reset_check(self):
        for i in range(0, self.row):
            for j in range(0, self.column):
                self.panel[i][j].is_checked = False

    def __win_check(self):
        mined = 0
        for i in range(0, self.row):
            for j in range(0, self.column):
                if self.panel[i][j].is_mined:
                    mined += 1
        if mined == (self.column * self.row) - self.mines:
            self.state = GameState.WIN

    def count_around(self, row: int, column: int) -> int:
        count = 0
        for r in range(row - 1, row + 2):
            for c in range(column - 1, column + 2):
                if not self.__is_valid_location(r, c):
                    continue
                if self.panel[r][c].is_mine:
                    count += 1
        if self.panel[row][column].is_mine:
            count -= 1
        return count

    @staticmethod
    def parse_input(input_text: str) -> Tuple[int, int]:
        if len(input_text) != 2:
            raise ValueError("非法位置")
        return COLUMN_NAME.index(input_text[0].upper()), COLUMN_NAME.index(input_text[1].upper())

    def __is_valid_location(self, row: int, column: int) -> bool:
        if row > self.row - 1 or column > self.column - 1 or row < 0 or column < 0:
            return False
        return True


if __name__ == '__main__':
    mine = MineSweeper(25, 25, 25)
    mine.draw_panel().show()
    while True:
        try:
            location = MineSweeper.parse_input(input())
            mine.mine(location[0], location[1])
            mine.draw_panel().show()
            print(mine.state)
        except Exception as e:
            print(e)
"""

"""
## barber_shop by crazytieguy on github
url: https://github.com/Crazytieguy/concurrency-problems/blob/master/barber_shop.py

import asyncio
import random
import time
from asyncio import Event, Queue, QueueFull

START = time.time()


def log(name: str, message: str):
    now = time.time() - START
    print(f"{now:.3f} {name}: {message}")


class BarberShop:
    queue: Queue[Event]

    def __init__(self):
        self.queue = Queue(5)

    async def get_haircut(self, name: str):
        event = Event()
        try:
            self.queue.put_nowait(event)
        except QueueFull:
            log(name, "Room full, leaving")
            return False
        log(name, "Waiting for haircut")
        await event.wait()
        log(name, "Got haircut")

    async def run_barber(self):
        while True:
            customer = await self.queue.get()
            log("barber", "Giving haircut")
            await asyncio.sleep(1)
            customer.set()


async def customer(barber_shop: BarberShop, name: str):
    await asyncio.sleep(random.random() * 10)
    await barber_shop.get_haircut(name)


async def main():
    barber_shop = BarberShop()
    asyncio.create_task(barber_shop.run_barber())
    await asyncio.gather(*[customer(barber_shop, f"Cust-{i}") for i in range(20)])


if __name__ == "__main__":
    asyncio.run(main())
"""



"""
### spreadsheet_converter.py
by  Hermann-web
url:  https://github.com/Hermann-web/file-converter/blob/main/file-conv-framework/examples/spreadsheet_converter.py



import sys
from pathlib import Path

import pandas as pd

sys.path.append(".")

from file_conv_framework.base_converter import BaseConverter, ResolvedInputFile
from file_conv_framework.filetypes import FileType
from file_conv_framework.io_handler import FileReader, ListToCsvWriter


class SpreadsheetToPandasReader(FileReader):
    input_format = pd.DataFrame

    def _check_input_format(self, content: pd.DataFrame):
        return isinstance(content, pd.DataFrame)

    def _read_content(self, input_path: Path) -> pd.DataFrame:
        return pd.read_excel(input_path)


class XLXSToCSVConverter(BaseConverter):

    file_reader = SpreadsheetToPandasReader()
    file_writer = ListToCsvWriter()

    @classmethod
    def _get_supported_input_type(cls) -> FileType:
        return FileType.EXCEL

    @classmethod
    def _get_supported_output_type(cls) -> FileType:
        return FileType.CSV

    def _convert(self, df: pd.DataFrame):
        # Convert DataFrame to a list of lists
        data_as_list = df.values.tolist()

        # Insert column names as the first sublist
        data_as_list.insert(0, df.columns.tolist())

        return data_as_list


if __name__ == "__main__":
    input_file_path = "examples/data/example.xlsx"
    output_file_path = "examples/data/example.csv"

    input_file = ResolvedInputFile(input_file_path)
    output_file = ResolvedInputFile(output_file_path, add_suffix=True)

    converter = XLXSToCSVConverter(input_file, output_file)
    converter.convert()
"""

"""
GUi application 1



class Tree:
    def __init__(self):
        self.root = None
        self.nodes = ""
    def addnode(self,data):
        currnode = Node(data)
        if self.root is None:
            self.root = currnode
        else:
            parent = None
            ptr = self.root
            while ptr is not None:
                parent = ptr
                if int(currnode.data) < int(ptr.data):
                    ptr = ptr.left
                else:
                    ptr = ptr.right
            if int(currnode.data) < int(parent.data):
                parent.left = currnode
            else:
                parent.right = currnode
    def inorder(self,root):
            if root != None:
                self.inorder(root.left)
                self.nodes += root.data + " "
                self.inorder(root.right)
    def preorder(self,root):
            if root != None:
                self.nodes += root.data + " "
                self.preorder(root.left)
                self.preorder(root.right)
    def postorder(self,root):
            if root != None:
                self.postorder(root.left)
                self.postorder(root.right)
                self.nodes += root.data + " "
    def visualizetree(self,root):
        dot = graphviz.Digraph()
        dot.node(str(root.data))
        self.addedge(root,dot)
        dot.render("tree",format="png")
    def addedge(self,node,dot):
        if node.left:
            dot.node(str(node.left.data))
            dot.edge(str(node.data),str(node.left.data))
            self.addedge(node.left,dot)
        if node.right:
            dot.node(str(node.right.data))
            dot.edge(str(node.data),str(node.right.data))
            self.addedge(node.right,dot)

    def add():
        tree.addnode(txtvalue.get())
        tree.visualizetree(tree.root)
        img = ImageTk.PhotoImage(Image.open("tree.png"))
        lblimage.configure(image=img)
        lblimage.image = img

    def inorder():
        tree.inorder(tree.root)
        messagebox.showinfo("Inorder",tree.nodes)
        tree.nodes = ""

    def preorder():
        tree.preorder(tree.root)
        messagebox.showinfo("Preorder",tree.nodes)
        tree.nodes = ""

    def postorder():
        tree.postorder(tree.root)
        messagebox.showinfo("Postorder",tree.nodes)
        tree.nodes = ""

    def showimage(event):
        os.system("tree.png") if os.path.exists("tree.png") else None

if __name__ == "__main__":

    tree = Tree()
    root = Tk()
    root.title("Binary Search Tree")
    root.geometry("500x300")

    lblvalue = Label(root,text="Enter data: ")
    lblvalue.place(x=50,y=50,width=100)

    txtvalue = Entry(root)
    txtvalue.place(x=150,y=50,width=100)

    btnadd = Button(root,text="Add",command=add)
    btnadd.place(x=50,y=100,width=100)

    btninorder = Button(root,text="Inorder",command=inorder)
    btninorder.place(x=150,y=100,width=100)

    btnpreorder = Button(root,text="Preorder",command=preorder)
    btnpreorder.place(x=50,y=150,width=100)

    btnpostorder = Button(root,text="Postorder",command=postorder)
    btnpostorder.place(x=150,y=150,width=100)

    lblimage = Label(root)
    lblimage.bind("<Button-1>",showimage)
    lblimage.place(x=300,y=50,width=150,height=150)
    root.mainloop()

    if os.path.exists("tree.png"):
       os.remove("tree.png")
       os.remove("tree")

"""

"""
performance sensitive application-- 1

from __future__ import annotations
import io
import timeit
import numpy as np
from typing import Callable, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.typing as npt
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

# Use a simplified style for plotting
plt.style.use('seaborn-darkgrid')

# SI time units for scaling
si_time_units = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

# Determine the most appropriate time unit for display
def auto_time_unit(time_seconds: float) -> str:
    for unit, magnitude in si_time_units.items():
        if time_seconds >= magnitude:
            return unit
    return "s"

# Check for equality between benchmark outputs
def default_equality_check(a, b):
    return np.allclose(a, b) if isinstance(a, np.ndarray) else a == b

# Core class for storing benchmarking data
class PerfplotData:
    def __init__(self, n_range: List[int], timings: np.ndarray, labels: List[str], xlabel: str = "", title: str = ""):
        self.n_range = np.asarray(n_range)
        self.timings = timings
        self.labels = labels
        self.xlabel = xlabel
        self.title = title

    # Plotting function simplified
    def plot(self, log_scale: bool = False, relative_to: int = None):
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(self.labels):
            y_data = self.timings[:, i] / si_time_units[auto_time_unit(min(self.timings[:, i]))] if relative_to is None else self.timings[:, i] / self.timings[:, relative_to]
            plt.plot(self.n_range, y_data, label=label)

        plt.xlabel(self.xlabel)
        plt.ylabel('Runtime [s]' if relative_to is None else f'Runtime relative to {self.labels[relative_to]}')
        plt.title(self.title)
        plt.yscale('log' if log_scale else 'linear')
        plt.legend()
        plt.grid(True)

    def show(self):
        plt.show()

# Benchmark class simplified
class Bench:
    def __init__(self, kernels: List[Callable], n_range: List[int], setup: Callable = None):
        self.kernels = kernels
        self.n_range = n_range
        self.setup = setup

    def run(self):
        timings = np.zeros((len(self.n_range), len(self.kernels)))
        for i, n in enumerate(self.n_range):
            data = self.setup(n) if self.setup else n
            for j, kernel in enumerate(self.kernels):
                timer = timeit.Timer(lambda: kernel(data))
                timings[i, j] = min(timer.repeat(3, 1))
        return PerfplotData(self.n_range, timings, [k.__name__ for k in self.kernels], "Input Size", "Benchmark Results")

# Example usage
if __name__ == "__main__":
    def setup(n):
        # Setup function for benchmarking
        return np.random.rand(n), np.random.rand(n)

    def kernel1(data):
        return np.fft.fft(data)

    def kernel2(data):
        return np.sort(data)

    bench = Bench([kernel1, kernel2], [1000, 10000, 100000], setup)
    data = bench.run()
    data.plot(log_scale=True)
    data.show()
"""




####################################################################



"""
Metahueristic hybrid oc the CS-SA algorithm itself

import random
import math
from scipy.special import gamma
import numpy as np
import ast
import time
import astor
# Define Cuckoo Search Algorithm Components


class CuckooSearch:
    def __init__(self, population_size, nest_size, pa, beta=3.0):
        
        self.population_size = population_size
        self.nest_size = nest_size
        self.pa = pa
        self.beta = beta
        self.nests = np.random.rand(self.population_size, self.nest_size)
        self.best_nest = None
        self.best_fitness = float('inf')
        self.previous_best_fitness = float('inf')  # Initialize previous_best_fitness


    def generate_initial_population(self):
        
        self.nests = np.random.rand(self.population_size, self.nest_size)

    def get_fitness(self, nest):
        
        return self.fitness(nest)

    def normalize_and_weight(self, score, max_score, weight, transform_type='linear'):
        normalized_score = min(score / max_score, 1.0)
        weighted_score = weight * normalized_score

        if transform_type == 'logarithmic':
            # Apply logarithmic transformation, ensuring score is always > 0
            weighted_score = np.log(weighted_score + 1)
        elif transform_type == 'power':
            # Apply power transformation with a power of 2 for demonstration
            weighted_score = weighted_score ** 2
        elif transform_type == 'exponential':
            # Apply exponential decay transformation with adjustment
            weighted_score = np.exp(-weighted_score) * weight

        return weighted_score

    def fitness(self, test_case):
        score = 0
        weights = {
            'inheritance': 2,
            'polymorphism': 2,
            'encapsulation': 1.5,
            'method_override': 1.5,
            'performance': 2.5,
            'error_handling': 3
        }
        max_scores = {
            'inheritance': 5,
            'polymorphism': 10,
            'encapsulation': 5,
            'method_override': 5,
            'performance': 100,
            'error_handling': 10
        }
        transform_types = {
            'inheritance': 'logarithmic',
            'polymorphism': 'power',
            'encapsulation': 'exponential',
            'method_override': 'logarithmic',
            'performance': 'power',
            'error_handling': 'exponential'
        }

        for test_type, weight in weights.items():
            test_score = test_case.get(f'{test_type}_score', 0)
            transform_type = transform_types[test_type]
            score += self.normalize_and_weight(test_score, max_scores[test_type], weight, transform_type)

        fault_score = self.evaluate_faults(test_case)
        # Consider using a non-linear transformation for the fault score as well
        fault_score_transformed = np.exp(-fault_score)
        score += fault_score_transformed

        # Final transformation to ensure non-linear differentiation
        score = 1 / (1 + score) if score != 0 else float('inf')
        # Further penalize with an exponential function
        score = np.exp(-10 * score)

        return score


    def evaluate_faults(self, test_case, weights=None):
        
        fault_score = 0

        # Safe retrieval of values, defaulting to 0 if not found or if None
        def safe_get(key, default=0):
            value = test_case.get(key, default)
            return default if value is None else value

        # Example penalty for not handling errors in negative tests
        if safe_get('is_negative', False) and not safe_get('error_handled', False):
            fault_score += 10

        # Polymorphism fault: Insufficient testing of polymorphic methods
        if test_case.get('test_type') == 'polymorphism' and test_case.get('num_polymorphic_methods', 0) < 3:
            fault_score += 10

        # Inheritance fault: Shallow inheritance testing
        if test_case.get('test_type') == 'inheritance' and test_case.get('inheritance_depth', 0) < 2:
            fault_score += 8

        # Encapsulation fault: Poor validation of encapsulation
        encapsulation_score = test_case.get('encapsulation_validation_score', 0)
        if test_case.get('test_type') == 'encapsulation' and encapsulation_score < 0.5:
            # Assuming a score below 0.5 indicates poor encapsulation validation
            fault_score += 12

        # Performance fault: Unjustifiably high execution time
        execution_time = test_case.get('execution_time', 100)
        if test_case.get('test_type') == 'performance' and execution_time > 50:
            # Assuming performance tests should not exceed 50ms execution time under normal conditions
            fault_score += 6

        # Error handling fault: Lack of error handling in negative test cases
        if test_case.get('is_negative', False) and not test_case.get('error_handled', False):
            fault_score += 14

        # Consider weights for fault detection, if weights are provided
        if weights:
            fault_score *= weights.get('fault_detection', 1)  # Default weight is 1 if not specified

        # Concurrency issues penalty
        concurrency_issues_tested = test_case.get('concurrency_issues_tested', 0)
        if concurrency_issues_tested < 2:  # Assuming less than 2 tests for concurrency issues indicate insufficient testing
            fault_score += 12

        # Resource utilization penalty
        resource_utilization_score = test_case.get('resource_utilization_score', 0)
        if resource_utilization_score < 5:  # Assuming scores below 5 indicate poor resource utilization testing
            fault_score += 6

        # Security vulnerability checks penalty
        security_checks = test_case.get('security_checks', 0)
        if security_checks < 2:  # Assuming less than 2 security checks indicates insufficient security testing
            fault_score += 14

        # Complexity penalty: Increase fault score for low complexity or coverage
        complexity_score = test_case.get('complexity_score', 0)
        if complexity_score < 3:  # Assuming a complexity score below 3 indicates low complexity or coverage
            fault_score += 10

        # Code style and best practices penalty
        code_style_score = test_case.get('code_style_score', 0)
        if code_style_score < 5:  # Assuming scores below 5 indicate poor adherence to code style or best practices
            fault_score += 8

        return fault_score

    def levy_flight(self):
        
        sigma = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                 (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, size=self.nest_size)
        v = np.random.normal(0, 1, size=self.nest_size)
        step = u / np.abs(v) ** (1 / self.beta)

        return step

    def update_nests(self, generation):
        improvement_threshold = 0.01  # Example threshold
        if generation > 0 and (self.best_fitness - self.previous_best_fitness) / self.previous_best_fitness < improvement_threshold:
            self.pa *= 1.1  # Increase pa by 10%
        for i, nest in enumerate(self.nests):
            step_size = self.levy_flight()
            new_nest = nest + step_size * np.random.rand(*nest.shape)
            new_fitness = self.get_fitness(new_nest)
            if new_fitness < self.get_fitness(nest):
                self.nests[i] = new_nest
                if new_fitness < self.best_fitness:
                    self.best_nest = new_nest
                    self.best_fitness = new_fitness
            elif np.random.rand() < self.pa:
                self.nests[i] = np.random.rand(*self.nests[i].shape)
        self.previous_best_fitness = self.best_fitness

        self.abandon_worse_nests()


    def abandon_worse_nests(self):
        for i, _ in enumerate(self.nests):
            if np.random.rand() < self.pa:
                self.nests[i] = np.random.rand(*self.nests[i].shape)

    def find_best_solution(self):
        # The best solution is already found during the nest updates
        return self.best_nest, self.best_fitness
# Define Simulated Annealing Algorithm Components
class SimulatedAnnealing:
    def __init__(self, initial_temperature, cooling_rate):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def initial_solution(self, cuckoo_search):
        # Use the best solution from Cuckoo Search as the initial solution for Simulated Annealing
        self.current_solution, _ = cuckoo_search.find_best_solution()
        return self.current_solution

    def get_neighbour(self, solution):
        neighbour = solution.copy()
        tweak_index = random.randint(0, len(neighbour) - 1)

        # Instead of doubling the value, let's make a smaller change
        change = random.uniform(-0.1, 0.1) * neighbour[tweak_index]
        neighbour[tweak_index] += change

        return neighbour

    def acceptance_probability(self, old_cost, new_cost):
        # Calculate the acceptance probability
        if new_cost < old_cost:
            return 1.0
        else:
            return math.exp((old_cost - new_cost) / self.temperature)

    def anneal(self, fitness_function):
        successful_attempts = 0
        attempts_since_last_success = 0
        while self.temperature > 1:
            new_solution = self.get_neighbour(self.current_solution)
            new_cost = fitness_function(new_solution)
            if self.acceptance_probability(self.current_cost, new_cost) > random.random():
                self.current_solution = new_solution
                self.current_cost = new_cost
                successful_attempts += 1
                attempts_since_last_success = 0
            else:
                attempts_since_last_success += 1
            if attempts_since_last_success > 10:  # If no success in the last 10 attempts, increase cooling rate
                self.cooling_rate *= 1.05
            elif successful_attempts % 10 == 0:  # Every 10 successful attempts, decrease the cooling rate
                self.cooling_rate /= 1.05
            self.temperature *= 1 - self.cooling_rate
        return self.current_solution

# Hybrid Algorithm Integration
class HybridAlgorithm:
    def __init__(self, user_input_code):
        self.cuckoo = CuckooSearch(population_size=14, nest_size=20, pa=1.0, beta=3.0)
        self.annealing = SimulatedAnnealing(initial_temperature=10010, cooling_rate=0.0015)
        self.user_input_code = user_input_code

    def generate_test_cases(self, program):
        analysis = ProgramAnalysis(program)
        analysis.extract_structure()

        return analysis.identify_test_scenarios()

    def hybrid_optimization(self, test_cases):
        # Optimize test cases using Cuckoo Search
        self.cuckoo.generate_initial_population()
        best_fitness_per_generation = []
        last_improvement_generation = 0
        best_ever_fitness = float('inf')
        NUM_GENERATIONS = 50  # Define the number of generations
        CONVERGENCE_THRESHOLD = 10  # Define convergence threshold

        for generation in range(NUM_GENERATIONS):
            self.cuckoo.update_nests(generation)
            _, current_best_fitness = self.cuckoo.find_best_solution()

            # Update the best fitness per generation
            best_fitness_per_generation.append(current_best_fitness)

            # Update best ever fitness and last improvement generation
            if current_best_fitness < best_ever_fitness:
                best_ever_fitness = current_best_fitness
                last_improvement_generation = generation

            # Check for convergence
            if generation - last_improvement_generation > CONVERGENCE_THRESHOLD:
                break

        # Further optimization with Simulated Annealing
        self.annealing.initial_solution(self.cuckoo.best_nest)
        optimized_solution = self.annealing.anneal(self.cuckoo.get_fitness)

        convergence_generation = last_improvement_generation
        return optimized_solution, best_fitness_per_generation, convergence_generation

    def evaluate_test_cases(self, test_cases):
        optimized_solution, fitness_data, convergence_gen = self.hybrid_optimization(test_cases)
        score = self.cuckoo.get_fitness(optimized_solution)
        return score, optimized_solution, fitness_data, convergence_gen






class ProgramAnalysis:
    def __init__(self, program_code):
        # Correct the code first
        self.program_code = self.correct_multiline_strings(program_code)
        self.class_inheritance = {}
        self.class_methods = {}
        self.coverage = set()

    def correct_multiline_strings(self, code):
        
        tree = ast.parse(code)
        self.visit(tree)
        corrected_code = astor.to_source(tree)
        return corrected_code

    def visit(self, node):
        
        for field, value in ast.iter_fields(node):
            if isinstance(value, str):
                setattr(node, field, self.correct_multiline_string(value))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.Str):
                        value[i] = ast.Str(s=self.correct_multiline_string(item.s))
                    elif isinstance(item, ast.JoinedStr):
                        self.correct_f_string(item)
                    else:
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    @staticmethod
    def correct_multiline_string(string):

        return string.replace('', ' ')

    def correct_f_string(self, node):
        
        for part in node.values:
            if isinstance(part, ast.Str):
                part.s = self.correct_multiline_string(part.s)
            else:
                self.visit(part)


    def extract_structure(self):
        try:
            tree = ast.parse(self.program_code)
            self._parse_ast(tree)
        except SyntaxError as e:
            raise ValueError(f"Error parsing the Python code: {e}")

    def _parse_ast(self, node):
        for item in ast.walk(node):
            if isinstance(item, ast.ClassDef):
                self._process_class_definition(item)

    def _process_class_definition(self, class_node):
        class_name = class_node.name
        self.class_inheritance[class_name] = [base.id for base in class_node.bases if isinstance(base, ast.Name)]
        self.class_methods[class_name] = {method.name: self._parse_method_parameters(method)
                                          for method in class_node.body if isinstance(method, ast.FunctionDef)}
        # Detect and process method overrides
        self._detect_method_overrides(class_node)

    def _detect_method_overrides(self, class_node):
        class_name = class_node.name
        for base in class_node.bases:
            base_name = base.id if isinstance(base, ast.Name) else None
            if base_name and base_name in self.class_methods:
                self._compare_methods_for_override(class_name, base_name)

    def _compare_methods_for_override(self, class_name, base_name):
        class_methods = self.class_methods[class_name]
        base_methods = self.class_methods[base_name]
        for method in class_methods:
            if method in base_methods:
                print(f"Method {method} in class {class_name} overrides method from {base_name}")

    def _parse_method_parameters(self, method_node):
        return {arg.arg: self._get_default_value(method_node, arg) for arg in method_node.args.args}

    def _get_default_value(self, function_node, arg):
        defaults_index = len(function_node.args.args) - len(function_node.args.defaults)
        arg_index = function_node.args.args.index(arg)
        if arg_index >= defaults_index:
            return repr(function_node.args.defaults[arg_index - defaults_index])
        return None

    def identify_test_inputs(self, class_name, method_name):
        method_info = self.class_methods.get(class_name, {}).get(method_name, {})
        positive_test_inputs = {}
        negative_test_inputs = {}

        for param, default in method_info.items():
            if default is not None:
                positive_test_inputs[param] = default  # Use the default value
                negative_test_inputs[param] = self.generate_negative_input(default)
            else:
                # Placeholder logic for generating test inputs
                positive_test_inputs[param] = "valid_test_value"
                negative_test_inputs[param] = "invalid_test_value"

        return positive_test_inputs, negative_test_inputs

    def generate_negative_input(self, default_value):
        # Improved logic for negative input generation based on type
        if isinstance(default_value, int):
            return -default_value  # Example: Use negative value for int
        elif isinstance(default_value, str) and default_value:
            return ""  # Example: Empty string for non-empty default
        # Extend logic for other types as necessary
        return "invalid_test_value"

    def identify_test_scenarios(self):
        test_cases = []
        for class_name, methods in self.class_methods.items():
            # Enhanced logic to include critical methods and exclude display methods
            critical_methods = [method for method in methods if not method.startswith('display_')]
            for method in critical_methods:
                test_cases.append(self.generate_test_case_for_method(class_name, method))
                self.mark_covered(class_name, method)  # Mark critical methods as covered
        return test_cases

    def generate_test_case_for_method(self, class_name, method):
        inputs, _ = self.identify_test_inputs(class_name, method)
        expected_output = self.define_expected_outputs(class_name, method)
        # Adding mock scores for fitness evaluation
        test_scores = {
            'inheritance_score': random.randint(1, 5),
            'polymorphism_score': random.randint(1, 10),
            'encapsulation_score': random.randint(1, 5),
            'method_override_score': random.randint(1, 5),
            'performance_score': random.randint(1, 100),
            'error_handling_score': random.randint(1, 10)
        }
        return {
            'test_type': 'functional',
            'class_name': class_name,
            'method_name': method,
            'inputs': inputs,
            'expected_output': expected_output,
            **test_scores  # Merge test scores into the test case dictionary
        }

    def define_expected_outputs(self, class_name, method_name):
        # Enhanced output definitions with specific checks for certain method patterns
        if method_name.startswith("is") or method_name.startswith("has"):
            return True  # Expect boolean true for methods starting with 'is' or 'has'
        # Extend with more heuristics as needed
        return "Specific expected output based on method functionality"

    def mark_covered(self, class_name, method_name):
        self.coverage.add((class_name, method_name))

    def generate_coverage_report(self):
        # Assuming all methods in self.class_methods should be covered
        all_methods = set()
        for class_name, methods in self.class_methods.items():
            for method in methods:
                all_methods.add((class_name, method))

        covered_methods = self.coverage
        uncovered_methods = all_methods - covered_methods

        coverage_percentage = (len(covered_methods) / len(all_methods) * 100) if all_methods else 0

        # Convert the sets of tuples into a list of method names
        covered_methods_list = [f'{class_name}.{method}' for class_name, method in covered_methods]
        uncovered_methods_list = [f'{class_name}.{method}' for class_name, method in uncovered_methods]

        return {
            "coverage_percentage": coverage_percentage,
            "covered_methods": covered_methods_list,  # This should be a list
            "uncovered_methods": uncovered_methods_list,  # This should also be a list
        }

    def collect_metrics(algorithm_name, execution_time, best_fitness, coverage_percentage, mode='w',
                        filename='new_hybrid_algorithm_metrics.txt'):
        metrics_content = (
            f"Algorithm: {algorithm_name}",
            f"Execution Time: {execution_time}",
            f"Best Fitness: {best_fitness}",
            f"Coverage: {coverage_percentage}"
        )
    
        with open(filename, mode) as f:
            f.write(metrics_content)


    # Main Function to Run the Tool
    def main():
        # Example usage of the tool
        program_code = Test_code
    
        start_time = time.perf_counter()
        analysis = ProgramAnalysis(program_code)
        analysis.extract_structure()
        test_scenarios = analysis.identify_test_scenarios()
        hybrid_tool = HybridAlgorithm(user_input_code=program_code)
    
        # Generate test cases
        test_cases = hybrid_tool.generate_test_cases(program_code)
    
        best_fitness = float('inf')  # Initialize best_fitness with the highest possible value
        best_test_case = None  # To keep track of the test case with the best fitness
    
        print("Generated Test Cases and Fitness Scores:")
        for test_case in test_cases:
            fitness_score = hybrid_tool.cuckoo.get_fitness(test_case)
            if fitness_score < best_fitness:
                best_fitness = fitness_score
                best_test_case = test_case
            print(f"Test Case: {test_case}, Fitness Score: {fitness_score}")
    
        print(f"Best fitness is {best_fitness}")  # Move this print statement outside the loop
    
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"The time taken to generate test cases was {execution_time}")
        # Generate and print the coverage report
        coverage_report = analysis.generate_coverage_report()
        print("Coverage Report:")
        print(f"Coverage Percentage: {coverage_report['coverage_percentage']:.2f}%")
        print("Covered Methods:", coverage_report['covered_methods'])
        print("Uncovered Methods:", coverage_report['uncovered_methods'])
    
        collect_metrics("HybridAlgorithm", execution_time, best_fitness, coverage_report['coverage_percentage'])
    
    
    if __name__ == "__main__":
        main()


"""