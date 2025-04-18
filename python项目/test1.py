# -*- coding: utf-8 -*-

import tkinter as tk#可视化
from tkinter import messagebox
from random import randrange, choice
from collections import defaultdict

class GameField:
    def __init__(self, height=4, width=4, win=2048):
        self.height = height
        self.width = width
        self.win_value = win
        self.score = 0
        self.highscore = 0
        self.field = None
        self.reset()
        
    def reset(self):
        if self.score > self.highscore:
            self.highscore = self.score
        self.score = 0
        self.field = [[0 for i in range(self.width)] for j in range(self.height)]
        self.spawn()
        self.spawn()
    
    def spawn(self):
        new_element = 4 if randrange(100) > 89 else 2
        empty_cells = [(i, j) for i in range(self.height) for j in range(self.width) if self.field[i][j] == 0]
        if empty_cells:
            (i, j) = choice(empty_cells)
            self.field[i][j] = new_element
    
    def transpose(self, field):
        return [list(row) for row in zip(*field)]
    
    def invert(self, field):
        return [row[::-1] for row in field]
    
    def move(self, direction):
        def move_row_left(row):
            def tighten(row):  # squeeze non-zero elements together
                new_row = [i for i in row if i != 0]
                new_row += [0 for i in range(len(row) - len(new_row))]
                return new_row

            def merge(row):
                pair = False
                new_row = []
                for i in range(len(row)):
                    if pair:
                        new_row.append(2 * row[i])
                        self.score += 2 * row[i]
                        pair = False
                    else:
                        if i + 1 < len(row) and row[i] == row[i + 1]:
                            pair = True
                            new_row.append(0)
                        else:
                            new_row.append(row[i])
                assert len(new_row) == len(row)
                return new_row

            return tighten(merge(tighten(row)))

        moves = {}
        moves['Left'] = lambda field: [move_row_left(row) for row in field]
        moves['Right'] = lambda field: self.invert(moves['Left'](self.invert(field)))
        moves['Up'] = lambda field: self.transpose(moves['Left'](self.transpose(field)))
        moves['Down'] = lambda field: self.transpose(moves['Right'](self.transpose(field)))

        if direction in moves:
            if self.move_is_possible(direction):
                self.field = moves[direction](self.field)
                self.spawn()
                return True
            else:
                return False
        return False
    
    def is_win(self):
        return any(any(i >= self.win_value for i in row) for row in self.field)

    def is_gameover(self):
        actions = ['Up', 'Left', 'Down', 'Right']
        return not any(self.move_is_possible(move) for move in actions)
    
    def move_is_possible(self, direction):
        def row_is_left_movable(row):
            def change(i):  # true if there'll be change in i-th tile
                if row[i] == 0 and row[i + 1] != 0:  # Move
                    return True
                if row[i] != 0 and row[i + 1] == row[i]:  # Merge
                    return True
                return False

            return any(change(i) for i in range(len(row) - 1))

        check = {}
        check['Left'] = lambda field: any(row_is_left_movable(row) for row in field)
        check['Right'] = lambda field: check['Left'](self.invert(field))
        check['Up'] = lambda field: check['Left'](self.transpose(field))
        check['Down'] = lambda field: check['Right'](self.transpose(field))

        if direction in check:
            return check[direction](self.field)
        else:
            return False


class Game2048(tk.Frame):
    CELL_COLORS = {
        0: "#CCC0B3",
        2: "#EEE4DA",
        4: "#EDE0C8",
        8: "#F2B179",
        16: "#F59563",
        32: "#F67C5F",
        64: "#F65E3B",
        128: "#EDCF72",
        256: "#EDCC61",
        512: "#EDC850",
        1024: "#EDC53F",
        2048: "#EDC22E",
        4096: "#3C3A32",
    }
    
    CELL_TEXT_COLORS = {
        0: "#CCC0B3",
        2: "#776E65",
        4: "#776E65",
        8: "#F9F6F2",
        16: "#F9F6F2",
        32: "#F9F6F2",
        64: "#F9F6F2",
        128: "#F9F6F2",
        256: "#F9F6F2",
        512: "#F9F6F2",
        1024: "#F9F6F2",
        2048: "#F9F6F2",
        4096: "#F9F6F2",
    }
    
    FONT = ("Verdana", 24, "bold")
    SCORE_FONT = ("Verdana", 16)
    CELL_SIZE = 80
    PADDING = 10
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("2048 Game")
        self.grid()
        
        # 创建游戏逻辑类实例
        self.game_field = GameField(win=32)
        
        # 创建UI组件
        self.create_widgets()
        
        # 绑定按键事件
        self.bind_keys()
        
        # 绘制初始游戏界面
        self.update_ui()
        
    def create_widgets(self):
        # 创建分数标签
        self.score_frame = tk.Frame(self)
        self.score_frame.grid(row=0, column=0, padx=10, pady=10)
        
        self.score_label = tk.Label(self.score_frame, text="分数: 0", font=self.SCORE_FONT)
        self.score_label.grid(row=0, column=0, padx=10)
        
        self.highscore_label = tk.Label(self.score_frame, text="最高分: 0", font=self.SCORE_FONT)
        self.highscore_label.grid(row=0, column=1, padx=10)
        
        # 创建游戏棋盘
        self.board_frame = tk.Frame(
            self, 
            bg="#BBADA0", 
            width=self.CELL_SIZE * 4 + self.PADDING * 5,
            height=self.CELL_SIZE * 4 + self.PADDING * 5
        )
        self.board_frame.grid(row=1, column=0, padx=10, pady=10)
        self.board_frame.grid_propagate(False)
        
        # 创建单元格
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = tk.Frame(
                    self.board_frame,
                    width=self.CELL_SIZE,
                    height=self.CELL_SIZE,
                    bg="#CCC0B3"
                )
                cell_frame.grid(
                    row=i, 
                    column=j, 
                    padx=self.PADDING, 
                    pady=self.PADDING
                )
                cell_frame.grid_propagate(False)
                
                cell_label = tk.Label(
                    cell_frame,
                    text="",
                    font=self.FONT,
                    bg="#CCC0B3",
                    justify=tk.CENTER,
                    width=4,
                    height=2
                )
                cell_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                
                row.append(cell_label)
            self.cells.append(row)
        
        # 添加重新开始按钮
        self.restart_button = tk.Button(self, text="重新开始", command=self.restart_game)
        self.restart_button.grid(row=2, column=0, pady=10)
        
    def bind_keys(self):
        self.master.bind("<Up>", lambda event: self.move("Up"))
        self.master.bind("<Down>", lambda event: self.move("Down"))
        self.master.bind("<Left>", lambda event: self.move("Left"))
        self.master.bind("<Right>", lambda event: self.move("Right"))
        self.master.bind("w", lambda event: self.move("Up"))
        self.master.bind("s", lambda event: self.move("Down"))
        self.master.bind("a", lambda event: self.move("Left"))
        self.master.bind("d", lambda event: self.move("Right"))
        self.master.bind("r", lambda event: self.restart_game())
        self.master.bind("q", lambda event: self.master.destroy())
        
    def update_ui(self):
        # 更新分数显示
        self.score_label.config(text=f"分数: {self.game_field.score}")
        self.highscore_label.config(text=f"最高分: {self.game_field.highscore}")
        
        # 更新棋盘显示
        for i in range(4):
            for j in range(4):
                value = self.game_field.field[i][j]
                cell = self.cells[i][j]
                
                # 设置颜色和文本
                if value == 0:
                    cell.config(text="", bg=self.CELL_COLORS[0])
                else:
                    cell.config(
                        text=str(value), 
                        bg=self.CELL_COLORS.get(value, self.CELL_COLORS[4096]),
                        fg=self.CELL_TEXT_COLORS.get(value, "#F9F6F2")
                    )
        
        # 检查游戏状态
        if self.game_field.is_win():
            messagebox.showinfo("恭喜", "你赢了！")
            self.restart_game()
        elif self.game_field.is_gameover():
            messagebox.showinfo("游戏结束", f"游戏结束！\n你的分数: {self.game_field.score}")
            self.restart_game()
                
    def move(self, direction):
        if self.game_field.move(direction):
            self.update_ui()
        
    def restart_game(self):
        self.game_field.reset()
        self.update_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = Game2048(root)
    app.mainloop()