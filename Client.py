import random
import socket
import time
import traceback
from typing import Callable, override

import pygame
import threading
import pickle

from numpy.f2py.auxfuncs import show
from sympy.plotting.series import ContourSeries

from Communication import TcpBySize

HOME = None
MATCHED = False

WIDTH = None
HEIGHT = None
RESOLUTION = None

game_surface = None
current_window = None
IS_TURN = True

LOBBY = None
CHAT = None
LOADING = None

PATHS = {
    "GameLobby" : r"LobbyPictures\GameLobby.png",
    "PlayHovered" : r"LobbyPictures\PlayHovered.png",
    "WatchHovered" : r"LobbyPictures\WatchHovered.png",
    "ScoreboardHovered" : r"LobbyPictures\ScoreboardHovered.png",
    "LoadingScreen" : r"Loading\Empty.png",
    "ChatScreen" : r"ChatPictures2\ChatRoom.png",
    "Choose1" : r"ChatPictures2\Choose1.png",
    "HumanHovered" : r"ChatPictures\HumanHovered.png",
    "BotHovered" : r"ChatPictures\BotHovered.png",
    "BackToLobby" : r"ChatPictures\BackToLobby.png",
    "BackToLobbyHovered" : r"ChatPictures\BackToLobbyHovered.png",
    "Continue" : r"ChatPictures\Continue.png",
    "ContinueHovered" : r"ChatPictures\ContinueHovered.png",
    "YouWon" : r"ChatPictures2\YouWon.png",
    "YouLost" : r"ChatPictures\YouLost.png",
    "Scoreboard" : r"LobbyPictures\ScoreBoardLobby.png",
    "HomeHovered" : r"LobbyPictures\HomeHovered.png",
    "RightHovered" : r"LobbyPictures\RightHovered.png",
    "LeftHovered" : r"LobbyPictures\LeftHovered.png"
}

lobby_lines = [
    "Maybe just one more...",
    "Feeling lucky?",
    "Number 1 yet?",
    "Good morning!",
    "Good evening!",
    "Good night!",
    "BOO",
    "Still here?",
    "Last time was a fluke.",
    "Go touch some grass first.",
    "Sleep is for the weak.",
    "Almost famous.",
    "Here we go again",
    "Sleep? In this economy?",
    "Skill issue",
    "Sharks are just wet dogs",
    "Your fridge is running",
    "Sir this is a Wendy's",
    "Never again (until next time)",
    "Eize mazal sheasiti clutch"
]

default_font = "PressStart2P-Regular.ttf"

IMAGES = {}

class Window:
    def __init__(self, background_picture, comms):
        self.background_picture = background_picture
        self.buttons = {}
        self.img_buttons = {}
        self.hovered = False
        self.mouse_clicked = False
        self.mouse_pos = None
        self.comms = comms
        self.methods = {
            "ERROR" : self.error
        }

    def display(self):
        global WIDTH, HEIGHT

        Controls.display_image(self.background_picture, (0, 0))
        self.show_img_buttons()

    def show_img_buttons(self):
        for name, image in self.img_buttons.items():
            button = self.buttons[name]
            Controls.display_image(image,
                            (WIDTH // button.ratios[0], HEIGHT // button.ratios[1]),
                                   scale_ratio=(button.ratios[2], button.ratios[3]))
            self.refresh_buttons(pygame.display.get_window_size())

    def add_button(self, name, ratios, func, image=None, hovered=None, args=None):
        if image:
            self.buttons[name] = Button(ratios, func, args, hovered, True)
            self.img_buttons[name] = image
        else:
            self.buttons[name] = Button(ratios, func, args, hovered, False)

    def refresh_buttons(self, screen_size: tuple):
        [button.refresh(screen_size) for button in self.buttons.values()]

    def event_window_resize(self, screen_size: tuple):
        Controls.scale()
        self.refresh_buttons(screen_size)

    def event_mouse_motion(self, event: pygame.event.Event):
        pass

    def event_mouse_down(self, event: pygame.event.Event):
        pass

    def event_mouse_up(self, event: pygame.event.Event):
        pass

    def event_key_pressed(self, event: pygame.event.Event):
        pass

    def event_mousewheel(self, event: pygame.event.Event):
        pass

    def extra_default_mechanics(self):
        data = None

        try:
            data = pickle.loads(self.comms.recv_by_size())
        except socket.timeout:
            return data
        except Exception:
            print(traceback.format_exc())

        if data:
            return data

        return {"code" : "ERROR", "description" : "Server closed unexpectedly"}

    @staticmethod
    def error(data):
        print("An error occurred")

class LobbyWindow(Window):
    def __init__(self, background_picture, comms):
        super().__init__(background_picture, comms)
        self.methods["SCORE_PLACERES"] = self.set_score

        data = {"code" : "SCORE_PLACEREQ"}
        comms.send_with_size(pickle.dumps(data))

        self.text_font = pygame.font.Font(default_font, 80)
        self.text_font2 = pygame.font.Font(default_font, 140)
        self.score = "0"
        self.placement = "0"

    def set_score(self, data):
        self.score = str(data["score"]) + "©"
        self.placement = "Rank #" + str(data["placement"] + 1)

        self.show_score(self.score, self.placement)
        self.show_lobby_line()


    def show_lobby_line(self):
        global lobby_lines

        line = random.choice(lobby_lines)

        f = pygame.font.Font(default_font, 25)

        line_surface = f.render(line, True, (255, 43, 248))
        line_surface = pygame.transform.rotate(line_surface, 30)

        x, y = WIDTH // 1.65, HEIGHT // 2.15

        x, y = Controls.center((x, y), (line_surface.get_size()), (WIDTH // 11.29, HEIGHT // 7.44))
        game_surface.blit(line_surface, (x, y))

        Controls.scale()


    def show_score(self, score, placement):
        score_surface = self.text_font2.render(score, True, (227, 189, 18))
        placement_surface = self.text_font.render(placement, True, (255, 255, 255))

        x = WIDTH // 3.39
        y = HEIGHT // 2.84
        window_size = WIDTH // 2.43, HEIGHT // 4.07
        obj_size = score_surface.get_size()

        x, y = Controls.center((x, y), obj_size, window_size)
        game_surface.blit((score_surface), (x, y))

        x = WIDTH // 3.33
        y = HEIGHT // 5.68
        window_size = WIDTH // 2.49, HEIGHT // 7.2
        obj_size = placement_surface.get_size()

        x, y = Controls.center((x, y), obj_size, window_size)
        game_surface.blit(placement_surface, (x, y))

        Controls.scale()

    @override
    def event_mouse_motion(self, event: pygame.event.Event):
        pos = event.pos[0], event.pos[1]
        for button in self.buttons.values():
            if button.is_pressed(pos):
                if button.hovered:
                    if self.hovered:
                        break

                    cords = (0,0)
                    ratios = (1, 1)
                    if button.is_img:
                        cords = (WIDTH // button.ratios[0], HEIGHT // button.ratios[1])
                        ratios = (button.ratios[2], button.ratios[3])

                    Controls.display_image(button.hovered, cords, scale_ratio=ratios)
                    self.hovered = True
                    break
        else:
            if self.hovered:
                self.hovered = False
                Controls.display_image(PATHS["GameLobby"], (0, 0))
                self.show_score(self.score, self.placement)
                self.show_lobby_line()

    @override
    def event_key_pressed(self, event: pygame.event.Event):
        if event.key == pygame.K_ESCAPE and pygame.display.is_fullscreen():
            Controls.leave_fullscreen(self)
            self.show_score(self.score, self.placement)

        elif event.key == pygame.K_F11:
            if not pygame.display.is_fullscreen():
                Controls.enter_fullscreen(self)
                self.show_score(self.score, self.placement)
            else:
                Controls.leave_fullscreen(self)
                self.show_score(self.score, self.placement)

    @override
    def event_mouse_down(self, event: pygame.event.Event):
        if event.button == 1:
            self.mouse_clicked = True
            self.mouse_pos = (event.pos[0], event.pos[1])

    @override
    def event_mouse_up(self, event: pygame.event.Event):
        if self.mouse_clicked and event.button == 1:
            self.mouse_clicked = False

            for button in self.buttons.values():
                if button.is_pressed(self.mouse_pos):
                    button.press()
                    break

    @override
    def extra_default_mechanics(self):
        data = super().extra_default_mechanics()

        if data:
            self.methods[data["code"]](data)

class LoadingWindow(Window):
    loading_images = [
        r"Loading\0.png",
        r"Loading\45.png",
        r"Loading\90.png",
        r"Loading\135.png",
        r"Loading\180.png",
        r"Loading\225.png",
        r"Loading\270.png",
        r"Loading\315.png",
    ]

    def __init__(self, background_picture, chat_window, comms):
        super().__init__(background_picture, comms)
        global WIDTH, HEIGHT
        self.methods["MATCHED"] = self.start
        self.chat_window = chat_window

        self.x = WIDTH / 2 - (WIDTH * 0.1) / 2
        self.y = HEIGHT / 2 - (HEIGHT * 0.1) / 2

        self.length = len(LoadingWindow.loading_images)

        self.counter = 0
        self.clock = pygame.time.Clock()

    @override
    def extra_default_mechanics(self):
        data = super().extra_default_mechanics()

        if data:
            self.methods[data["code"]](data)

        if Controls.LOADING:
            Controls.display_image(LoadingWindow.loading_images[self.counter % self.length], (self.x, self.y), (10, 10))
            self.counter += 1
            self.clock.tick(5)

    def start(self, data):

        global current_window
        Controls.LOADING = False

        self.chat_window.display()

        current_window = self.chat_window
        self.chat_window.start_game(data)

class ChatWindow(Window):
    def __init__(self, background_picture, font_path, comms: TcpBySize, username):
        global WIDTH, HEIGHT

        super().__init__(background_picture, comms)
        self.text = ""
        self.msgs = {}
        self.left = 0
        self.right = 0
        self.text_font = pygame.font.Font(font_path, 35)
        self.timer_font = pygame.font.Font(font_path, 59)
        self.time = 120
        self.counter = 0
        self.mouse_pos = None
        self.mouse_clicked = False
        self.hovered = False
        self.first_key = True
        self.measurements = {
            "text_box_x" : WIDTH // 3.31,
            "text_box_y" : HEIGHT // 1.08,
            "text_box_w" : WIDTH // 2.52,
            "text_box_h" : HEIGHT // 18,
            "chat_y"     : HEIGHT // 18,
            "chat_h"     : HEIGHT // 1.17
        }

        self.methods["CHAT_MSG"] = self.add_chat_message

        self.input_box = pygame.Rect(
            self.measurements["text_box_x"] + 2,
            self.measurements["text_box_y"] + 2,
            self.measurements["text_box_w"] - 4,
            self.measurements["text_box_h"] - 4
        )
        self.chat_box = pygame.Rect(
            self.measurements["text_box_x"] + 2,
            self.measurements["chat_y"] + 2,
            self.measurements["text_box_w"] - 4,
            self.measurements["chat_h"] - 4
        )
        self.comms = comms
        self.username = username

        self.rects = {
            "text_box_1" : (pygame.Rect(
            self.measurements["text_box_x"],
            self.measurements["text_box_y"],
            self.measurements["text_box_w"],
            self.measurements["text_box_h"]
        ), (0, 255, 200), 0),
            "text_box_2" : (pygame.Rect(
            self.measurements["text_box_x"] + 2,
            self.measurements["text_box_y"] + 2,
            self.measurements["text_box_w"] - 4,
            self.measurements["text_box_h"] - 4
        ), (77, 91, 112), 0),
            "chat_box_1" : (pygame.Rect(
            self.measurements["text_box_x"],
            self.measurements["chat_y"],
            self.measurements["text_box_w"],
            self.measurements["chat_h"]
        ), (0, 255, 200), 0),
            "chat_box_2" : (pygame.Rect(
            self.measurements["text_box_x"] + 2,
            self.measurements["chat_y"] + 2,
            self.measurements["text_box_w"] - 4,
            self.measurements["chat_h"] - 4
        ), (39, 48, 61), 0),
            "timer_1" : (pygame.Rect(
            WIDTH // 2 - HEIGHT // 7.71 - 2,
            self.measurements["chat_y"] - HEIGHT // 54 - 2,
            HEIGHT // 3.85 + 4,
            self.measurements["text_box_h"] + 4
        ), (0, 255, 200), 30),
            "timer_2" : (pygame.Rect(
            WIDTH // 2 - HEIGHT // 7.71,
            self.measurements["chat_y"] - HEIGHT // 54,
            HEIGHT // 3.85,
            self.measurements["text_box_h"]
        ), (77, 91, 112), 30)
        }

    @override
    def event_key_pressed(self, event: pygame.event.Event):
        global WIDTH, HEIGHT, HOME, IS_TURN

        if event.key == pygame.K_BACKSPACE and self.text:
            self.text = self.text[:-1]

            if self.text_font.render(self.text, True, (255, 255, 255)).get_width() + (WIDTH // 38.4) > self.measurements["text_box_w"]:
                self.left -= 1

        elif event.key == pygame.K_RETURN and IS_TURN:
            if self.text.strip():
                for i in range(1, len(self.text)):
                    if i % 45 == 0:
                        self.text = self.text[:i] + "\n  " + self.text[i:]

                data = pickle.dumps({
                    "code": "GAME_MSG",
                    "msg": self.text,
                    "username": self.username
                })
                self.text = f"{self.username}: \n  " + self.text

                self.add_message(self.text, (71, 232, 65))
                self.text = ""

                self.left, self.right = 0, 0

                self.comms.send_with_size(data)

                IS_TURN = False

        elif event.key == pygame.K_ESCAPE:
            if pygame.display.is_fullscreen():
                Controls.leave_fullscreen(self)

        elif event.key == pygame.K_F11:
            if not pygame.display.is_fullscreen():
                Controls.enter_fullscreen(self)
            else:
                Controls.leave_fullscreen(self)

        elif event.key != pygame.K_BACKSPACE and event.key != pygame.K_RETURN and self.time != 0:
            if self.first_key:
                self.text = event.unicode
                self.first_key = False

            elif len(self.text) < 150:
                self.text += event.unicode
                self.right += 1

            if self.text_font.render(self.text, True, (255, 255, 255)).get_width() + (WIDTH // 64) > self.measurements["text_box_w"]:
                self.left += 1

    @override
    def event_mouse_motion(self, event: pygame.event.Event):
        pos = event.pos[0], event.pos[1]
        for button in self.buttons.values():
            if button.is_pressed(pos):
                if button.hovered:
                    cords = (0, 0)
                    ratios = (1, 1)
                    if button.is_img:
                        cords = (WIDTH // button.ratios[0], HEIGHT // button.ratios[1])
                        ratios = (button.ratios[2], button.ratios[3])

                    Controls.display_image(button.hovered, cords, scale_ratio=ratios)
                    self.hovered = True
                    break
        else:
            if self.hovered:
                self.hovered = False
                self.show_img_buttons()

    @override
    def event_mouse_down(self, event: pygame.event.Event):
        if event.button == 1:
            self.mouse_clicked = True
            self.mouse_pos = (event.pos[0], event.pos[1])

    @override
    def event_mouse_up(self, event: pygame.event.Event):
        if self.mouse_clicked and event.button == 1:
            self.mouse_clicked = False

            for button in self.buttons.values():
                if button.is_pressed(self.mouse_pos):
                    button.press()
                    break

    def add_message(self, text, color):
        self.msgs[text] = color

    @override
    def extra_default_mechanics(self):
        data = super().extra_default_mechanics()

        if data:
            self.methods[data["code"]](data)

        if self.time >= 0:
            [pygame.draw.rect(game_surface, self.rects[rct][1], self.rects[rct][0], border_radius=self.rects[rct][2]) for rct in self.rects]

            y = self.chat_box.bottom - 10

            for msg, color in reversed(self.msgs.items()):
                msg_surface = self.text_font.render(msg, True, color)
                msg_height = msg_surface.get_height()

                y -= msg_height + 5

                if y < self.measurements["chat_y"]:
                    break

                game_surface.blit(msg_surface, (self.chat_box.x + 10, y))

            pygame.draw.rect(game_surface, (255, 255, 255), self.input_box, 2)

            text_surface = self.text_font.render(self.text[self.left:self.right], True, (255, 255, 255))

            game_surface.blit(text_surface, (self.input_box.x + 10, self.input_box.y + 10))
            self.blit_clock(self.time)

            if self.time == 0:
                self.display_continue()

            self.counter += 1

            if self.counter == 60:
                self.counter = 0
                self.time -= 1

            pygame.display.flip()

            Controls.scale()
            self.show_img_buttons()

        pygame.display.flip()

    def blit_clock(self, _time):
        t = self.timer_font.render(ChatWindow.seconds_to_minutes(_time), True, (255, 255, 255))
        game_surface.blit(t, (WIDTH // 2 - HEIGHT // 16.36, self.measurements["chat_y"] - HEIGHT // 54))

    def start_game(self, data):
        Controls.enter_fullscreen(self)
        global IS_TURN

        self.reset_timer()
        IS_TURN = True if data["turn"] else False
        self.text = "You start the conversation!" if IS_TURN else "Other side starts the conversation!"
        self.right = len(self.text)

    @staticmethod
    def seconds_to_minutes(seconds):
        minutes, seconds = divmod(seconds, 60)
        m = "0" + str(minutes) if minutes < 10 else str(minutes)
        s = "0" + str(seconds) if seconds < 10 else str(seconds)

        return f"{m}:{s}"

    def add_chat_message(self, data):
        global IS_TURN

        IS_TURN = True
        text = f"User:\n  {data["data"]["msg"]}"

        self.add_message(text, (240, 50, 50))

    def reset_timer(self):
        self.time = 120

    def display_continue(self):
        choose = ChooseWindow(PATHS["Choose1"], self.comms, self.username)

        self.add_button("Continue",
                        (3.31, 1.3, 2.56, 5.53),
                        ButtonClick.continue_game,
                        args=(choose, self.comms),
                        image=PATHS["Continue"],
                        hovered=PATHS["ContinueHovered"])
        self.show_img_buttons()

class ChooseWindow(Window):
    def __init__(self, background_picture, comms, username):
        super().__init__(background_picture, comms)
        self.add_button("Human",
                        (7.68, 5.4, 2.82, 1.8),
                        ButtonClick.check_chosen,
                        hovered=PATHS["HumanHovered"],
                        args=("Human", comms))
        self.add_button("Bot",
                        (1.89, 5.4, 2.82, 1.8),
                        ButtonClick.check_chosen,
                        hovered=PATHS["BotHovered"],
                        args=("Bot", comms))

        self.comms = comms
        self.chosen = False
        self.methods["VERDICT"] = self.verdict
        self.username = username

    @override
    def event_mouse_down(self, event: pygame.event.Event):
        if event.button == 1:
            self.mouse_clicked = True
            self.mouse_pos = (event.pos[0], event.pos[1])

    @override
    def event_mouse_up(self, event: pygame.event.Event):
        if self.mouse_clicked and event.button == 1:
            self.mouse_clicked = False

            for button in self.buttons.values():
                if button.is_pressed(self.mouse_pos):
                    button.press()
                    break

    @override
    def event_mouse_motion(self, event: pygame.event.Event):
        pos = event.pos[0], event.pos[1]
        for button in self.buttons.values():
            if button.is_pressed(pos):
                if button.hovered:
                    cords = (0, 0)
                    ratios = (1, 1)
                    if button.is_img:
                        cords = (WIDTH // button.ratios[0], HEIGHT // button.ratios[1])
                        ratios = (button.ratios[2], button.ratios[3])

                    Controls.display_image(button.hovered, cords, scale_ratio=ratios)
                    self.hovered = True
                    break
        else:
            if self.hovered:
                self.hovered = False
                Controls.display_image(self.background_picture, (0, 0))
                self.show_img_buttons()

    @override
    def event_key_pressed(self, event: pygame.event.Event):
        if event.key == pygame.K_ESCAPE and pygame.display.is_fullscreen():
            Controls.leave_fullscreen(self)

        elif event.key == pygame.K_F11:
            if not pygame.display.is_fullscreen():
                Controls.enter_fullscreen(self)
            else:
                Controls.leave_fullscreen(self)

    @override
    def extra_default_mechanics(self):
        data = super().extra_default_mechanics()

        if data:
            if data["code"] in self.methods:
                self.methods[data["code"]](data)

    def verdict(self, data):
        global LOBBY

        #self.add_button("BackToLobby", (1, 1 ,1 ,1), ButtonClick.back_to_lobby, PATHS["BackToLobby"], args=(LOBBY,), hovered=PATHS["BackToLobbyHovered"])

        print(data["verdict"])

        if data["verdict"]:
            self.background_picture = PATHS["YouWon"]
            self.display_win()
        else:
            self.background_picture = PATHS["YouLost"]
            self.display_lose()

        self.buttons = {}
        self.add_button("BackToLobby",
                        (4.46, 1.35, 1.89, 8),
                        Controls.boot,
                        args=(self.comms, self.username),
                        image=PATHS["BackToLobby"],
                        hovered=PATHS["BackToLobbyHovered"])

    def display_win(self):
        Controls.display_image(PATHS["YouWon"],(0, 0), scale_ratio=(1, 1))

    def display_lose(self):
        Controls.display_image(PATHS["YouLost"],(0, 0), scale_ratio=(1, 1))

class ScoreBoardWindow(Window):
    def __init__(self, background_picture, comms):
        super().__init__(background_picture, comms)
        self.rect_width = WIDTH // 2.1
        self.rect_height = HEIGHT // 11

        self.rect = pygame.Rect(0, 0, self.rect_width, self.rect_height)
        self.text_font = pygame.font.Font(default_font, 80)

        self.methods["SCOREBOARD_RES"] = self.show_scoreboard_from_server
        self.sb = []
        self.page = 1

        self.add_button("Home", (64, 54, 16, 13.5), ButtonClick.back_to_lobby, args=(LOBBY,), hovered=PATHS["HomeHovered"])
        self.add_button("Left", (21.33, 3.54, 9.6, 2.16), func=self.show_4, args=(-1,), hovered=PATHS["LeftHovered"])
        self.add_button("Right", (1.17, 3.72, 9.6, 2.16), func=self.show_4, args=(1,), hovered=PATHS["RightHovered"])
        self.refresh_buttons(pygame.display.get_window_size())

        data = pickle.dumps({"code" : "SCOREBOARD_REQ"})
        comms.send_with_size(data)

    def show_scoreboard_from_server(self, data):
        self.sb = [(u, s) for u, s in data["scoreboard"].items()]
        self.show_scoreboard()

    def show_scoreboard(self):
        first = "Vacant" if len(self.sb) == 0 else f"{self.sb[0][0]} {self.sb[0][1]}"
        second = "Vacant" if len(self.sb) == 1 else f"{self.sb[1][0]} {self.sb[1][1]}"
        third = "Vacant" if len(self.sb) == 2 else f"{self.sb[2][0]} {self.sb[2][1]}"

        txt = self.text_font.render(first, True, (0, 0, 0))
        txt2 = self.text_font.render(second, True, (0, 0, 0))
        txt3 = self.text_font.render(third, True, (0, 0, 0))

        x, y = Controls.center((WIDTH // 2.24, HEIGHT // 21.6), txt.get_size(), (WIDTH // 9.36, HEIGHT // 5.83))
        x1, y1 = Controls.center((WIDTH // 3.46, HEIGHT // 4.90), txt2.get_size(), (WIDTH // 9.36, HEIGHT // 5.83))
        x2, y2 = Controls.center((WIDTH // 1.65, HEIGHT // 3.6), txt3.get_size(), (WIDTH // 9.36, HEIGHT // 5.83))

        game_surface.blit(txt2, (x1, y1))
        game_surface.blit(txt3, (x2, y2))
        game_surface.blit(txt, (x, y))

        self.show_4(0)

    def show_4(self, page):
        self.page += page

        if self.page == 0:
            self.page = 1
            return

        r = pygame.rect.Rect((WIDTH // 4.57, HEIGHT // 1.89), (WIDTH // 1.77942, HEIGHT // 2.2))
        pygame.draw.rect(game_surface, (30, 35, 40), r)

        x, y = Controls.center((WIDTH // 4.57, HEIGHT // 1.95), (self.rect_width, self.rect_height), (WIDTH // 1.77942, HEIGHT // 2.2 // 3))

        self.rect.x , self.rect.y = x, y

        for i in range(4):
            txt = "Vacant" if len(self.sb) < 3 + i + (self.page - 1) * 4 else f"{self.sb[3 + i + (self.page - 1) * 4][0]} {self.sb[3 + i + (self.page - 1) * 4][1]}"
            txt = self.text_font.render(txt, True, (0, 0, 0))

            place = f"#{4 + i + (self.page - 1) * 4}"
            txt2 = self.text_font.render(place, True, (0, 0, 0))

            x, y = Controls.center((self.rect.x, self.rect.y), txt.get_size(), (self.rect.width, self.rect.height))

            pygame.draw.rect(game_surface, (60, 110, 180), self.rect)
            game_surface.blit(txt, (x, y))
            game_surface.blit(txt2, (self.rect.x, y))

            self.rect.y += self.rect.height + HEIGHT // 48

        Controls.scale()

    @override
    def event_mouse_motion(self, event: pygame.event.Event):
        pos = event.pos[0], event.pos[1]
        for button in self.buttons.values():
            if button.is_pressed(pos):
                if button.hovered:
                    if self.hovered:
                        break

                    cords = (0, 0)
                    ratios = (1, 1)
                    if button.is_img:
                        cords = (WIDTH // button.ratios[0], HEIGHT // button.ratios[1])
                        ratios = (button.ratios[2], button.ratios[3])

                    Controls.display_image(button.hovered, cords, scale_ratio=ratios)
                    self.show_scoreboard()
                    self.hovered = True
                    break
        else:
            if self.hovered:
                self.hovered = False
                Controls.display_image(PATHS["Scoreboard"], (0, 0))
                self.show_scoreboard()

    @override
    def event_key_pressed(self, event: pygame.event.Event):
        if event.key == pygame.K_ESCAPE and pygame.display.is_fullscreen():
            Controls.leave_fullscreen(self)
            self.refresh_buttons(pygame.display.get_window_size())
            self.show_scoreboard()

        elif event.key == pygame.K_F11:
            if not pygame.display.is_fullscreen():
                Controls.enter_fullscreen(self)
                self.refresh_buttons(pygame.display.get_window_size())
                self.show_scoreboard()
            else:
                Controls.leave_fullscreen(self)
                self.refresh_buttons(pygame.display.get_window_size())
                self.show_scoreboard()

    @override
    def event_mouse_down(self, event: pygame.event.Event):
        if event.button == 1:
            self.mouse_clicked = True
            self.mouse_pos = (event.pos[0], event.pos[1])

    @override
    def event_mouse_up(self, event: pygame.event.Event):
        if self.mouse_clicked and event.button == 1:
            self.mouse_clicked = False

            for button in self.buttons.values():
                if button.is_pressed(self.mouse_pos):
                    button.press()
                    break

    @override
    def extra_default_mechanics(self):
        data = super().extra_default_mechanics()

        if data:
            self.methods[data["code"]](data)

class Button:
    def __init__(self, ratios: tuple, func: Callable, args: tuple, hovered: str, is_img: bool):
        self.ratios = ratios
        self.rect = pygame.Rect(
            RESOLUTION[0] // self.ratios[0],
            RESOLUTION[1] // self.ratios[1],
            RESOLUTION[0] // self.ratios[2],
            RESOLUTION[1] // self.ratios[3]
        )
        self.func = func

        self.hovered = hovered
        self.args = args
        self.is_img = is_img

    def press(self):
        if self.args:
            self.func(*self.args)
        else:
            self.func()

    def is_pressed(self, click_pos: tuple):
        return self.rect.collidepoint(click_pos)

    def refresh(self, screen_size: tuple):
        self.rect = pygame.Rect(
            screen_size[0] // self.ratios[0],
            screen_size[1] // self.ratios[1],
            screen_size[0] // self.ratios[2],
            screen_size[1] // self.ratios[3]
        )

class ButtonClick:
    @staticmethod
    def init_game(loading_window: LoadingWindow, comms: TcpBySize):
        global current_window

        print("Initializing game...")

        Controls.LOADING = True
        current_window = loading_window
        loading_window.display()

        data = pickle.dumps({"code" : "START"})
        comms.send_with_size(data)

    @staticmethod
    def continue_game(choose_window, comms):
        global current_window

        current_window = choose_window
        choose_window.display()

    @staticmethod
    def check_chosen(chosen: str, comms: TcpBySize):
        data = pickle.dumps({
            "code" : "CHECK",
            "option" : chosen
        })

        comms.send_with_size(data)

    @staticmethod
    def back_to_lobby(lobby):
        global current_window

        lobby.display()
        lobby.refresh_buttons(pygame.display.get_window_size())
        current_window = lobby

    @staticmethod
    def watch():
        global current_window


    @staticmethod
    def scoreboard(comms):
        global current_window
        scoreboard = ScoreBoardWindow(PATHS["Scoreboard"], comms)
        scoreboard.display()
        current_window = scoreboard

class Controls:
    LOADING = False

    @staticmethod
    def boot(comms: TcpBySize, username: str):
        global current_window, LOBBY, CHAT, LOADING

        Controls.load_images()
        LOBBY, CHAT, LOADING = Controls.get_windows(comms, username)
        current_window = LOBBY
        LOBBY.display()

    @staticmethod
    def scale():
        screen_w, screen_h = HOME.get_size()

        scaled = pygame.transform.smoothscale(game_surface, (screen_w, screen_h))

        HOME.blit(scaled, (0, 0))
        pygame.display.flip()

    @staticmethod
    def get_windows(comms, username):
        chat = ChatWindow(PATHS["ChatScreen"], default_font, comms, username)
        loading = LoadingWindow(PATHS["LoadingScreen"], chat, comms)

        lobby = LobbyWindow(PATHS["GameLobby"], comms)
        lobby.add_button("Play", (2.56, 1.44, 4.57, 9.39), ButtonClick.init_game, args=(loading, comms), hovered=PATHS["PlayHovered"])
        lobby.add_button("Watch", (1.53, 1.37, 5.48, 12.7), ButtonClick.watch, hovered=PATHS["WatchHovered"])
        lobby.add_button("Scoreboard", (7.24, 1.37, 4.74, 12.7), ButtonClick.scoreboard, args=(comms,), hovered=PATHS["ScoreboardHovered"])

        return lobby, loading, chat

    @staticmethod
    def enter_fullscreen(window: Window):
        global HOME
        HOME = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        window.refresh_buttons(HOME.get_size())
        Controls.scale()

    @staticmethod
    def leave_fullscreen(window: Window):
        global WIDTH, HEIGHT, HOME
        HOME = pygame.display.set_mode((WIDTH / 2, HEIGHT / 2), pygame.RESIZABLE)
        window.refresh_buttons(HOME.get_size())
        Controls.scale()

    @staticmethod
    def display_image(path, cords, scale_ratio=(1.0, 1.0)):
        if path in IMAGES.keys():
            home_screen = IMAGES[path]
        else:
            home_screen = pygame.image.load(path).convert_alpha()
            IMAGES[path] = home_screen

        home_screen = pygame.transform.smoothscale(home_screen, (int(game_surface.get_width() // scale_ratio[0]), int(game_surface.get_height() // scale_ratio[1])))
        game_surface.blit(home_screen, cords)
        Controls.scale()

    @staticmethod
    def load_images():
        for img in PATHS.values():
            if img in IMAGES.keys():
                continue
            else:
                home_screen = pygame.image.load(img).convert_alpha()
                IMAGES[img] = home_screen

        for img in LoadingWindow.loading_images:
            if img in IMAGES.keys():
                continue
            else:
                home_screen = pygame.image.load(img).convert_alpha()
                IMAGES[img] = home_screen

    @staticmethod
    def center(obj_cords, obj_size, window_size):
        obj_width, obj_height = obj_size
        window_width, window_height = window_size
        x, y = obj_cords
        x += (window_width - obj_width) // 2
        y += (window_height - obj_height) // 2

        return x, y

def end_login(comms, username):
    global HOME, WIDTH, HEIGHT, RESOLUTION, game_surface, PATHS, current_window, LOBBY, CHAT, LOADING

    pygame.init()
    HOME = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    WIDTH = pygame.display.Info().current_w
    HEIGHT = pygame.display.Info().current_h
    RESOLUTION = (WIDTH, HEIGHT)

    game_surface = pygame.Surface(RESOLUTION)

    comms.sock.settimeout(0.01)

    Controls.boot(comms, username)

    running = True
    clock = pygame.time.Clock()

    while running:
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEMOTION:
                current_window.event_mouse_motion(event)

            elif event.type == pygame.KEYDOWN:
                current_window.event_key_pressed(event)

            elif event.type == pygame.WINDOWRESIZED:
                current_window.event_window_resize(HOME.get_size())

            elif event.type == pygame.MOUSEBUTTONDOWN:
                current_window.event_mouse_down(event)

            elif event.type == pygame.MOUSEBUTTONUP:
                current_window.event_mouse_up(event)

            elif event.type == pygame.MOUSEWHEEL:
                current_window.event_mousewheel(event)

        current_window.extra_default_mechanics()
        clock.tick(60)

    pygame.quit()
