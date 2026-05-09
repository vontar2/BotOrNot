import pygame



PATHS = {
    "GameLobby" : r"LobbyPictures\GameLobby.png",
    "PlayHovered" : r"LobbyPictures\PlayHovered.png",
    "WatchHovered" : r"LobbyPictures\WatchHovered.png",
    "ScoreboardHovered" : r"LobbyPictures\ScoreboardHovered.png",
    "LoadingScreen" : r"Loading\Empty.png",
    "ChatScreen" : r"ChatPictures\ChatRoom.png",
    "Choose1" : r"ChatPictures\Choose1.png",
    "HumanHovered" : r"ChatPictures\HumanHovered.png",
    "BotHovered" : r"ChatPictures\BotHovered.png",
    "BackToLobby" : r"ChatPictures\BackToLobby.png",
    "BackToLobbyHovered" : r"ChatPictures\BackToLobbyHovered.png",
    "Continue" : r"ChatPictures\Continue.png",
    "ContinueHovered" : r"ChatPictures\ContinueHovered.png",
    "YouWon" : r"ChatPictures\YouWon.png",
    "YouLost" : r"ChatPictures\YouLost.png"
}


pygame.init()
WIDTH = pygame.display.Info().current_w
HEIGHT = pygame.display.Info().current_h
RESOLUTION = (WIDTH, HEIGHT)

game_surface = pygame.Surface(RESOLUTION)
w_game = game_surface.get_width()
h_game = game_surface.get_height()

HOME = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
text_font = pygame.font.Font("Robot-Regular.ttf", 35)
timer = pygame.font.Font("Broken Robot.ttf", 59)

text_box_x = w_game // 3.31
text_box_y = h_game // 1.08
text_box_w = w_game // 2.52
text_box_h = h_game // 18

input_box = pygame.Rect(text_box_x + 2, text_box_y + 2,text_box_w - 4,text_box_h - 4)

chat_y = h_game // 18
chat_h = h_game // 1.17

#pygame.draw.rect(game_surface, (0, 0, 0), (text_box_x,chat_y,text_box_w,chat_h))
chat_box = pygame.Rect(text_box_x + 2, chat_y + 2,text_box_w - 4,chat_h - 4)


def main():
    global HOME, game_surface, w_game, h_game, input_box, chat_box

    running = True
    text = f""
    display_image("ChatPictures/ChatRoom.png", (0, 0))
    clock = pygame.time.Clock()
    msgs = []
    left = 0
    right = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    HOME = pygame.display.set_mode((WIDTH / 2, HEIGHT / 2))
                    scale()
                    pygame.display.flip()

                elif event.key == pygame.K_BACKSPACE and text:
                    text = text[:-1]

                    if text_font.render(text, True, (255, 255, 255)).get_width() + (w_game // 38.4) > text_box_w:
                        left -= 1

                elif event.key == pygame.K_RETURN:
                    if text.strip():
                        for i in range(1, len(text)):
                            if i % 45 == 0:
                                text = text[:i] + "\n  " + text[i:]
                        text = f"User: \n  " + text

                        msgs.append(text)
                        text = ""

                        left, right = 0, 0

                elif event.key != pygame.K_BACKSPACE:
                    text += event.unicode
                    right += 1

                    if text_font.render(text, True, (255, 255, 255)).get_width() + 30 > text_box_w:
                        left += 1

        pygame.draw.rect(game_surface, (0, 0, 0), (text_box_x,text_box_y,text_box_w,text_box_h))
        pygame.draw.rect(game_surface, (77, 91, 112), (text_box_x + 2, text_box_y + 2,text_box_w - 4,text_box_h - 4))

        pygame.draw.rect(game_surface, (0, 255, 200), (text_box_x,chat_y,text_box_w,chat_h))
        pygame.draw.rect(game_surface, (39, 48, 61), (text_box_x + 2, chat_y + 2,text_box_w - 4,chat_h - 4))


        y = chat_box.bottom - 10

        for msg in reversed(msgs):
            msg_surface = text_font.render(msg, True, (71, 232, 65))
            msg_height = msg_surface.get_height()

            y -= msg_height + 5

            if y < chat_y:
                break

            game_surface.blit(msg_surface, (chat_box.x + 10, y))

        pygame.draw.rect(game_surface, (0, 255, 200), input_box, 2)

        text_surface = text_font.render(text[left:right], True, (255, 255, 255))

        game_surface.blit(text_surface, (input_box.x + 10, input_box.y + 10))


        pygame.draw.rect(game_surface, (0, 255, 200), (WIDTH // 2 - 142, chat_y - 22 ,284 ,text_box_h + 4), border_radius=30 )
        pygame.draw.rect(game_surface, (77, 91, 112), (WIDTH // 2 - 140, chat_y - 20 ,280 ,text_box_h), border_radius=30 )
        t = timer.render("02:00", True, (255, 255, 255))
        print(t.size)
        game_surface.blit(t, (WIDTH // 2 - 100 + 34, chat_y - 20))

        pygame.display.flip()

        scale()
        pygame.display.flip()

        clock.tick(60)


def scale():
    screen_w, screen_h = HOME.get_size()

    scaled = pygame.transform.smoothscale(game_surface, (screen_w, screen_h))

    HOME.blit(scaled, (0, 0))
    pygame.display.flip()

def display_image(path, cords, scale_ratio=(1.0, 1.0)):
    home_screen = pygame.image.load(path).convert_alpha()
    home_screen = pygame.transform.smoothscale(home_screen, (int(game_surface.get_width() * scale_ratio[0]), int(game_surface.get_height() * scale_ratio[1])))
    game_surface.blit(home_screen, cords)
    scale()

if __name__ == "__main__":
    display_image(PATHS["Choose1"], (0, 0))
    while True:
        pass