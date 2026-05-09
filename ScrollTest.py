import pygame

def main():
    pygame.init()

    HOME = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
    rect = pygame.Rect(0, 0, 100, 100)
    font = pygame.font.SysFont("Arial", 20)
    text = font.render("Hello World!", True, (255, 255, 255))


    while True:
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    print("Scrolled Up")
                    rect.y = rect.y - 5
                elif event.y < 0:
                    print("Scrolled Down")
                    rect.y = rect.y + 5

                if rect.right < 0:
                    rect.left = HOME.get_width()

                if rect.left > HOME.get_width():
                    rect.right = 0

                if rect.bottom < 0:
                    rect.top = HOME.get_height()

                if rect.top > HOME.get_height():
                    rect.bottom = 0

            HOME.fill((0, 0, 0))
            pygame.draw.rect(HOME, pygame.Color(255, 0, 0), rect)
            HOME.blit(text, (rect.x, rect.y))
            pygame.display.flip()


if __name__ == "__main__":
    main()