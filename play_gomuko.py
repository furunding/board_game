from game import Gomuku
import pygame
from absl import app
from absl import flags
import importlib

FLAGS = flags.FLAGS

flags.DEFINE_string("white", "human", "white role")
flags.DEFINE_string("black", "ai.EvaluateAI", "black role")

def main(unused_argv):
    def str2agent(str_):
        agent_module, agent_name = str_.rsplit(".", 1)
        return getattr(importlib.import_module(agent_module), agent_name)

    GAME_VERSION = "0.01"
    game = Gomuku("FIVE CHESS " + GAME_VERSION, players={"white": "human" if FLAGS.white == "human" else str2agent(FLAGS.white), \
        "black": "human" if FLAGS.black == "human" else str2agent(FLAGS.black)})
    while True:
    	game.play()
    	pygame.display.update()
    
    	for event in pygame.event.get():
    		if event.type == pygame.QUIT:
    			pygame.quit()
    			exit()
    		elif event.type == pygame.MOUSEBUTTONDOWN:
    			mouse_x, mouse_y = pygame.mouse.get_pos()
    			game.mouseClick(mouse_x, mouse_y)
    			game.check_buttons(mouse_x, mouse_y)


if __name__ == "__main__":
    app.run(main)