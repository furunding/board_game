from enum import IntEnum
import pygame
from pygame.locals import *

GAME_VERSION = 'V1.0'

REC_SIZE = 50
CHESS_RADIUS = REC_SIZE//2 - 2
CHESS_LEN = 15
MAP_WIDTH = CHESS_LEN * REC_SIZE
MAP_HEIGHT = CHESS_LEN * REC_SIZE

INFO_WIDTH = 200
BUTTON_WIDTH = 140
BUTTON_HEIGHT = 50

SCREEN_WIDTH = MAP_WIDTH + INFO_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT

class MAP_ENTRY_TYPE(IntEnum):
	MAP_EMPTY = 0,
	MAP_PLAYER_ONE = 1,
	MAP_PLAYER_TWO = 2,
	MAP_NONE = 3, # out of map range
	
class Map():
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.map = [[0 for x in range(self.width)] for y in range(self.height)]
		self.steps = []
	
	def reset(self):
		for y in range(self.height):
			for x in range(self.width):
				self.map[y][x] = 0
		self.steps = []
	
	def reverseTurn(self, turn):
		if turn == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
			return MAP_ENTRY_TYPE.MAP_PLAYER_TWO
		else:
			return MAP_ENTRY_TYPE.MAP_PLAYER_ONE

	def getMapUnitRect(self, x, y):
		map_x = x * REC_SIZE
		map_y = y * REC_SIZE
		
		return (map_x, map_y, REC_SIZE, REC_SIZE)
	
	def MapPosToIndex(self, map_x, map_y):
		x = map_x // REC_SIZE
		y = map_y // REC_SIZE		
		return (x, y)
	
	def isInMap(self, map_x, map_y):
		if (map_x <= 0 or map_x >= MAP_WIDTH or 
			map_y <= 0 or map_y >= MAP_HEIGHT):
			return False
		return True
	
	def isEmpty(self, x, y):
		return (self.map[y][x] == 0)
	
	def click(self, x, y, type):
		self.map[y][x] = type.value
		self.steps.append((x,y))

	def drawChess(self, screen):
		player_one = (255, 251, 240)
		player_two = (88, 87, 86)
		player_color = [player_one, player_two]
		
		font = pygame.font.SysFont(None, REC_SIZE*2//3)
		for i in range(len(self.steps)):
			x, y = self.steps[i]
			map_x, map_y, width, height = self.getMapUnitRect(x, y)
			pos, radius = (map_x + width//2, map_y + height//2), CHESS_RADIUS
			turn = self.map[y][x]
			if turn == 1:
				op_turn = 2
			else:
				op_turn = 1
			pygame.draw.circle(screen, player_color[turn-1], pos, radius)
			
			msg_image = font.render(str(i), True, player_color[op_turn-1], player_color[turn-1])
			msg_image_rect = msg_image.get_rect()
			msg_image_rect.center = pos
			screen.blit(msg_image, msg_image_rect)
			
		
		if len(self.steps) > 0:
			last_pos = self.steps[-1]
			map_x, map_y, width, height = self.getMapUnitRect(last_pos[0], last_pos[1])
			purple_color = (255, 0, 255)
			point_list = [(map_x, map_y), (map_x + width, map_y), 
					(map_x + width, map_y + height), (map_x, map_y + height)]
			pygame.draw.lines(screen, purple_color, True, point_list, 1)
			
	def drawBackground(self, screen):
		color = (0, 0, 0)
		for y in range(self.height):
			# draw a horizontal line
			start_pos, end_pos= (REC_SIZE//2, REC_SIZE//2 + REC_SIZE * y), (MAP_WIDTH - REC_SIZE//2, REC_SIZE//2 + REC_SIZE * y)
			if y == (self.height)//2:
				width = 2
			else:
				width = 1
			pygame.draw.line(screen, color, start_pos, end_pos, width)
		
		for x in range(self.width):
			# draw a horizontal line
			start_pos, end_pos= (REC_SIZE//2 + REC_SIZE * x, REC_SIZE//2), (REC_SIZE//2 + REC_SIZE * x, MAP_HEIGHT - REC_SIZE//2)
			if x == (self.width)//2:
				width = 2
			else:
				width = 1
			pygame.draw.line(screen, color, start_pos, end_pos, width)
				
		
		rec_size = 8
		pos = [(3,3), (11,3), (3, 11), (11,11), (7,7)]
		for (x, y) in pos:
			pygame.draw.rect(screen, color, (REC_SIZE//2 + x * REC_SIZE - rec_size//2, REC_SIZE//2 + y * REC_SIZE - rec_size//2, rec_size, rec_size))

class Button():
	def __init__(self, screen, text, x, y, color, enable):
		self.screen = screen
		self.width = BUTTON_WIDTH
		self.height = BUTTON_HEIGHT
		self.button_color = color
		self.text_color = (255, 255, 255)
		self.enable = enable
		self.font = pygame.font.SysFont(None, BUTTON_HEIGHT*2//3)
		
		self.rect = pygame.Rect(0, 0, self.width, self.height)
		self.rect.topleft = (x, y)
		self.text = text
		self.init_msg()
		
	def init_msg(self):
		if self.enable:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
		else:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
		self.msg_image_rect = self.msg_image.get_rect()
		self.msg_image_rect.center = self.rect.center
		
	def draw(self):
		if self.enable:
			self.screen.fill(self.button_color[0], self.rect)
		else:
			self.screen.fill(self.button_color[1], self.rect)
		self.screen.blit(self.msg_image, self.msg_image_rect)
		

class StartButton(Button):
	def __init__(self, screen, text, x, y):
		super().__init__(screen, text, x, y, [(26, 173, 25),(158, 217, 157)], True)
	
	def click(self, game):
		if self.enable: 
			game.start()
			game.winner = None
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
			self.enable = False
			return True
		return False
	
	def unclick(self):
		if not self.enable:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
			self.enable = True
		
class GiveupButton(Button):
	def __init__(self, screen, text, x, y):
		super().__init__(screen, text, x, y, [(230, 67, 64),(236, 139, 137)], False)
		
	def click(self, game):
		if self.enable:
			game.is_play = False
			if game.winner is None:
				game.winner = game.map.reverseTurn(game.player)
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
			self.enable = False
			return True
		return False

	def unclick(self):
		if not self.enable:
			self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
			self.enable = True

class Gomuku():
	def __init__(self, caption, players={}):
		pygame.init()
		self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
		pygame.display.set_caption(caption)
		self.clock = pygame.time.Clock()
		self.buttons = []
		self.buttons.append(StartButton(self.screen, 'Start', MAP_WIDTH + 30, 15))
		self.buttons.append(GiveupButton(self.screen, 'Giveup', MAP_WIDTH + 30, BUTTON_HEIGHT + 45))
		self.is_play = False

		self.map = Map(CHESS_LEN, CHESS_LEN)
		self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
		self.action = None
		self.AI = None
		self.useAI = False
		self.winner = None
	
	def start(self):
		self.is_play = True
		self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
		self.map.reset()

	def play(self):
		self.clock.tick(60)
		
		light_yellow = (247, 238, 214)
		pygame.draw.rect(self.screen, light_yellow, pygame.Rect(0, 0, MAP_WIDTH, SCREEN_HEIGHT))
		pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(MAP_WIDTH, 0, INFO_WIDTH, SCREEN_HEIGHT))
		
		for button in self.buttons:
			button.draw()
		
		if self.is_play and not self.isOver():
			if self.useAI:
				x, y = self.AI.findBestChess(self.map.map, self.player)
				self.checkClick(x, y, True)
				self.useAI = False

			if self.action is not None:
				self.checkClick(self.action[0], self.action[1])
				self.action = None
			
			if not self.isOver():
				self.changeMouseShow()
			
		if self.isOver():
			self.showWinner()

		self.map.drawBackground(self.screen)
		self.map.drawChess(self.screen)

	
	def changeMouseShow(self):
		map_x, map_y = pygame.mouse.get_pos()
		x, y = self.map.MapPosToIndex(map_x, map_y)
		if self.map.isInMap(map_x, map_y) and self.map.isEmpty(x, y):
			pygame.mouse.set_visible(False)
			light_red = (213, 90, 107)
			pos, radius = (map_x, map_y), CHESS_RADIUS
			pygame.draw.circle(self.screen, light_red, pos, radius)
		else:
			pygame.mouse.set_visible(True)
	
	def checkClick(self,x, y, isAI=False):
		self.map.click(x, y, self.player)
		if self.AI.isWin(self.map.map, self.player):
			self.winner = self.player
			self.click_button(self.buttons[1])
		else:	
			self.player = self.map.reverseTurn(self.player)
			if not isAI:	
				self.useAI = True
	
	def mouseClick(self, map_x, map_y):
		if self.is_play and self.map.isInMap(map_x, map_y) and not self.isOver():
			x, y = self.map.MapPosToIndex(map_x, map_y)
			if self.map.isEmpty(x, y):
				self.action = (x, y)
	
	def isOver(self):
		return self.winner is not None

	def showWinner(self):
		def showFont(screen, text, location_x, locaiton_y, height):
			font = pygame.font.SysFont(None, height)
			font_image = font.render(text, True, (0, 0, 255), (255, 255, 255))
			font_image_rect = font_image.get_rect()
			font_image_rect.x = location_x
			font_image_rect.y = locaiton_y
			screen.blit(font_image, font_image_rect)
		if self.winner == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
			str = 'Winner is White'
		else:
			str = 'Winner is Black'
		showFont(self.screen, str, MAP_WIDTH + 25, SCREEN_HEIGHT - 60, 30)
		pygame.mouse.set_visible(True)
	
	def click_button(self, button):
		if button.click(self):
			for tmp in self.buttons:
				if tmp != button:
					tmp.unclick()
					
	def check_buttons(self, mouse_x, mouse_y):
		for button in self.buttons:
			if button.rect.collidepoint(mouse_x, mouse_y):
				self.click_button(button)
				break