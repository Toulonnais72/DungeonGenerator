import random, math
from Display import *
game = True
room_type = ['Hall', 'Cave', 'Bedroom', 'Dining Room', 'Cellar']

class Direction2D:

    def __init__(self):
        pass

    def get_random_cardinal_direction(self):
        cardinal_directions_list = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        return random.choice(cardinal_directions_list)

class Entry:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.door = random.randint(1,3) #1 = no door, 2 = door, 3 = locked door

class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw_wall1(self):
        #pygame.draw.line(display_surface, white, [self.x1, self.y1], [self.x2, self.y2])
        for x in range(0, self.x2 - self.x1 + 2 * tilesize, tilesize):
            index = random.randint(0, 15)
            greytiles.draw(display_surface, index, self.x1 + x - tilesize, self.y1 - tilesize)

    def draw_wall2(self):
        for y in range(0, self.y2 - self.y1 + 2 * tilesize, tilesize):
            index = random.randint(0, 15)
            greytiles.draw(display_surface, index, self.x1, self.y1 + y - tilesize)

    def draw_wall3(self):
        for x in range(0, self.x1 - self.x2 , tilesize):
            index = random.randint(0, 15)
            greytiles.draw(display_surface, index, self.x2 + x, self.y2)

    def draw_wall4(self):
        for y in range(0, self.y1 - self.y2 + tilesize, tilesize):
            index = random.randint(0, 15)
            greytiles.draw(display_surface, index, self.x2 - tilesize, self.y2 + y)


    def draw_red(self):
        pygame.draw.line(display_surface, red, [self.x1, self.y1], [self.x2, self.y2])

    def draw_green(self):
        pygame.draw.line(display_surface, green, [self.x1, self.y1], [self.x2, self.y2])


class Room:

    def __init__(self, x, y, size, type, nb_entries):
        self.x = x
        self.y = y
        self.angle = 0 #random.randint(-45,45)*0.7854 #Angle converted to radians
        self.size = size
        #self.width = int(math.sqrt(self.size*5/8) // tilesize)
        #self.height = int(self.width*8/5 // tilesize)
        self.width = random.randint(2, int(size / 2))
        #self.width = self.width // tilesize
        self.height = int(round(self.size / self.width))
        self.size = self.width * self.height

        #self.height = self.height // tilesize
        print("Room size =", self.size, "(W x H):", self.width, "x", self.height)
        self.nb_entries = nb_entries
        self.entry = []
        self.type = type
        self.walls = []
        self.create_walls()
        self.center_x = int(self.walls[0].x1 + self.walls[2].x1)/2
        self.center_y = int(self.walls[0].y1 + self.walls[2].y1)/2
        self.radius = int(math.sqrt((self.height*tilesize)**2 + (self.width*tilesize)**2) / 2)
        #self.create_entries()
        self.nb_doors = random.randint(1, self.nb_entries)
        self.present = True
        #self.create_doors()

    def create_entries(self):
        for i in range(self.nb_entries):
            self.entry.append(Entry(0,0))

    def create_walls(self):
        x1 = self.x * tilesize
        y1 = self.y * tilesize
        x2 = int(x1 + self.width*math.cos(self.angle) * tilesize)
        y2 = int(y1 + self.width*math.sin(self.angle) * tilesize)
        x3 = int(x2 - self.height*math.sin(self.angle) * tilesize)
        y3 = int(y2 + self.height*math.cos(self.angle) * tilesize)
        x4 = int(x1 - self.height*math.sin(self.angle) * tilesize)
        y4 = int(y1 + self.height*math.cos(self.angle) * tilesize)

        if self.type in room_type:
            self.walls.append(Wall(x1, y1, x2, y2))
            self.walls.append(Wall(x2, y2, x3, y3))
            self.walls.append(Wall(x3, y3, x4, y4))
            self.walls.append(Wall(x4, y4, x1, y1))

    '''def normalize(self):
        for wall in self.walls:
            wall.x1 = wall.x1 // tilesize
            wall.y1 = wall.y1 // tilesize
            wall.x2 = wall.x2 // tilesize
            wall.y2 = wall.y2 // tilesize'''


    def draw(self):
        self.walls[0].draw_wall1()
        self.walls[1].draw_wall2()
        self.walls[2].draw_wall3()
        self.walls[3].draw_wall4()

    def draw_red(self):
        for wall in self.walls:
            wall.draw_red()

    def draw_green(self):
        for wall in self.walls:
            wall.draw_green()

class Dungeon:

    def __init__(self, X, Y, nb_rooms, offset):
        self.size = X * Y
        self.X = X
        self.Y = Y
        self.nb_rooms = nb_rooms
        self.rooms = []
        self.create_rooms()
        self.remove_colliding_rooms()
        self.treasures = random.randint(1, self.nb_rooms)
        self.traps = random.randint(1, self.nb_rooms)
        self.monsters = random.randint(1,self.nb_rooms)
        self.npc = random.randint(1,5)
        self.entries = random.randint(1,self.nb_rooms)
        self.doors = random.randint(1,self.entries)
        self.offset = offset
        self.path = []

    def create_rooms(self):
        for i in range(self.nb_rooms):
            x = random.randint(1, self.X)
            y = random.randint(1, self.Y)
            self.rooms.append(Room(x, y, random.randint(4, 100), 'Cave', random.randint(1,4)))

    def delete_room(self, room_number):
        del self.rooms[room_number]

    def draw_dungeon(self):
        for room in self.rooms:
            if room.present:
                room.draw()
                room.draw_green()
            else:
                room.draw_red()

    def detect_room_collide(self, room1, room2):
        if (room1.center_x, room1.center_y) == (room2.center_x, room2.center_y):
            return False
        distance = math.sqrt((room1.center_x - room2.center_x)**2 + (room1.center_y - room2.center_y)**2)
        if distance <= room1.radius + room2.radius:
            return True
        else:
            return False

    def remove_colliding_rooms(self):
        for room1 in self.rooms:
            for room2 in self.rooms[self.rooms.index(room1)+1:]:
                if room1.present and room2.present:
                    if self.detect_room_collide(room1, room2):
                        print("Collision between", self.rooms.index(room1), " and ", self.rooms.index(room2))
                        print("Deleting ", self.rooms.index(room1))
                        #self.delete_room(self.rooms.index(room1))
                        if room1.size < room2.size: room1.present = False
                        else: room2.present = False
                    else:
                        print("No collision between", self.rooms.index(room1), " and ", self.rooms.index(room2))

    def simple_random_walk(self, start_position, walk_length):
        previous_position = start_position
        new_position = [0, 0]
        direction = Direction2D()
        newpath = []
        for i in range(walk_length):
            dirvector = direction.get_random_cardinal_direction()
            new_position[0] = previous_position[0] + dirvector[0]
            new_position[1] = previous_position[1] + dirvector[1]
            newpath.append(new_position[:])
            previous_position = new_position
        self.path = newpath

    def binary_space_partitioning(size, max_split):
        pass



def main():
    global game
    while game:
        dungeon = Dungeon(int(1000 / tilesize), int(800 / tilesize) , 200,1)
        x, y = 800, 300
        '''dungeon.simple_random_walk([0, 0],5000)
        for tiles in dungeon.path[:]:
            print(tiles)
            new_x, new_y = x + tiles[0]*30, y + tiles[1]*30
            index = random.randint(1, greytiles.cols*greytiles.rows-1)
            greytiles.draw(display_surface, index, new_x, new_y)
            print(index)'''

        display_surface.fill(black)
        dungeon.draw_dungeon()

            #pygame.display.flip()
            #pygame.time.wait(1)
            #x, y = new_x, new_y
        pygame.display.flip()
        pygame.time.wait(3000)
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                game = False
                pygame.display.quit()
                pygame.quit()

if __name__ == '__main__':
    main()