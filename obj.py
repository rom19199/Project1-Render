# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:05:25 2021

@author: hugo_
"""

import struct 

def try_int(s, base=10, val=None):
  try:
    return int(s, base)
  except ValueError:
    return val

def color(r, g, b):
    return bytes([int(b * 255), int(g * 255), int(r * 255)])


class Obj1(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                prefix,value = line.split(' ',1)
                print(value)
             
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "  ").replace("/", " ")
                   
                    
                   
                    ##
                    i = string.find(" ") + 1
                    
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append((string[i:-1]))
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    ##
                    
                    
                    # print(face)
                    # results = list(map(int, face))
                    (self.faces.append(list(face)))
                    # self.faces.append([list(map(int, fa.split('/'))) for fa in a.split(' ')])

            f.close()
        except IOError:
            print(".obj file not found.")

class Obj2(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.lines = f.read().splitlines()

        self.vertices = []
        self.faces = []
        self.read()

    def read(self):
        for line in self.lines:
            if line:
              prefix, value = line.split(' ', 1) 
             

              if prefix == 'v':
                self.vertices.append(list(map(float, value.split(' '))))
              elif prefix == 'f':
                  
                # for face in value.split(' '):
                 self.faces.append([list(map(int , face.split('/'))) for face in value.split(' ')])
                 # if face.split('/') or face.split('//'):
                     # self.faces.append([list(map(int,face.split('//')))])
            
# cara = Obj('./stormtrooper.obj')
# print(cara.vertices)
# print(cara.faces)

def try_int(s, base=10, val=None):
  try:
    return int(s, base)
  except ValueError:
    return val


# class Obj(object):
#     def __init__(self, filename):
#         with open(filename) as f:
#             self.lines = f.read().splitlines()
#         self.vertices = []
#         self.vfaces = []
#         self.read()

#     def read(self):
#         for line in self.lines:
#             if line:
#                 prefix, value = line.split(' ', 1)
#                 if prefix == 'v':
#                     self.vertices.append(list(map(float, value.split(' '))))
#                 elif prefix == 'f':
#                     self.vfaces.append([list(map(try_int, face.split('/'))) for face in value.split(' ')])


# class Obj(object):

#  def __init__(self, filename):
#         with open(filename) as f:
#             self.lines = f.read().splitlines()
#         self.vertices = []
#         self.tvertices = []
#         self.vfaces = []
#         self.read()

#  def read(self):
#         for line in self.lines:
#             if line:
#                 try:
#                     prefix, value = line.split(' ', 1)
#                 except:
#                     prefix = ''
#                 if prefix == 'v':
#                     self.vertices.append(list(map(float, value.split(' '))))
#                 if prefix == 'vt':
#                     self.tvertices.append(list(map(float, value.split(' '))))                    
#                 elif prefix == 'f':
#                     self.vfaces.append([list(map(try_int, face.split('/'))) for face in value.split(' ')])

class Obj(object):
    # def __init__(self, filename):
    #     with open(filename) as f:
    #         self.lines = f.read().splitlines()
    #     self.vertices = []
    #     self.tvertices = []
    #     self.vfaces = []
    #     self.read()

    # def read(self):
    #     for line in self.lines:
    #         if line:
    #             try:
    #                 prefix, value = line.split(' ', 1)
    #             except:
    #                 prefix = ''
    #             if prefix == 'v':
    #                 self.vertices.append(list(map(float, value.split(' '))))
    #             if prefix == 'vt':
    #                 self.tvertices.append(list(map(float, value.split(' '))))                    
    #             elif prefix == 'f':
    #                 self.vfaces.append([list(map(try_int, face.split('/'))) for face in value.split(' ')])
    
    
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.lines = file.read().splitlines()

        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.read()

    def read(self):
        for line in self.lines:
            if line:
                if (len(line.split(" ")) > 1) :

                    prefix, value = line.split(' ', 1)

                    if prefix == 'v': # Vertices
                        self.vertices.append(list(map(float,value.split(' '))))
                    elif prefix == 'vn': # Vertives normales
                        self.normals.append(list(map(float,value.split(' '))))
                    elif prefix == 'vt': #Coordenada de texturas 
                        self.texcoords.append(list(map(float,value.split(' '))))
                    elif prefix == 'f': #Cara del poligono
                        self.faces.append([list(map(int,vert.split('/'))) for vert in value.split(' ')])




                    
class Texture(object):
    def __init__(self, path):
        self.path = path
        self.read()

    def read(self):
        image = open(self.path, "rb")
        # we ignore all the header stuff
        image.seek(2 + 4 + 4)  # skip BM, skip bmp size, skip zeros
        header_size = struct.unpack("=l", image.read(4))[0]  # read header size
        image.seek(2 + 4 + 4 + 4 + 4)
        
        self.width = struct.unpack("=l", image.read(4))[0]  # read width
        self.height = struct.unpack("=l", image.read(4))[0]  # read width
        self.pixels = []
        image.seek(header_size)
        for y in range(self.height):
            self.pixels.append([])
            for x in range(self.width):
                b = ord(image.read(1))
                g = ord(image.read(1))
                r = ord(image.read(1))
                self.pixels[y].append(color(r,g,b))
        image.close()

    def get_color(self, tx, ty, intensity=1):
        x = int(tx * self.width)
        y = int(ty * self.height)
        # return self.pixels[y][x]
        try:
            return bytes(map(lambda b: round(b*intensity) if b*intensity > 0 else 0, self.pixels[y][x]))
        except:
            pass  # what causes this
            
class TextureF(object):
    def __init__(self, path):
        self.path = path
        self.read()
        
    def read(self):
        image = open(self.path, 'rb')
        image.seek(10)
        headerSize = struct.unpack('=l', image.read(4))[0]

        image.seek(14 + 4)
        self.width = struct.unpack('=l', image.read(4))[0]
        self.height = struct.unpack('=l', image.read(4))[0]
        image.seek(headerSize)

        self.pixels = []

        for y in range(self.height):
            self.pixels.append([])
            for x in range(self.width):
                b = ord(image.read(1)) / 255
                g = ord(image.read(1)) / 255
                r = ord(image.read(1)) / 255
                self.pixels[y].append(color(r,g,b))

        image.close()

    def getColor(self, tx, ty):
        if tx >= 0 and tx <= 1 and ty >= 0 and ty <= 1:
            x = int(tx * self.width)
            y = int(ty * self.height)

            return self.pixels[y][x]
        else:
            return color(0,0,0)
        
 