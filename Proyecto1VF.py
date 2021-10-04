# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:54:15 2021

@author: hugo_
"""

#Operaciones matematicas a utilizar
def sum(x0, x1, y0, y1, z0, z1):
    arr_sum = []
    arr_sum.extend((x0 + x1, y0 + y1, z0 + z1))
    return arr_sum

def sub(x0, x1, y0, y1, z0, z1):
    arr_sub = []
    arr_sub.extend((x0 - x1, y0 - y1, z0 - z1))
    return arr_sub

def sub2(x0, x1, y0, y1):
    arr_sub = []
    arr_sub.extend((x0 - x1, y0 - y1))
    return arr_sub

def subVectors(vec1, vec2):
    subList = []
    subList.extend((vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]))
    return subList
    
def cross(v0, v1):
    arr_cross = []
    arr_cross.extend((v0[1] * v1[2] - v1[1] * v0[2], -(v0[0] * v1[2] - v1[0] * v0[2]), v0[0] * v1[1] - v1[0] * v0[1]))
    return arr_cross

def dot(norm, lX, lY, lZ):
    return ((norm[0] * lX) + (norm[1] * lY) + (norm[2] * lZ))

def norm(v0):
    if (v0 == 0):
        arr0_norm = []
        arr0_norm.extend((0,0,0))
        return arr0_norm

    return((v0[0]**2 + v0[1]**2 + v0[2]**2)**(1/2))

def frobeniusNorm(v0):
        return((v0[0]**2 + v0[1]**2 + v0[2]**2)**(1/2))

def div(v0, norm):
    if (norm == 0):
        arr0_norm = []
        arr0_norm.extend((0,0,0))
        return arr0_norm
    else:
        arr_div = []
        arr_div.extend((v0[0] / norm, v0[1] / norm, v0[2] / norm))
        return arr_div

def zeros_matrix(rows, cols):
    m = []
    while len(m) < rows:
        m.append([])
        while len(m[-1]) < cols:
            m[-1].append(0.0)

    return m

def matrix_multiply(m1, m2):
    rowsM1 = len(m1)
    colsM1 = len(m1[0])
    colsM2 = len(m2[0])
 
    c = zeros_matrix(rowsM1, colsM2)
    for i in range(rowsM1):
        for j in range(colsM2):
            total = 0
            for k in range(colsM1):
                total += m1[i][k] * m2[k][j]
            c[i][j] = total
 
    return c

def multiplyVM(v, m):
    result = []
    for i in range(len(m)):
        total = 0
        for j in range(len(v)):
            total += m[i][j] * v[j]
        result.append(total)
    return result  

def degToRad(number):
    pi = 3.141592653589793
    return number * (pi/180)

def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                return -1
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret

def multiply(dotNumber, normal):
    arrMul = []
    arrMul.extend((dotNumber * normal[0], dotNumber * normal[1], dotNumber * normal[2]))
    return arrMul


def baryCentric(Ax, Bx, Cx, Ay, By, Cy, Px, Py):
    try:
        u = ( ((By - Cy)*(Px - Cx) + (Cx - Bx)*(Py - Cy) ) /
              ((By - Cy)*(Ax - Cx) + (Cx - Bx)*(Ay - Cy)) )

        v = ( ((Cy - Ay)*(Px - Cx) + (Ax - Cx)*(Py - Cy) ) /
              ((By - Cy)*(Ax - Cx) + (Cx - Bx)*(Ay - Cy)) )

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w


import struct
from numpy import cos, sin, tan
from obj import Obj, TextureF
import random

def char(c):
  
  
  return struct.pack('=c', c.encode('ascii'))

def word(w):
  """
  Input: requires a number such that (-0x7fff - 1) <= number <= 0x7fff
         ie. (-32768, 32767)
  Output: 2 bytes
  Example:  
  >>> struct.pack('=h', 1)
  b'\x01\x00'
  """
  return struct.pack('=h', w)

def dword(d):
  """
  Input: requires a number such that -2147483648 <= number <= 2147483647
  Output: 4 bytes
  Example:
  >>> struct.pack('=l', 1)
  b'\x01\x00\x00\x00'
  """
  return struct.pack('=l', d)



def color(r, g, b):
    return bytes([int(b*255), int(g*255), int(r*255)])


BLACK = color(0,0,0)
WHITE = color(1,1,1)

pi = 3.1416

class Render(object):
     def __init__(self, width, height):
         
         self.current_color = WHITE
         self.bmp_color = BLACK
         
         self.CreateW(width, height)

         self.luzX, self.luzY, self.luzZ = 0, 0, 1
         self.luz = (self.luzX, self.luzY, self.luzZ)
         
         self.atexture = None
         self.amap = None
         self.aShader = None

         self.createViewMatrix()
         self.createProjectionMatrix()
         
         
     def display(self, filename='out.bmp'):
            """
            Displays the image, a external library (wand) is used, but only for convenience during development
            """
            self.write(filename)
        
            try:
              from wand.image import Image
              from wand.display import display
        
              with Image(filename=filename) as image:
                display(image)
            except ImportError:
              pass  # do nothing if no wand is installed
         
     def clear(self):
             self.pixels = [
                 [self.bmp_color for x in range(self.width)] 
                 for y in range(self.height)
                 ]
             self.zbuffer = [
                 [-float('inf') for x in range(self.width)]
                 for y in range(self.height)
                 ]
             
    
     def ClearC(self, r, g, b):
            red = int(r * 255)
            green = int(g * 255)
            blue = int(b * 255)
            self.bmp_color = color(red, green, blue)
            
     def CreateW(self, width, height):
        self.width = width
        self.height = height
        self.clear()
        self.ViewP(0, 0, width, height)
        
        
        
     def ViewP(self, x, y, width, height):
             self.viewPW = width
             self.viewPH = height
             self.viewPX = x
             self.viewPY = y

             self.viewportMatrix = [
                [width/2, 0, 0, x + width/2],
                [0, height/2, 0, y + height/2],
                [0, 0, 0.5, 0.5],
                [0, 0, 0, 1]
            ]
             
     
        
     def Vertex(self, x, y):
       
            vertexX = int((x+1)*(self.viewPW/2)+self.viewPX)
            vertexY = int((y+1)*(self.viewPH/2)+self.viewPY)
            try:
                self.pixels[vertexY][vertexX] = self.current_color
            except:
                pass

     def VertexC(self, x, y, color = None):
             if x >= self.width or x < 0 or y >= self.height or y < 0:
                 return

             try:
                self.pixels[y][x] = color or self.current_color
             except:
                pass
            
     def Color(self, r, g, b):
             red = int(r * 255)
             green = int(g * 255)
             blue = int(b * 255)
             self.current_color = color(red, green, blue)
             
     def createViewMatrix(self, camPosition = (0, 0, 0), camRotation = (0, 0, 0)):
             camMatrix = self.createModelMatrix(translate = camPosition, rotate = camRotation)
             self.viewMatrix = inverse(camMatrix)
             
     def createProjectionMatrix(self, n = 0.1, f = 1000, fov = 60):
             t = tan((fov * pi / 180) / 2) * n
             r = t * self.viewPW / self.viewPH

             self.projectionMatrix = [
                [n / r, 0, 0, 0],
                [0, n / t, 0, 0],
                [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                [0, 0, -1, 0]
                ]
            
     def write(self, filename):
            f = open(filename, 'bw')

            # File header (14 bytes)
            f.write(char('B'))
            f.write(char('M'))
            f.write(dword(14 + 40 + self.width * self.height * 3))
            f.write(dword(0))
            f.write(dword(14 + 40))
        
            # Image header (40 bytes)
            f.write(dword(40))
            f.write(dword(self.width))
            f.write(dword(self.height))
            f.write(word(1))
            f.write(word(24))
            f.write(dword(0))
            f.write(dword(self.width * self.height * 3))
            f.write(dword(0))
            f.write(dword(0))
            f.write(dword(0))
            f.write(dword(0))
        
            for x in range(self.height):
              for y in range(self.width):
                f.write(self.pixels[x][y])
        
            f.close()
            
     def shader(self,x,y):
      
      center_x, center_y = 215,215
      radius = 25
      
      center_x2, center_y2 = 215,215
      radius2 = 20
      
      if y >= 100 and y <= 160  + sin (x/3.1416):
          # return color(222,184,135)
          return color(194,155,97)
      
      elif y >= 160 and y <= 180 + sin (x/3.1416) :
          return color(222,184,135)
      
      elif (x-center_x)**2 + (y-center_y)**2 < radius**2 and (x-center_x2)**2 + (y-center_y2)**2 < radius2**2 :
            return color(128,64,0)
      
      elif (x-center_x)**2 + (y-center_y)**2 < radius**2:
          return color(205,133,63)
   
      elif y >= 180 and y <= 220 + sin (x/3.1416):
          return color(128,64,0)
      
      elif y >= 220 and y <= 280 + sin (x/3.1416):
          return color(222,184,135)
      
      elif y >= 280 and y <= 320 + sin (x/3.1416):
          return color(128,64,0)
      
      elif y >= 330 and y <= 337  + sin (x/3.1416):
          # return color(222,184,135)
          return color(194,155,97)
      
      elif y >= 320 and y <= 340 + sin (x/3.1416) :
          return color(222,184,135)
      
      elif y >= 340 and y <= 400 + sin (x/3.1416) :
          # return color(222,184,135)
          return color(194,155,97)
      
      
      
     
      
      # else:
      #      return color(194,155,97)
       
     
          
      
      
      
      # if y > 200 and y < 210:
      #     return color(255,255,255)
      
      # else:
      #     return color(194,155,97)
             
     def Line(self, x0, y0, x1, y1):
           
            x0 = int((x0 + 1) * (self.viewPW/2) + self.viewPX)
            x1 = int((x1 + 1) * (self.viewPW/2) + self.viewPX)
            y0 = int((y0 + 1) * (self.viewPH/2) + self.viewPY)
            y1 = int((y1 + 1) * (self.viewPH/2) + self.viewPY)
    
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
    
            
            steep = dy > dx
    
           
            if steep:
                x0, y0 = y0, x0
                x1, y1 = y1, x1
    
            if x0 > x1:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
    
            offset = 0 
            limit = 0.5
    
            m = dy/dx
            y = y0
    
          
            for x in range(x0, x1+1):
                if steep:
                    self.glVertexC(y, x)
                else:
                    self.glVertexC(x, y)
    
                offset += m
    
                if offset >= limit:
                    y += 1 if y0 < y1 else -1
                    limit += 1
                    
     def LineC(self, x0, y0, x1, y1):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
    
            steep = dy > dx
    
           
            if steep:
                x0, y0 = y0, x0
                x1, y1 = y1, x1
    
            if x0 > x1:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
    
            offset = 0 
            limit = 0.5
    
            try:
                m = dy/dx
            except ZeroDivisionError:
                pass
            else:
                y = y0
    
                for x in range(x0, x1+1):
                    if steep:
                        self.glVertexCoord(y, x)
                    else:
                        self.glVertexCoord(x, y)
    
                    offset += m
    
                    if offset >= limit:
                        y += 1 if y0 < y1 else -1
                        limit += 1
                        
                        
     def transform(self, vertex, vMatrix):
            augVertex = (vertex[0], vertex[1], vertex[2], 1)
            transVertex = multiplyVM(augVertex, vMatrix)
            transVertex = (transVertex[0]/transVertex[3],
                       transVertex[1]/transVertex[3],
                       transVertex[2]/transVertex[3])
    
            return transVertex
        
     def camTransform(self, vertex):
             augVertex = (vertex[0], vertex[1], vertex[2], 1)

             transVertex1 = matrix_multiply(self.viewportMatrix, self.projectionMatrix)
             transVertex2 = matrix_multiply(transVertex1, self.viewMatrix)
             transVertex = multiplyVM(augVertex, transVertex2)
    
             transVertex = (transVertex[0] / transVertex[3],
                               transVertex[1] / transVertex[3],
                               transVertex[2] / transVertex[3])
             return transVertex
        
     def dirTransform(self, vertex, vMatrix):
            augVertex = (vertex[0], vertex[1], vertex[2], 0)
            transVertex = multiplyVM(augVertex, vMatrix)
            transVertex = (transVertex[0],
                       transVertex[1],
                       transVertex[2])

            return transVertex
        
        
     def createModelMatrix(self, translate=(0,0,0), scale=(1,1,1), rotate=(0,0,0)):
        
            translateMatrix = [
                [1, 0, 0, translate[0]],
                [0, 1, 0, translate[1]],
                [0, 0, 1, translate[2]],
                [0, 0, 0, 1]
            ]
    
            scaleMatrix = [
                [scale[0], 0, 0, 0],
                [0, scale[1], 0, 0],
                [0, 0, scale[2], 0],
                [0, 0, 0, 1]
            ]
    
            rotationMatrix = self.createRotationMatrix(rotate)
    
           
            finalObjectMatrix1 = matrix_multiply(translateMatrix, rotationMatrix)
            finalObjectMatrix = matrix_multiply(finalObjectMatrix1, scaleMatrix)
    
            return finalObjectMatrix
        
        
        
     def createRotationMatrix(self, rotate=(0,0,0)):
            pitch = degToRad(rotate[0])
            yaw = degToRad(rotate[1])
            roll = degToRad(rotate[2])
    
            rotationX = [
                [1, 0, 0, 0],
                [0, cos(pitch), -sin(pitch), 0],
                [0, sin(pitch), cos(pitch), 0],
                [0, 0, 0, 1]
            ]
    
            rotationY = [
                [cos(yaw), 0, sin(yaw), 0],
                [0, 1, 0, 0],
                [-sin(yaw), 0, cos(yaw), 0],
                [0, 0, 0, 1]
            ]
    
            rotationZ = [
                [cos(roll), -sin(roll), 0, 0],
                [sin(roll), cos(roll), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
    
            finalMatrixRotation1 = matrix_multiply(rotationX, rotationY)
            finalMatrixRotation = matrix_multiply(finalMatrixRotation1, rotationZ)
            
            return finalMatrixRotation
        
        
     def load(self, filename, translate=(0,0,0), scale=(1,1,1), rotate=(0,0,0)):
            model = Obj(filename)
            modelMatrix = self.createModelMatrix(translate, scale, rotate)
            rotationMatrix = self.createRotationMatrix(rotate)
    
            for face in model.faces:
    
                vertCount = len(face)
    
                v0 = model.vertices[ face[0][0] - 1 ]
                v1 = model.vertices[ face[1][0] - 1 ]
                v2 = model.vertices[ face[2][0] - 1 ]
    
                v0 = self.transform(v0, modelMatrix)
                v1 = self.transform(v1, modelMatrix)
                v2 = self.transform(v2, modelMatrix)
    
                Ax, Bx, Cx = int(v0[0]), int(v1[0]), int(v2[0])
                Ay, By, Cy = int(v0[1]), int(v1[1]), int(v2[1])
                Az, Bz, Cz = int(v0[2]), int(v1[2]), int(v2[2])
                A = (Ax, Ay, Az)
                B = (Bx, By, Bz) 
                C = (Cx, Cy, Cz)
    
                v0 = self.camTransform(v0)
                v1 = self.camTransform(v1)
                v2 = self.camTransform(v2)
    
    
                # Si los vertices son mayores a 4 se asigna un 3 valor en las dimensiones
                if vertCount > 3: 
                    v3 = model.vertices[face[3][0] - 1]
                    v3 = self.transform(v3, modelMatrix)
                    # D = v3
                    Dx = int(v3[0])
                    Dy = int(v3[1])
                    Dz = int(v3[2])
                    D = (Ax, By, Cz)
    
                    v3 = self.camTransform(v3)
    
                try:
                    vt0 = model.texcoords[face[0][1] - 1]
                    vt1 = model.texcoords[face[1][1] - 1]
                    vt2 = model.texcoords[face[2][1] - 1]
                    vt0X, vt0Y = vt0[0], vt0[1]
                    vt1X, vt1Y = vt1[0], vt1[1]
                    vt2X, vt2Y = vt2[0], vt2[1]
    
                    if vertCount > 3:
                        vt3 = model.texcoords[face[3][1] - 1]
                        vt3X, vt3Y = vt3[0], vt3[1]
    
                    # Normales de los vertices del obj
                    vn0 = model.normals[face[0][2] - 1]
                    vn1 = model.normals[face[1][2] - 1]
                    vn2 = model.normals[face[2][2] - 1]
    
                    vn0 = self.dirTransform(vn0, rotationMatrix)
                    vn1 = self.dirTransform(vn1, rotationMatrix)
                    vn2 = self.dirTransform(vn2, rotationMatrix)
    
                    if vertCount > 3:
                        vn3 = model.normals[face[3][2] - 1]
                        vn3 = self.dirTransform(vn3, rotationMatrix)
    
                except:
                    pass
    
    
                self.triangle(Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz, vt0X, vt1X, vt2X, vt0Y, vt1Y, vt2Y, verts = (A, B, C), normals = (vn0, vn1, vn2))
                if vertCount > 3:
                    self.triangle(Ax, Cx, Dx, Ay, Cy, Dy, Az, Cz, Dz, vt0X, vt2X, vt3X, vt0Y, vt2Y, vt3Y, verts = (A, B, D), normals = (vn0, vn2, vn3))
                
                
            
     def triangle(self, Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz, taX, tbX, tcX, taY, tbY, tcY, normals = (), verts = (), _color = None):
            minX = int(min(Ax, Bx, Cx))
            minY = int(min(Ay, By, Cy))
            maxX = int(max(Ax, Bx, Cx))
            maxY = int(max(Ay, By, Cy))
    
            for x in range(minX, maxX + 1):
                for y in range(minY, maxY + 1):
                    if x >= self.width or x < 0 or y >= self.height or y < 0:
                        continue
    
                    u, v, w = baryCentric(Ax, Bx, Cx, Ay, By, Cy, x, y)
    
                    if u >= 0 and v >= 0 and w >= 0:
                        z = Az * u + Bz * v + Cz * w
                        if z > self.zbuffer[y][x]:
                            
                            r, g, b = self.aShader(
                                self,
                                verts = (Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz),
                                vecVerts = verts,
                                baryCoords = (u, v, w),
                                texCoords = (taX, tbX, tcX, taY, tbY, tcY),
                                normals = normals,
                                color = _color or self.current_color)
    
                            self.VertexC(x, y, color(r, g, b))
                            self.zbuffer[y][x] = z
                
                

    
def normalMap(render, **kwargs):
        Ax, Bx, Cx, Ay, By, Cy, Az, Bz, Cz = kwargs['verts']
        A, B, C = kwargs['vecVerts']
        u, v, w = kwargs['baryCoords']
        taX, tbX, tcX, taY, tbY, tcY = kwargs['texCoords']
        na, nb, nc = kwargs['normals']
        b, g ,r = kwargs['color']
    
        
        ta = (taX, taY)
        tb = (tbX, tbY)
        tc = (tcX, tcY)
    
        b /= 255
        g /= 255
        r /= 255
    
        tx = taX * u + tbX * v + tcX * w
        ty = taY * u + tbY * v + tcY * w
    
        if render.atexture:
            texColor = render.atexture.getColor(tx, ty)
            b *= texColor[0] / 255
            g *= texColor[1] / 255
            r *= texColor[2] / 255
    
        nx = na[0] * u + nb[0] * v + nc[0] * w
        ny = na[1] * u + nb[1] * v + nc[1] * w
        nz = na[2] * u + nb[2] * v + nc[2] * w
        normal = (nx, ny, nz)
    
        if render.amap:
            texNormal = render.amap.getColor(tx, ty)
            texNormal = [ (texNormal[2] / 255) * 2 - 1,
                          (texNormal[1] / 255) * 2 - 1,
                          (texNormal[0] / 255) * 2 - 1]
    
            texNormal = div(texNormal, frobeniusNorm(texNormal))
    
            # B - A
            edge1 = sub(B[0], A[0], B[1], A[1], B[2], A[2])
            # C - A
            edge2 = sub(C[0], A[0], C[1], A[1], C[2], A[2])
            # tb - ta 
            deltaUV1 = sub2(tb[0], ta[0], tb[1], ta[1])
            # tc - ta
            deltaUV2 = sub2(tc[0], ta[0], tc[1], ta[1])
    
            tangent = [0,0,0]
            f = 1 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
            tangent[0] = f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0])
            tangent[1] = f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1])
            tangent[2] = f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
            tangent = div(tangent, frobeniusNorm(tangent))
            tangent = div(tangent, frobeniusNorm(tangent))
            tangent = subVectors(tangent, multiply(dot(tangent, normal[0], normal[1], normal[2]), normal))
            tangent = tangent / frobeniusNorm(tangent)
    
            bitangent = cross(normal, tangent)
            bitangent = bitangent / frobeniusNorm(bitangent)
    
    
            tangentMatrix = [
                [tangent[0],bitangent[0],normal[0]],
                [tangent[1],bitangent[1],normal[1]],
                [tangent[2],bitangent[2],normal[2]]
            ]
    
            light = render.luz
            light = multiplyVM(light, tangentMatrix)
            light = div(light, frobeniusNorm(light))
    
            intensity = dot(texNormal, light[0], light[1], light[2])
        else:
            intensity = dot(normal, render.luz[0], render.luz[1], render.luz[2])
    
        b *= intensity
        g *= intensity
        r *= intensity
    
        if intensity > 0:
            return r, g, b
        else:
            return 0,0,0           
                
    
r = Render(3840, 2160)

bg = TextureF('./mars2.bmp')
r.pixels = bg.pixels


r.atexture = TextureF('./StormtrooperT.bmp')
r.amap = TextureF('./Stormtrooper3.bmp')
r.aShader = normalMap
r.load('./stormtrooper.obj', (1100, 200, 0), (170, 170, 170), (0, 30 , 0))

r.atexture = TextureF('./craft-texture.bmp')
r.amap = TextureF('./craftT2.bmp')
r.asShader = normalMap
r.load('./craft.obj', (2600, 200, 0), (100, 100, 100), (0, 0, 0))


r.atexture = TextureF('./Clouds_2K.bmp')
r.amap = TextureF('./Clouds_2K.bmp')
r.aShader = normalMap
r.load('./Mars_2K.obj', (1100, 1700, 0), (150, 150, 150), (0, 180, 0))


r.atexture = TextureF('./E45T.bmp')
r.amap = TextureF('./E45T2.bmp')
r.asShader = normalMap
r.load('./E 45 Aircraft_obj.obj', (2600, 1300, 0), (100, 100, 100), (0, 60, 0))


r.write('out.bmp')