from abc import ABCMeta

import numpy as np
import plotly.graph_objects as px
from simpleai.search import SearchProblem, breadth_first, depth_first, astar, uniform_cost, greedy
import math
import time

start_time= time.time()

mars_map = np.load('mars_map.npy')
nr, nc = mars_map.shape
# print(nr,nc)

scale = 10.0174

# Para rutas más amplias cambiar las coordenadas en las líneas siguientes:
# -para ruta de menos de 500m :Coord inicial (x= 3905, y= 10187),  Coord final (x= 4000, y= 9737) 
# - para ruta de más de 1000 y menos de 5000m: (x= 6100, y= 6420),  Coord final (x= 4087, y= 8244) 
# -para ruta de más de 10,000m (20,000m en específico): Coord inicial (x= 4798, y= 2756),  Coord final (x= 1582, y= 13273) 

# renglón columna de inicio x = 2850 y = 6400
ci = round(2850/ scale)
ri = nr - round(6400/ scale)
print("-" * 55)
print("Coord inicial dentro de la matriz (x,y): ", ci, ",", ri)

# renglón columna goal x = 3150 y = 6800
cf = round(3150/ scale)
rf = nr - round(6800/ scale)
print("Coord final dentro de la matriz (x,y): ", cf, ",", rf)

# Altura del punto de inicio
print("-" * 55)
print("Altura inicial:", mars_map[ri, ci], "metros")

# Altura del punto final
print("Altura final:", mars_map[rf, cf], "metros")
print("-" * 55)

# Clases

costos = {  # lista de costos al ejecutar cada movimiento
    "arriba": 1.0,
    "abajo": 1.0,
    "izq": 1.0,
    "der": 1.0,
    "arriba izq": 1.4,
    "arriba der": 1.4,
    "abajo izq": 1.4,
    "abajo der": 1.4,
}


# Definición de la clase


class MarsPath(SearchProblem, metaclass=ABCMeta):

    def __init__(self, board):  # constructor de la clase

        # Inicialización de los estados
        self.board = board
        self.goal = (0, 0)

        self.initial = (ci, ri)  # Llamada a la función de encontrar posición
        self.goal = (cf, rf)

        super(MarsPath, self).__init__(initial_state=self.initial)  # Regresa el estado de nuevo a la clase

    def actions(self, state):
        actions = []
        for action in list(costos.keys()):
            xact, yact = state  # Revisa las acciones en la lista de costos
            posx, posy = self.result(state, action)  # Nueva posición dependiendo de la acción realizada de la lista
            # Revisa que no haya un obstáculo en esa nueva posición
            if self.board[posy][posx] != -1 and abs(self.board[yact][xact] - self.board[posy][posx]) <= 0.25:
                actions.append(action)  # Agrega la acción que si se pudo realizar a la lista
        return actions

    def result(self, state, action):
        x, y = state

        # Modifica el estado dependiendo de la lista de acciones

        if action.count("arriba"):
            y -= 1
        if action.count("abajo"):
            y += 1
        if action.count("izq"):
            x -= 1
        if action.count("der"):
            x += 1

        new_state = (x, y)  # Regresa el nuevo estado
        return new_state

    def is_goal(self, state):  # Objetivo
        return state == self.goal

    def cost(self, state, action, state2):  # Costos
        return costos[action]

    def heuristic(self, state):  # Calcula el aproximado de la distancia del estado al estado objetivo
        x, y = state
        x_g, y_g = self.goal
        return math.sqrt((x - x_g)**2 + (y - y_g)**2)  # Euclidean Distance


# Resolver el laberinto
problem = MarsPath(mars_map)
result = astar(problem, graph_search=True)
path = [x[1] for x in result.path()]
path_x = []
path_y = []
path_z = []
for i in path:
    u = i
    x = u[0]
    y = u[1]
    path_x.append(x)
    path_y.append(y)
    z = mars_map[y][x]
    path_z.append(z)

sumad=0
cont=0

for i in range(len(path_x)):
    path_x[i] = path_x[i] * scale

for i in range(len(path_y)):
    path_y[i] = (nr - path_y[i]) * scale

while cont < len(path)-1:
    s = math.sqrt((path_x[cont+1]-path_x[cont])**2+(path_y[cont+1]-path_y[cont])**2)
    sumad+=s
    cont+=1
print("Distancia recorrida: ",round(sumad,4))

tiempo = time.time()-start_time
print('tiempo de ejecución: ', tiempo)

# Printing path

x = scale * np.arange(mars_map.shape[1])
y = scale * np.arange(mars_map.shape[0])
X, Y = np.meshgrid(x, y)
fig = px.Figure(data=[px.Surface(x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin=0,
                                 lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
                                 lightposition=dict(x=0, y=nr / 2, z=2 * mars_map.max())),
                      px.Scatter3d(x=path_x, y=path_y, z=path_z, name='path', mode="markers",
                                   marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Bluered", size=4))],
                layout=px.Layout(scene_aspectmode='manual',
                                 scene_aspectratio=dict(x=1, y=nr / nc, z=max(mars_map.max() / x.max(), 0.2)),
                                 scene_zaxis_range=[0, mars_map.max()]))
fig.show()
