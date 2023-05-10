import numpy as np
import plotly.graph_objects as px
import math
import time

start_time= time.time()

crater_map = np.load('crater_map.npy')
nr, nc = crater_map.shape
# print(nr,nc)

scale = 10.04502

# renglón columna de inicio x = 3350 y = 5800
ci = round(3350/ scale)
ri = nr - round(5800/ scale)
print("-" * 55)
print("Coord inicial dentro de la matriz (x,y): ", ci, ",", ri)

# Altura del punto de inicio
print("-" * 55)
print("Altura inicial:", crater_map[ri, ci], "metros")

# Definición de la clase

class Coordinate:

    def __init__(self, state):  # constructor de la clase

        self.state = state

                     # Regresa el estado de nuevo a la clase

    def neighbors(self):
        # Modifica el estado dependiendo de la lista de acciones

        xcor= self.state[0]
        ycor= self.state[1]
        neigh_list=[]

        opciones=[[xcor+1, ycor],[xcor-1, ycor],
                [xcor, ycor+1],[xcor, ycor-1],
                [xcor+1, ycor+1],[xcor+1, ycor-1],
                [xcor-1, ycor+1], [xcor-1, ycor-1]]
        
        for i in range(len(opciones)):
            xnew= opciones[i][0]
            ynew= opciones[i][1]
            if crater_map[ynew][xnew] != -1 and abs(crater_map[ycor][xcor] - crater_map[ynew][xnew]) <= 2.0:
                neigh_list.append(Coordinate((xnew,ynew)))  

         # Regresa el nuevo vecino
        return neigh_list

    def cost(self):  # Costos
        x= self.state[0]
        y=self.state[1]
        return crater_map[y][x]
    
if __name__ == '__main__':

    path = []

    c_inicial = Coordinate((ci,ri))

    cost0 = Coordinate.cost(c_inicial)  # Costo inicial
    
    step = 0  # Step count
     

    path_x=[]
    path_y=[]
    path_z=[]

    path_x.append(ci)
    path_y.append(ri)
    path_z.append(crater_map[ri][ci])
    
    while step < 10000 and cost0 > 0:

        step += 1 
        # Get random neighbor
        neighbors_p= c_inicial.neighbors()

        for i in range(len(neighbors_p)):
            costo=[]
            vecino= neighbors_p[i].cost()
            costo.append(vecino)
        
        min_index = costo.index(min(costo))
        nachbarn= neighbors_p[min_index]

        new_cost = nachbarn.cost()

        # Test neighbor 
        if new_cost < cost0:
            c_inicial = nachbarn
            cost0 = new_cost

            path_x.append(c_inicial.state[0])
            path_y.append(c_inicial.state[1])
            path_z.append(crater_map[c_inicial.state[1]][c_inicial.state[0]])

        else: 
            temp = c_inicial
            c_inicial = nachbarn
            nachbarn = temp

        
        print('Iteration: ', step, "    Cost: ", cost0)


sumad=0
cont=0

for i in range(len(path_x)):
    path_x[i] = path_x[i] * scale

for i in range(len(path_y)):
    path_y[i] = (nr - path_y[i]) * scale


tiempo = time.time()-start_time
print('tiempo de ejecución: ', tiempo)

# Printing path

x = scale * np.arange(crater_map.shape[1])
y = scale * np.arange(crater_map.shape[0])

X, Y = np.meshgrid(x, y)
fig = px.Figure(data=[px.Surface(x=X, y=Y, z=np.flipud(crater_map), colorscale='hot', cmin=0,
                                 lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
                                 lightposition=dict(x=0, y=nr / 2, z=2 *  crater_map.max())),
                      px.Scatter3d(x=path_x, y=path_y, z=path_z, name='path', mode="markers",
                                   marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Bluered", size=4))],
                layout=px.Layout(scene_aspectmode='manual',
                                 scene_aspectratio=dict(x=1, y=nr / nc, z=max(crater_map.max() / x.max(), 0.2)),
                                 scene_zaxis_range=[0, crater_map.max()]))
fig.show()
