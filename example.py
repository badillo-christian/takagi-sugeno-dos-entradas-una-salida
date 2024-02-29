from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from lkfuzzy import *


def main():
    food = InputVariable('food', range=[0, 10])
    service = InputVariable('service', range=[0, 10])
    
    food['baja'] = TriangularFunction(0, 0, 5)
    food['regular'] = TriangularFunction(0, 5, 10)
    food['excelente'] = TriangularFunction(5, 10, 10)

    service['baja'] = TriangularFunction(0, 0, 5)
    service['regular'] = TriangularFunction(0, 5, 10)
    service['excelente'] = TriangularFunction(5, 10, 10)

    rules = [
        Rule(food['baja'] & service['baja'], 0),
        Rule(food['baja'] & service['regular'], 5),
        Rule(food['regular'] & service['baja'], 8),
        Rule(food['baja'] & service['excelente'], 10),
        Rule(food['excelente'] & service['baja'], 9),
        Rule(food['regular'] & service['regular'], 10),
        Rule(food['regular'] & service['excelente'], 12),
        Rule(food['excelente'] & service['regular'], 15),
        Rule(food['excelente'] & service['excelente'], 20),
    ]

    system = FuzzySystem(rules)

    test_on_examples(system)
    draw_heatmap(system)
    draw_surface(system)


def test_on_examples(system):
    ValorPrueba = namedtuple('valorPrueba', 'food service')

    valoresPrueba = [
        ValorPrueba(10, 10),
        ValorPrueba(4, 4),
        ValorPrueba(0, 0),
        ValorPrueba(10, 0),
        ValorPrueba(0, 10),
        ValorPrueba(8, 8),
        ValorPrueba(2, 6),
        ValorPrueba(6, 2),
        ValorPrueba(9, 1),
        ValorPrueba(1, 9),
        ValorPrueba(3, 7),
        ValorPrueba(0, 0),
    ]

    for valorPrueba in valoresPrueba:
        
        propina = system.compute(food=valorPrueba.food, service=valorPrueba.service)
        print(f'valor entrada calificación comida: {valorPrueba.food:2}/10, valor entrada calificación servicio: {valorPrueba.service:2}/10 -> propina: {propina:.1f}%')


def draw_heatmap(system):
    resolution = 20
    food_values = np.linspace(0, 10, resolution)
    service_values = np.linspace(0, 10, resolution)

    food_grid, service_grid = np.meshgrid(food_values, service_values)
    tip_grid = np.zeros_like(food_grid)
    for food_index in range(resolution):
        for service_index in range(resolution):
            food_value = food_values[food_index]
            service_value = service_values[service_index]
            tip_grid[food_index, service_index] = system.compute(food=food_value, service=service_value)

    fig, ax = plt.subplots()
    cp = ax.contourf(service_grid, food_grid, tip_grid, 200, cmap=plt.cm.RdYlGn,extend='both')

    ax.set_xlabel('Calidad Servicio')
    ax.set_ylabel('Calidad Comida')
    ax.set_title('Comida y Servicio vs % Propina')
    ax.grid()
    fig.colorbar(cp, label='% Propina')
    fig.show()
    plt.pause(1)
            
def draw_surface(system):
    resolution = 20
    food_values = np.linspace(0, 10, resolution)
    service_values = np.linspace(0, 10, resolution)

    food_grid, service_grid = np.meshgrid(food_values, service_values)
    tip_grid = np.zeros_like(food_grid)
    for food_index in range(resolution):
        for service_index in range(resolution):
            food_value = food_values[food_index]
            service_value = service_values[service_index]
            tip_grid[food_index, service_index] = system.compute(food=food_value, service=service_value)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(service_grid, food_grid, tip_grid, edgecolor='royalblue', lw=0.5, rstride=1, cstride=1,
                alpha=0.2)
    
    ax.contour(service_grid, food_grid, tip_grid, zdir='z', offset=-20, cmap='RdYlGn')
    ax.contour(service_grid, food_grid, tip_grid, zdir='x', offset=-0, cmap='RdYlGn')
    ax.contour(service_grid, food_grid, tip_grid, zdir='y', offset=10, cmap='RdYlGn')

    ax.set(xlim=(0, 10), ylim=(0, 10), zlim=(0, 20), xlabel='calidad servicio', ylabel='calidad comida', zlabel='% Propina')

    plt.show()
    plt.pause(1)

if __name__ == '__main__':
    main()
