import random
import math
import matplotlib.pyplot as plt

def generate_map(num_cities):
    city_map = [[0] * num_cities for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            road_length = random.randint(10, 100)
            city_map[i][j] = road_length
            city_map[j][i] = road_length
    return city_map

def save_map(city_map, filename):
    with open(filename, 'w') as f:
        for row in city_map:
            f.write(' '.join(map(str, row)) + '\n')

def load_map(filename):
    city_map = []
    with open(filename, 'r') as f:
        for line in f:
            city_map.append(list(map(int, line.strip().split())))
    return city_map


class Ant:
    def __init__(self, city_map, pheromone, alpha, beta):
        self.city_map = city_map
        self.pheromone = pheromone
        self.alpha = alpha
        self.beta = beta
        self.num_cities = len(city_map)

        self.visited_cities = []
        self.visited_edges = set()
        self.total_distance = 0

    def select_next_city(self, current_city):
        unvisited_cities = [i for i in range(self.num_cities) if i not in self.visited_cities]
        probabilities = [((self.pheromone[current_city][j])**self.alpha) * ((self.city_map[current_city][j])**self.beta) for j in unvisited_cities]
        total_probability = sum(probabilities)
        probabilities = [prob / total_probability for prob in probabilities]
        next_city = random.choices(unvisited_cities, probabilities)[0]
        return next_city


    def find_path(self):
        current_city = random.randint(0, self.num_cities - 1)
        self.visited_cities.append(current_city)

        while len(self.visited_cities) < self.num_cities:
            next_city = self.select_next_city(current_city)
            self.visited_cities.append(next_city)
            self.visited_edges.add((current_city, next_city))
            self.total_distance += self.city_map[current_city][next_city]
            current_city = next_city

        self.total_distance += self.city_map[self.visited_cities[-1]][self.visited_cities[0]]
        self.visited_edges.add((self.visited_cities[-1], self.visited_cities[0]))



class AntColony:
    def __init__(self, city_map, num_ants=10, evaporation_rate=0.1, alpha=4, beta=2):
        self.city_map = city_map
        self.num_cities = len(city_map)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta

        self.pheromone = [[1 / (self.num_cities * self.num_ants) for _ in range(self.num_cities)] for _ in range(self.num_cities)]


    def _update_pheromone(self, ants):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.pheromone[i][j] *= (1 - self.evaporation_rate)
                for ant in ants:
                    if (i, j) in ant.visited_edges or (j, i) in ant.visited_edges:
                        self.pheromone[i][j] += 1 / ant.total_distance

    def run(self, iterations=100):
        best_route = None
        best_distance = math.inf

        for _ in range(iterations):
            ants = [Ant(self.city_map, self.pheromone, self.alpha, self.beta) for _ in range(self.num_ants)]
            for ant in ants:
                ant.find_path()

                if ant.total_distance < best_distance:
                    best_distance = ant.total_distance
                    best_route = ant.visited_cities
            self._update_pheromone(ants)

        return best_route, best_distance


    def print_parameters(self):
        print("Кількість мурах у «мурашнику»:", self.num_ants)
        print("Константа випаровування ферменту:", self.evaporation_rate)
        print("Співвідношення α/β:", self.alpha, "/", self.beta)


    def plot_route(self, route, city_coordinates):
        plt.figure(figsize=(10, 6))
        for i in range(self.num_cities):
            if(i == best_route[0] or i==best_route[len(best_route)-1]):
               plt.scatter(city_coordinates[i][0], city_coordinates[i][1], color= 'red')
            else:
               plt.scatter(city_coordinates[i][0], city_coordinates[i][1], color='black')
            plt.text(city_coordinates[i][0], city_coordinates[i][1], str(i), fontsize=12)

        for i in range(len(route) - 1):
            city1 = route[i]
            city2 = route[i + 1]
            x_values = [city_coordinates[city1][0], city_coordinates[city2][0]]
            y_values = [city_coordinates[city1][1], city_coordinates[city2][1]]
            plt.plot(x_values, y_values,  color='blue')
            arrow_x = x_values[1] - 0.1 * (x_values[1] - x_values[0])
            arrow_y = y_values[1] - 0.1 * (y_values[1] - y_values[0])
        
            plt.annotate('', xy=(x_values[1], y_values[1]), xytext=(arrow_x, arrow_y),
               arrowprops=dict(facecolor='blue', arrowstyle='->'))

        plt.title("Best Route")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    #num_cities = random.randint(25, 35)
    #city_map = generate_map(num_cities)
    #save_map(city_map, "C:\\Users\\user\\Desktop\\AI\\city_map.txt")

    loaded_map = load_map("C:\\Users\\user\\Desktop\\AI\\city_map.txt")
    algorithm = AntColony(loaded_map)
    best_route, best_distance = algorithm.run(iterations=100)
    print("Шлях:", best_route)
    print("Відстань:", best_distance)
    algorithm.print_parameters()
    algorithm.plot_route(best_route, loaded_map)
