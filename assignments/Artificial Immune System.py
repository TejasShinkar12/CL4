# ARTIFICIAL IMMUNE SYSTEM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data, labels = make_classification(n_features=10, n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state = 22)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.transform(X_train)
num_detectors = 10

detectors = np.random.rand(num_detectors, 10)
detectors_labels = np.zeros(num_detectors)

for i in range(100):
    for j in range(num_detectors):
        nearest = np.argmax(np.linalg.norm(X_train - detectors[j], axis = 1))
        detectors[j] = detectors[j] + 0.1 * (X_train[nearest] - detectors[j])
        detectors_labels[j] = y_train[nearest]

predictions = []

for x in X_test:
    idx = np.argmax(np.linalg.norm(detectors - x, axis = 1))
    predictions.append(detectors_labels[idx])

print("ACCURACY : ", accuracy_score(predictions, y_test))






#ANT COLONY OPTIMIZATION
import random, math

cities = {0: (0, 0), 1: (1, 5), 2: (5, 2), 3: (6, 6), 4: (10, 3)}
dist = {(i, j): math.dist(cities[i], cities[j]) for i in cities for j in cities if i != j}
pher = {edge: 1.0 for edge in dist}


ants, iters, alpha, beta, evap, Q = 10, 100, 1.0, 5.0, 0.5, 100
best_tour, best_len = None, float("inf")

def choose(city, visited):
    probs = [(c, (pher[(city, c)] ** alpha) * ((1 / dist[(city, c)]) ** beta))
             for c in cities if c not in visited]
    total = sum(p for _, p in probs)
    r, cum = random.uniform(0, total), 0
    for c, p in probs:
        cum += p
        if r <= cum: return c
    return probs[-1][0]

def tour():
    path = [random.choice(list(cities))]
    visited = set(path)
    while len(path) < len(cities):
        path.append(choose(path[-1], visited))
        visited.add(path[-1])
    return path + [path[0]]

for _ in range(iters):
    tours = [(t := tour(), sum(dist[(t[i], t[i+1])] for i in range(len(t)-1))) for _ in range(ants)]
    for t, l in tours:
        if l < best_len: best_tour, best_len = t, l
    for e in pher: pher[e] *= (1 - evap)
    for t, l in tours:
        for i in range(len(t)-1):
            a, b = t[i], t[i+1]
            pher[(a, b)] += Q / l
            pher[(b, a)] += Q / l

print("\nBest Tour Found:")
print(" -> ".join(map(str, best_tour)))
print(f"Total Distance: {best_len:.2f}")






# DISTRIUTED SYSTEM
import itertools
import random

servers = ['A', 'B', 'C', 'D']

rr = itertools.cycle(servers)

def rr_req():
    return next(rr)

def rand():
    return random.choice(servers)


def simulate(servers, strategy, num_req = 10):
    assignment = {s : 0 for s in servers}

    for _ in range(num_req):
        if strategy == "rr" :
            srv = rr_req()

        elif strategy == "rand":
            srv = rand()

        else:
            print("Error")

        assignment[srv] += 1

    return assignment

print(simulate(servers, "rr"))
print(simulate(servers, "rand"))





#  FUZZY SET

# Example fuzzy sets over U = {a, b, c}
A = {'a': 0.2, 'b': 0.7, 'c': 1.0}
B = {'a': 0.5, 'b': 0.3, 'd': 0.8}

def union(A, B):
    return {u : max(A.get(u, 0), B.get(u, 0)) for u in set(A) | set(B)}

def intersection(A, B):
    return {u : min(A.get(u, 0), B.get(u, 0)) for u in set(A) | set(B)}

def complement(A):
    return {u : 1 - mu for u, mu in A.items()}

def cart(A, B):
    return {(x, y) : min(mu_x, mu_y) for x, mu_x in A.items() for y, mu_y in B.items()}


# RPC

# SERVER
from xmlrpc.server import SimpleXMLRPCServer

def fact(n):
    if n == 0:
        return 1

    result = 1

    for i in range(1, n + 1):
        result *= i

    return result


if __name__ == "__main__":
    server = SimpleXMLRPCServer(("localhost", 8000))
    server.register_function(fact, 'factorial')
    print("Server listening on  8000")
    server.serve_forever()


# CLIENT
import xmlrpc.client

if __name__ == "__main__":
    proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")
    n = int(input("Enter num : "))
    result = proxy.factorial(n)
    print("FACT IS : ", result)





# RMI

#SERVER

import Pyro4
import threading

@Pyro4.expose
class StringConcatenator:
    def concatenate(self, str1, str2):
        print(f"Received: {str1}, {str2}")
        return str1 + str2

def start_server():
    daemon = Pyro4.Daemon(host="localhost")  # Bind to localhost
    uri = daemon.register(StringConcatenator)
    print("Server running. URI:", uri)

    # Run the server loop in a separate thread
    daemon.requestLoop()

# Run the server in a background thread
thread = threading.Thread(target=start_server)
thread.daemon = True
thread.start()



# CLIENT CELL
import Pyro4

# Paste the URI from the server output
uri = "PYRO:obj_26f8411c05f5468d9b05ee4c828fdf63@localhost:50544"  # Replace with your actual URI

proxy = Pyro4.Proxy(uri)

str1 = input("Enter first string: ")
str2 = input("Enter second string: ")

result = proxy.concatenate(str1, str2)
print("Concatenated result:", result)



# IMAGE STYLE TRANSFER

import tensorflow as tf
import numpy as np
import random
import tensorflow_hub as hub
from PIL import Image

def stylize(content_path, style_path, max_dim=512):
    def proc(p):
        img = Image.open(p).convert('RGB')
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        t = tf.image.convert_image_dtype(np.array(img), tf.float32)[None]
        return t

    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    out = model(proc(content_path), proc(style_path))[0][0]
    out = (out*255).numpy().astype('uint8')
    Image.fromarray(out).show()

# Usage:
stylize('/Users/saieshagre/Downloads/content.jpg', '/Users/saieshagre/Downloads/paint.jpeg')





# CLONAL SELECTION
import numpy as np

def objective(x): return x**2 - 4*x + 4
def init_pop(n, low, high): return np.random.uniform(low, high, n)
def evaluate(pop): return np.array([objective(x) for x in pop])
def select(pop, fitness, n): return pop[np.argsort(fitness)[:n]]
def mutate(sel, rate, low, high): return np.clip(sel + np.random.uniform(-rate, rate, sel.shape), low, high)

def clonal_selection(pop_size, low, high, gens, rate, n_sel):
    pop = init_pop(pop_size, low, high)
    for g in range(gens):
        fit = evaluate(pop)
        best = select(pop, fit, n_sel)
        new = mutate(best, rate, low, high)
        pop[np.argsort(fit)[:n_sel]] = new
        print(f"Gen {g+1}: Best = {pop[np.argmin(fit)]:.4f}, Fitness = {min(fit):.4f}")
    return pop[np.argmin(evaluate(pop))]

best = clonal_selection(10, -10, 10, 50, 0.5, 5)
print(f"Best Solution: {best:.4f}")




#GENETIC ALGORITHM


import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random

# Simulated spray drying dataset: [InletTemp, FeedRate, AtomSpeed] -> MoistureContent
np.random.seed(42)
X = np.random.uniform(low=150, high=200, size=(100, 3))  # features
y = 0.3*X[:,0] - 0.2*X[:,1] + 0.1*X[:,2] + np.random.normal(0, 2, 100)  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Neural Network Model (MLP)
def create_nn(weights):
    model = MLPRegressor(hidden_layer_sizes=(5,), max_iter=1, warm_start=True)
    model.fit(X_train, y_train)
    weights = np.array(weights)  # <-- convert to NumPy array
    i = 0
    for layer_weights in model.coefs_:
        shape = layer_weights.shape
        model.coefs_[i] = weights[:np.prod(shape)].reshape(shape)
        weights = weights[np.prod(shape):]
        i += 1
    return model

# Genetic Algorithm
def fitness_function(weights):
    model = create_nn(weights)
    preds = model.predict(X_train)
    return mean_squared_error(y_train, preds)

def crossover(p1, p2):
    point = random.randint(0, len(p1)-1)
    return p1[:point] + p2[point:]

def mutate(ind, rate=0.1):
    return [w + np.random.randn()*rate if random.random() < 0.1 else w for w in ind]

# Initialize population
n_weights = (3*5) + (5*1)  # assuming 3 input, 5 hidden, 1 output
pop = [np.random.uniform(-1, 1, n_weights).tolist() for _ in range(10)]

# Run GA
for generation in range(10):
    pop = sorted(pop, key=fitness_function)
    new_pop = pop[:2]  # elitism
    while len(new_pop) < len(pop):
        p1, p2 = random.sample(pop[:5], 2)
        child = mutate(crossover(p1, p2))
        new_pop.append(child)
    pop = new_pop
    print(f"Gen {generation+1}, MSE: {fitness_function(pop[0]):.4f}")

# Final model
best_weights = pop[0]
final_model = create_nn(best_weights)
test_preds = final_model.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, test_preds))





# DEAP

import random
from deap import base, creator, tools, algorithms

# Define the problem as a maximization task
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define an individual creation function
def create_individual():
    return [random.randint(0, 1) for _ in range(10)]  # A binary individual of size 10

# Define a fitness function (Maximization)
def evaluate(individual):
    return sum(individual),  # The fitness is the sum of the bits in the individual

# Set up the DEAP framework
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Create an initial population of 100 individuals
population = toolbox.population(n=30)

# Run the algorithm
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10,verbose=True)

best_individual = tools.selBest(population, 1)[0]
print("Best Individual:", best_individual)

