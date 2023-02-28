import sys
import itertools as it
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser

stein_file = "data/B/b02.stp"
#stein_file = "data/B/b04.stp"
#stein_file = "data/test.std"

# draw a graph in a window
def print_graph(graph, terms=None, sol=None):

    pos = nx.kamada_kawai_layout(graph)

    nx.draw(graph, pos)
    if (not (terms is None)):
        nx.draw_networkx_nodes(graph, pos, nodelist=terms, node_color='r')
    if (not (sol is None)):
        nx.draw_networkx_edges(graph, pos, edgelist=sol, edge_color='r')
    plt.show()
    return


# verify if a solution is correct and evaluate it
def eval_sol(graph, terms, sol):

    graph_sol = nx.Graph()
    for (i, j) in sol:
        graph_sol.add_edge(i, j, weight=graph[i][j]['weight'])

    # is sol a tree
    if (not (nx.is_tree(graph_sol))):
        print("Error: the proposed solution is not a tree")
        return -1

    # are the terminals covered
    for i in terms:
        if not i in graph_sol:
            print("Error: a terminal is missing from the solution")
            return -1

    # cost of solution
    cost = graph_sol.size(weight='weight')

    return cost

def transformDictToGraph(d, graph):
    graph_sol = nx.Graph()
    for i in d:
        if d[i]:
            graph_sol.add_node(i)
            graph_sol.add_edges_from(graph.edges(i))
        
    sol = []

    for [x, y] in nx.minimum_spanning_tree(graph_sol).edges:
        sol.append([x, y])

    return sol

# compute a approximate solution to the steiner problem
def approx_steiner(graph, terms):
    complet = nx.complete_graph(terms) #metric closure
    pairs = dict(nx.all_pairs_dijkstra(graph)) #shortest path
    for x in terms:
        for y in terms:
            if x is not y:
                complet[x][y]["weight"] = pairs[x][0][y]
                
    arbre_min = nx.minimum_spanning_tree(complet)   #kruskal
    
    res = []
    for (x, y) in arbre_min.edges:
        path = pairs[x][1][y]
        for i in range(len(path) - 1):
            res.append([path[i], path[i+1]]) #deploy shortest path

    return res #res is a sub graph, an edges list "[ [node1, node2], [node2, node3]]"

def metaheuristic_evaluation(sol, terms, graph):
    #penalties: terms not in sol, more than one connected component
    score = 0
    pred = {}
        
    graph_sol = transformDictToGraph(sol, graph)
    #graph_sol_debug = nx.Graph()
    for [x, y] in graph_sol:
        score += graph.edges[x, y]["weight"]    #sum edges score
        
        if x in pred:
            pred[x].append(y)
        else:
             pred[x] = [y]
            
        if y in pred:
            pred[y].append(x)
        else:
            pred[y] = [x]
            
        #graph_sol_debug.add_edge(x, y)

        
    #connected component
    counted = []
    connexPart = 0
    keys = list(pred.keys())
    while len(counted) < len(pred):
        c = random.choice(keys)
        while c in counted:
            c = random.choice(keys)
        queue = [c]
        while len(queue) > 0:
            c = queue[0]
            queue.remove(c)
            for x in pred[c]:
                if x not in counted:
                    queue.append(x)
            counted.append(c)
        connexPart += 1
        
    sizeGraph = len(graph.edges)
    
    score += (connexPart - 1) * sizeGraph  #only one tolerated
    
    #terms not in sol:
    for x in terms:
        if x not in pred:
            score += sizeGraph
            
    #print_graph(graph_sol_debug)
    #print(score)
    return score

def simulated_annealing(graph, terms, initProba, endProba, learnRate):
    best = {}
    for x in graph.nodes:
        best[x] = random.random() > 0.5 

    keys = list(best.keys())
    bestScore = metaheuristic_evaluation(best, terms, graph)

    #1/ln 0.5*-(currentScore - bestScore) = T
    T = 1.0/math.log(initProba)*-(len(graph.edges))
    Tend = 1.0/math.log(endProba)*-(1)
    
    #print("T: " + str(T))
    #print("Tend: " + str(Tend))
    
    bestOverGenerations = []
    while T > Tend:
        current = best.copy()
        
        #neighbor: random edge shifting
        x = random.choice(keys)
        current[x] = not current[x]
        currentScore = metaheuristic_evaluation(current, terms, graph)
        
        if currentScore < bestScore:
            proba = 1
        else:
            proba = math.exp(-(currentScore - bestScore) / T)
            #print("proba: " + str(proba))
        if random.random() < proba:
            best = current.copy()
            bestScore = currentScore
        T = learnRate * T
        
        bestOverGenerations.append(bestScore)
   
    sol = transformDictToGraph(best, graph)
    #print(sol)
    return (sol, bestOverGenerations)

def genetic_gen_child(indiv1, indiv2, mutationProb):
    nbgens = len(indiv1)
    crossoverPoint = random.randint(0, nbgens-1)
    gens = list(indiv1.keys())
    
    child = {}
    for x in range(crossoverPoint):
        child[gens[x]] = indiv1[gens[x]]
    for x in range(crossoverPoint, nbgens):
        child[gens[x]] = indiv2[gens[x]]

    #while allow multiple mutations
    while random.random() < mutationProb:
        mutationPoint = random.randint(0, nbgens - 1)
        child[gens[mutationPoint]] = not child[gens[mutationPoint]]

    return child

def genetic(graph, terms, nbGen, nbIndiv, nbChildren, mutationProb):
    indivs = []
    
    #init population with random genes
    for i in range(nbIndiv):
        indiv = {}
        for x in graph.nodes:
            indiv[x] = random.random() > 0.5
        indivs.append(indiv)
        
    bestOverGenerations = []
    for gen in range(nbGen):
        #children generation and addition to the pop
        for i in range(nbChildren):
            parent1 = indivs[random.randint(0, nbIndiv-1)]
            parent2 = indivs[random.randint(0, nbIndiv-1)]
            indivs.append(genetic_gen_child(parent1, parent2, mutationProb))
            
        #function sorting because we cannot add parameters in list.sort
        def sorting(x):
            return metaheuristic_evaluation(x, terms, graph)
        
        indivs.sort(key=sorting) 
        del indivs[nbIndiv:]        
        
        bestOverGenerations.append(metaheuristic_evaluation(indivs[0], terms, graph))
        
    return (transformDictToGraph(indivs[0], graph), bestOverGenerations)

# class used to read a steinlib instance
class MySteinlibInstance(SteinlibInstance):

    my_graph = nx.Graph()
    terms = []

    def terminals__t(self, line, converted_token):
        self.terms.append(converted_token[0])

    def graph__e(self, line, converted_token):
        e_start = converted_token[0]
        e_end = converted_token[1]
        weight = converted_token[2]
        self.my_graph.add_edge(e_start, e_end, weight=weight)
        
#compute average and fluctuation
def analyze_graph_results(values, nbruns):
    moy = []
    for i in range(len(values[0])):
        moy.append(float(sum([values[j][i] for j in range(nbruns)])) / nbruns )
        
    var = []
    for i in range(len(values[0])):
        var.append(float(sum([pow(values[j][i] - moy[i], 2) for j in range(nbruns)])) / nbruns)
    
    ecarts = []
    for i in range(len(values[0])):
        ecarts.append(math.sqrt(var[i]))
        
    fluctu = []
    moyMinus = []
    moyPlus = []
    for i in range(len(values[0])):
        fluctu.append(2.0*ecarts[i] / math.sqrt(nbruns))
        moyMinus.append(moy[i] - fluctu[i])
        moyPlus.append(moy[i] + fluctu[i])
        
    return (moy, moyMinus, moyPlus)
        
def graph_simulated_annealing_update(nbruns, values, graph, terms):
    for learnRate in values:
        results = []
        for i in range(nbruns):
            (sol, stepByStep) = simulated_annealing(graph, terms, 0.6, 0.01, learnRate)
            results.append(stepByStep)
        
        (moy, moyMinus, moyPlus) = analyze_graph_results(results, nbruns)
        
        moy = moy[3000:]
        moyMinus = moyMinus[3000:]
        moyPlus = moyPlus[3000:]

        plt.plot(moy)
        plt.fill_between(range(len(moy)), moyMinus, moyPlus, alpha=0.2)
        plt.title("B04 avec learnRate = " + str(learnRate) + " , tronqué")
        plt.show()
        
def graph_simulated_annealing_temp(nbruns, values, graph, terms):
    for (init, end) in values:
        results = []
        for i in range(nbruns):
            (sol, stepByStep) = simulated_annealing(graph, terms, init, end, 0.999)
            results.append(stepByStep)
        
        (moy, moyMinus, moyPlus) = analyze_graph_results(results, nbruns)

        plt.plot(moy, label= "(" + str(init) + ", " + str(end) + "): " + str(moy[len(moy) - 1]))
        plt.fill_between(range(len(moy)), moyMinus, moyPlus, alpha=0.2)
        plt.title("B02")
        plt.legend(labelcolor='linecolor')
    plt.show()
    
def graph_genetic_population(nbruns, values, graph, terms):
    for (pop, children) in values:
        results = []
        for i in range(nbruns):
            (sol, stepByStep) = genetic(graph, terms, 50, pop, children, 0.2)
            results.append(stepByStep)
        
        (moy, moyMinus, moyPlus) = analyze_graph_results(results, nbruns) 

        plt.plot(moy, label= "(" + str(pop) + ", " + str(children) + "): " + str(moy[len(moy) - 1]))
        plt.fill_between(range(len(moy)), moyMinus, moyPlus, alpha=0.2)
        plt.title("B04")
        plt.legend(labelcolor='linecolor')
    plt.show()
    
def graph_genetic_mutation(nbruns, values, graph, terms):
    for proba in values:
        results = []
        for i in range(nbruns):
            (sol, stepByStep) = genetic(graph, terms, 50, 10, 30, proba)
            results.append(stepByStep)
        
        (moy, moyMinus, moyPlus) = analyze_graph_results(results, nbruns) 

        plt.plot(moy, label= str(proba) + ": " + str(moy[len(moy) - 1]))
        plt.fill_between(range(len(moy)), moyMinus, moyPlus, alpha=0.2)
        plt.title("B02 pop=10")
        plt.legend(labelcolor='linecolor')
    plt.show()
    
def graph_genetic_generation(nbruns, values, graph, terms):
    for gen in values:
        results = []
        for i in range(nbruns):
            (sol, stepByStep) = genetic(graph, terms, gen, 40, 30, 0.5)
            results.append(stepByStep)
        
        (moy, moyMinus, moyPlus) = analyze_graph_results(results, nbruns) 

        plt.plot(moy, label= str(gen) + ": " + str(moy[len(moy) - 1]))
        plt.fill_between(range(len(moy)), moyMinus, moyPlus, alpha=0.2)
        plt.title("B04")
        plt.legend(labelcolor='linecolor')
    plt.show()
    
def graph_perf(nbruns, graph, terms):
    results = []
    for i in range(nbruns):
        (sol, stepByStep) = genetic(graph, terms, 150, 40, 30, 0.5)
        results.append(eval_sol(graph, terms, sol))
    
    plt.plot(results, label= "Génétique: " + str(sum(results) / nbruns))
    
    results = []
    for i in range(nbruns):
        (sol, stepByStep) = simulated_annealing(graph, terms, 0.5, 0.01, 0.999)
        results.append(eval_sol(graph, terms, sol))
    
    plt.plot(results, label= "Recuit simulé: " + str(sum(results) / nbruns))
    
    results = []
    for i in range(nbruns):
        sol = approx_steiner(graph, terms)
        results.append(eval_sol(graph, terms, sol))
    
    plt.plot(results, label= "Approximation: " + str(sum(results) / nbruns))
    
    plt.plot([83 for j in range(nbruns)], label= "Optimal: " + str(83))
    
    plt.legend(labelcolor='linecolor')
    plt.title("B02")
    plt.show()

if __name__ == "__main__":
    my_class = MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        
        #print_graph(graph, terms)
        #sol = approx_steiner(graph, terms)
        #(sol, _) = simulated_annealing(graph, terms, 0.6, 0.01, 0.999)
        #(sol, _) = genetic(graph, terms, 100, 40, 30, 0.2)
        #print_graph(graph, terms, sol)
        #print(eval_sol(graph, terms, sol))
        
        #graph_simulated_annealing_update(10, [0.9, 0.99, 0.999], graph, terms)
        #graph_simulated_annealing_temp(10, [(0.7, 0.2), (0.6, 0.1), (0.5, 0.01), (0.9, 0.000001)], graph, terms)
        
        #graph_genetic_population(10, [(5, 10), (10, 10), (35, 10), (60, 10)], graph, terms)
        #graph_genetic_mutation(10, [0.7, 0.5, 0.25, 0.1, 0.05], graph, terms)
        #graph_genetic_generation(10, [200, 100, 50], graph, terms)
        #graph_perf(50, graph, terms) 