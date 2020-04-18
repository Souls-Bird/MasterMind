import random
import numpy as np

N = 4   #Nombre de pions
K = 12   #Nombre de couleurs
L = 3   #coefficient de poid entre couleur bien placée et mal placée
taille_pop = 200 #taille initiale d'un échantillon de population
m = 100 #nombre de candidats qui vont recevoir des mutations/croisements
p1 = 0.3 #taux de mutation 1
p2 = 0.3 #taux de mutation 2
p3 = 0.7 #taux de croisement SP
p4 = 0.7 #taux de croisement uniforme
MAXGEN = 2000 #Nombre maximum de générations

def score(myPM, l = L):
    return (l*myPM[0] + myPM[1])

def compare(sol, cand):
    p = 0
    m = 0
    for i in range(N):
        if sol[i] == cand[i]:
            p += 1
        elif np.any(sol[:] == cand[i]):
            m += 1
    return (p, m)

#Eval évalue la cohérence d'un candidat par rapport à un autre candidat duquel on connait le score.
#On considère que c est la solution, on le compare à cj le coup déjà joué et on fait la différence entre leurs scores respectifs.
#Si eval() est nul, il est possible que c soit la solution.
def eval(c, cj, scj):
    return abs(scj - score(compare(c, cj)))

#prend un numpy array et un historique
def fitness(c, historique):
    res = 0
    for value in historique.values():
        res += eval(c, value[0], score(value[1]))
    return res

#Select a candidate weighted-randomly
def selection(weighted_gen):
    max = sum(weighted_gen.values())
    pick = random.uniform(0, max)
    current = 0
    for key, value in weighted_gen.items():
        current += value
        if current > pick:
            return key

def mutation1(candidat):
    candidat[random.randint(0, N-1)] = random.randint(0, K-1)
    return candidat

def mutation2(candidat):
    myList = []
    for i in range(N):
        myList.append(i)
    rand = random.sample(myList, 2)
    candidat[rand[0]] = candidat[rand[1]]
    return candidat

#prend deux numpy arrays
#return un tuple de deux numpy arrays
def SP_crossover(a, b):
    crossover_point = random.randint(1,N-1)
    c = np.concatenate([a[0:crossover_point],b[crossover_point:N]])
    d = np.concatenate([b[0:crossover_point],a[crossover_point:N]])
    return (c, d)

#prend deux numpy arrays
#return un numpy array
def U_crossover(a, b):
    c = np.zeros(N, dtype = int)
    for i in range(N):
        if random.randint(0,1):
            c[i] = a[i]
        else:
            c[i] = b[i]
    return c

def gen_rand_candidate():
    rand = np.zeros(N, dtype = int)
    for i in range(N):
        rand[i] = random.randint(0,K)
    return rand

def print_historique(historique):
    for key, value in historique.items():
        print("Coup", str(key) + ":", value[0], "-->", value[1])


if __name__ == '__main__':
    points = 0
    P = 1   #nombre de propositions données

    #choix de la solution
    x = input("Voulez-vous choisir la solution ? (y/N)")
    if x == 'N':
        sol = gen_rand_candidate()
        print("La solution choisie est", sol)
    elif x == 'y':
        choix = input("Veillez saisir la solution en écrivant "+str(N)+" chiffres entre 0 et "+str(K-1)+". (pas d'espace entre les chiffres) :\n")
        y = []
        for char in choix:
            y.append(int(char))
        sol = np.array(y)
        print("La solution choisie est", y)

    else:
        print("Je n'ai pas compris votre choix. Sortie.")

    #random guess 1 :
    guess = tuple(gen_rand_candidate())
    print("1er guess", guess)

    response = compare(sol, guess)
    print(response)
    historique = {P: (guess, response)}

    while points != L*N:
        #initialisation de la population aléatoirement:
        population = []
        for i in range(taille_pop):
            population.append(gen_rand_candidate())

        best_fitness = 10000
        guess = ()
        generation = 1

        weighted_gen = {}

        #Premier calcul du fitness
        for candidat in population:
            current_fitness = fitness(np.array(candidat), historique)
            weighted_gen[tuple(candidat.tolist())] = current_fitness
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                guess = tuple(candidat)
        # print("La population de base a un best fitness de", best_fitness, "pour", guess)

        while best_fitness != 0 and generation < MAXGEN: #Tant qu'on a pas trouvé un candidat cohérent avec l'historique des coups joués

            #Sélection des meilleurs individus
            best_candidates = []
            for i in range(m):
                best_candidates.append(selection(weighted_gen))

            #Creation de nouveaux individus
            nb = 0

            #MUTATIONS
            for a in best_candidates:
                if random.uniform(0,1) < p1:
                    best_candidates.append(tuple(mutation1(list(a))))
                    nb += 1

                if random.uniform(0, 1) < p2:
                    best_candidates.append(tuple(mutation2(list(a))))
                    nb += 1

            #CROISEMENTS
            for i in range(10): #On donne 10 chances que chaque croisement ait lieu
                if random.uniform(0, 1) < p3:
                    childrens = SP_crossover(np.array(best_candidates[random.randint(0, len(best_candidates)-1)]), np.array(best_candidates[random.randint(0, len(best_candidates)-1)]))
                    best_candidates.append(tuple(childrens[0].tolist()))
                    best_candidates.append(tuple(childrens[1].tolist()))
                    nb += 2

                if random.uniform(0, 1) < p4:
                    best_candidates.append(tuple(U_crossover(np.array(best_candidates[random.randint(0, len(best_candidates)-1)]), np.array(best_candidates[random.randint(0, len(best_candidates)-1)])).tolist()))
                    nb += 1

            # print("Nombre d'ajouts cette génération :", nb)
            # print("Taille de la population :", len(best_candidates))

            #Recalcul des fitness
            weighted_gen = {}
            for candidat in best_candidates:
                current_fitness = fitness(np.array(candidat), historique)
                weighted_gen[candidat] = current_fitness
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    guess = candidat
                    # print("candidat", candidat, "fait baisser le meilleur fitness à", best_fitness)

            # population = best_candidates
            print("Generation", generation, "| population :", len(best_candidates), "| meilleur fitness :", best_fitness, "|", end='\r', flush=True)
            generation += 1

        P += 1
        points =  score(compare(sol, guess))
        historique[P] = (guess, compare(sol, guess))
        print("\n\nGuess", P, ":", guess, "-->", points, "points")
        print_historique(historique)
