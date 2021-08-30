from math import ceil, floor
import random
from random import randint
import matplotlib.pyplot as plt

B = 15

def simple(Npred, N):
    """
    Algorithme simple 1-consistent
    
    Parameters
    ----------
    Npred : int
        Le nombre de jours de skis prédit.
        
    N : int
            Le nombre de jours de skis réél.

    Returns
    -------
    int
        Coût du séjour en terme de skis.

    """
    if Npred >= B:
        #Achat dès le premier jour
        return B
    else:
        #Location des skis tout le long
        return N
    
def simpleRobCons(Npred, lambd, N):
    """
    Algorithme simple robuste et consistent
    
    Parameters
    ----------
    Npred : int
            Le nombre de jours de skis prédit.
            
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
            Le nombre de jours de skis réél.

    Returns
    -------
    int
        Coût du séjour en terme de skis.

    """
    if Npred >= B:
        #Achat au début du jour ceil(lambd*B)
        if N < ceil(B*lambd):
            return N
        return ceil(B*lambd)-1 + B
    else:
        #Achat au début du jour ceil(B/lambd)
        if N < ceil(B/lambd):
            return N
        return ceil(B/lambd)-1 + B
    
def randomizedRobCons(Npred, lambd, N):
    """
    Algorithme randomisé robuste et consistent
    
    Parameters
    ----------
    Npred : int
            Le nombre de jours de skis prédit.
            
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
            Le nombre de jours de skis réél.

    Returns
    -------
    int
        Coût du séjour en terme de skis.

    """
    if Npred >= B:
        k = floor(B*lambd)
        listQi = []
        for i in range(1, k+1):
            listQi.append(((B-1)/B)**(k-i)/(B*(1-(1-1/B)**k)))
        # Choisir un j entre 1 et k aléatoirement depuis la distribution définie par listQi
        j = random.choices([i for i in range(1,k+1)], weights = listQi)[0]
        #print("ListeQ = ",listQi, "J=", j)

    else:
        l = ceil(B/lambd) 
        listRi = []
        for i in range(1, l+1):
            listRi.append(((B-1)/B)**(l-i)/(B*(1-(1-1/B)**l)))
        # Choisir un j entre 1 et l aléatoirement depuis la distribution définie par listRi
        j = random.choices([i for i in range(1,l+1)], weights = listRi)[0]
        #print("ListeR = ",listRi, "J=", j)
    
    #Achat au début du jour j
    if N<j: #On n'atteint pas le jour j d'achat des skis
        return N
        
    return B + (j-1) #On loue jusqu'au jour j-1 puis on achète 
   
def primalDualSansPred(N):
    """
    Algorithme primal-dual sans prédictions
    
    Parameters
    ----------
    N : int
            Le nombre de jours de skis réél.

    Returns
    -------
    int
        Coût du séjour en terme de skis.

    """
    def e(x):
        """
        Fonction locale utile pour les calculs
        
        Parameters
        ----------
        x : float
    
        Returns
        -------
        float
    
        """
        return (1+1/B)**(x*B)
        
    x = 0
    listZj = [0 for i in range(N)]
    c = e(1)
    #cprime = 1
    
    for j in range(N): #Pour chaque nouveau jour
        if x + listZj[j] < 1:
            # Mise à jour du primal
            listZj[j] = 1-x
            x = (1+1/B)*x + (1/((c-1)*B))
            #Mise à jour du dual : yj = cprime
            
    return sum(listZj) + B*x
            
def primalDualAvecPred(Npred, lambd, N):
    """
    Algorithme primal-dual avec prédictions
    
    Parameters
    ----------
    Npred : int
            Le nombre de jours de skis prédit.
            
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
            Le nombre de jours de skis réél.

    Returns
    -------
    int
        Coût du séjour en terme de skis.

    """
    def e(x):
        """
        Fonction locale utile pour les calculs
        
        Parameters
        ----------
        x : float
    
        Returns
        -------
        float
    
        """
        return (1+1/B)**(x*B)
        
    x = 0
    listZj = [0 for i in range(N)]
    
    if Npred >= B:
        #Les prédictions suggèrent d'acheter
        c = e(lambd)
        #cprime = 1
        
    else:
        #Les prédictions suggèrent de louer
        c = e(1/lambd)
        #cprime = lambd
    
    for j in range(N): #Pour chaque nouveau jour
        if x + listZj[j] < 1:
            # Mise à jour du primal
            listZj[j] = 1-x
            x = (1+1/B)*x + (1/((c-1)*B))
            #Mise à jour du dual : yj = cprime
            
    return sum(listZj) + B*x

def erreur_prediction(Npred, N):
    """ 
    Calcule l'erreur de prédiction
    
    Parameters
    ----------
    Npred : int
           La prédiction de la durée du séjour 
           
    N : int
           La vraie durée du séjour 
           
    Returns
    -------
    float 
        L'erreur de prédiction
    """
    return abs(Npred - N)

def show_multiple_sims(XYlegendes):
    """ 
    Affiche le graphique demandé
    
    Parameters
    ----------
    XYlegendes : list[(list[float], list[float], str)]
           Une liste composée de tuple dont le premier élément est une liste des X (échantilloné à intervalle régulier), 
           le second une liste des Y (ratio de compétitivité) et le dernier le nom pour la légende     
    """
    for _ in XYlegendes:
        X, Y, legende = _
        plt.plot([X[s] for s in range(len(X[:-1])) if Y[s] != 0], [y for y in Y if y != 0] , label=legende)
        #plt.plot(X, Y, label=legende)

        plt.xlabel('Erreur de prédiction')
        plt.ylabel('Rapport de compétitivité')
        #plt.legend(loc='best')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
def simulations():
    """
    Permet de faire une simulations des algorithmes précédents
    
    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # Tirage aléatoire de N et de Npred

    nbExemples = 15000 #Nombre d'exemples du problème
    Nmax = 30 #Nombre de jours de vacances maximum
    
    Ns = [randint(1, Nmax) for i in range(nbExemples)]
    Npreds = [Ns[i] + randint(0, Nmax-Ns[i]) - randint(0, Ns[i]-1)  for i in range(nbExemples)]
    #print("Ns = ", Ns)
    #print("Npreds = ", Npreds)
    
    Ns_Npred = [(Ns[i], Npreds[i]) for i in range(nbExemples)]
    Ns_Npred.sort(key=lambda vpa: erreur_prediction(vpa[1], vpa[0]))
    
    lambd = 0.5 # A faire varier
    # Définition des points du graphique
    X = [erreur_prediction(p, v) for (v, p) in Ns_Npred]
    #Ratio de compétitibité déterminé par algo / simple(N,N) puisque simple(N,N) est l'optimum
    Ysimple = [simple(Npred, N) / simple(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons = [simpleRobCons(Npred, lambd, N) / simple(N, N) for (N, Npred) in Ns_Npred]
    YrandomizedRobCons = [randomizedRobCons(Npred, lambd, N) / simple(N, N) for (N, Npred) in Ns_Npred]
    YprimalDualSansPred = [primalDualSansPred(N) / simple(N, N) for (N, Npred) in Ns_Npred]
    YprimalDualAvecPred = [primalDualAvecPred(Npred, lambd, N) / simple(N, N) for (N, Npred) in Ns_Npred]
    
    # À partir d'ici, on crée des classes de valeurs comme pour un histogramme, dont on tirera le maximum, 
    # puisque le rapport de compétitivité est calculé en fonction du pire cas des algorithmes
    
    def locale(X, Y, legend):
        """
        Permet de créer des classes de valeurs comme pour un histogramme, dont on tirera le maximum
        
        Parameters
        ----------
        X : list[float]
            Liste des erreurs de prédictions.
                
        Y : list[float]
            Liste des ratios de compétitivité.
                
        legend : str
            Nom de la légende.
    
        Returns
        -------
        (list[float], list[float], str)
            Un tuple contenant la liste des X et des Y du graphique et la légende associée.

    """
    # precision : int, nombre de classes voulues. Si trop élevé, peut avoir un impact néfaste (classes vides).
        precision = 100
        xmin = min(X)
        xmax = max(X)
        step = (xmax - xmin )/precision
        stages = [xmin]
        while max(stages) < xmax:
            stages.append(stages[-1] + step)
        classes = [[Y[j] for j in range(len(X)) if X[j] >= stages[i] and X[j] < stages[i+1]] for i in range(len(stages) - 1)]
        
        # On met un 0 pour les classes vides. Les fonctions d'affichage devront en tenir compte.
        newY = [max(c) if len(c) > 0 else 0 for c in classes]
        return (stages, newY, legend)
    
    resA = []
    # Algorithme simple
    resA.append(locale(X, Ysimple, "Simple"))
    # Algorithme simple robuste et consistant
    resA.append(locale(X, YsimpleRobCons, "Simple Robuste Consistant lamb = "+str(lambd)))
    # Algorithme randomisé robuste et consistant
    resA.append(locale(X, YrandomizedRobCons, "Randomisé lamb = "+str(lambd)))
    # Algorithme primal dual sans prédictions
    resA.append(locale(X, YprimalDualSansPred, "Primal-Dual sans Pred "))
    # Algorithme primal dual avec prédictions
    resA.append(locale(X, YprimalDualAvecPred, "Primal-Dual avec Pred lamb = "+str(lambd)))
    
    # affichage du graphique
    show_multiple_sims(resA)
    


simulations()