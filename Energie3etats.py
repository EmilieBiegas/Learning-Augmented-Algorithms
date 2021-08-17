
from random import randint
import matplotlib.pyplot as plt

A1 = 5
A2 = 3
A3 = 0
C1 = 10
C2 = 25

def aveugle(Npred, N):
    """
    Algorithme aveugle avec prédictions
    
    Parameters
    ----------
    Npred : int
        Le nombre d'instant prédit entre les deux tâches.
        
    N : int
        Le nombre d'instant réél entre les deux tâches.
     
    Returns
    -------
    int
        Coût en terme d'énergie dépensée entre les deux tâches.

    """
    if Npred < C1/(A1-A2):
        # Rester dans l'état ON jusqu'à l'arrivée d'une nouvelle tâche pour un coût de A1.N
        return A1*N
    if Npred < (C2-C1)/(A2-A3):
        # Passer dans l'état VEI dès le premier instant pour un coût de A_2.N + C_1
        return A2*N + C1
    # Sinon, c'est-à-dire si Npred >= (C2-C1)/(A2-A3), passer dans l'état OFF dès le premier instant pour 
    # un coût de A_3.N + C_2
    return A3*N +C2

def simpleRobCons(Npred, lambd, N):
    """
    Algorithme aveugle avec prédictions
    
    Parameters
    ----------
    Npred : int
        Le nombre d'instant prédit entre les deux tâches.
     
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
        Le nombre d'instant réél entre les deux tâches.
     
    Returns
    -------
    int
        Coût en terme d'énergie dépensée entre les deux tâches.

    """
    if Npred < C1/(A1-A2):
        # Passer dans l'état VEI au temps C1/(lambda(A1-A2)) puis dans l'état OFF au temps 
        # (C2-C1)/(lambda(A2-A3))
        if N < C1/(lambd*(A1-A2)):
            # On n'a pas le temps de passer à l'état VEI
            return A1*N
        if N < (C2-C1)/(lambd*(A2-A3)):
            # On a le temps de passer à l'état VEI mais pas à l'état OFF
            return A1 * (C1/(lambd*(A1-A2))-1) + A2 * (N - C1/(lambd*(A1-A2)) + 1) + C1
        # On a le temps pour passer à l'état VEI puis à l'état OFF
        return A1*(C1/(lambd*(A1-A2))-1) + A2*((C2-C1)/(lambd*(A2-A3))-C1/(lambd*(A1-A2))) + A3*(N - (C2-C1)/(lambd*(A2-A3)) + 1) + C2
    
    if Npred < (C2-C1)/(A2-A3):
        # Passer dans l'état VEI au temps lambda*C1/(A1-A2) puis passer dans l'état OFF au temps 
        # (C2-C1)/(lambda(A2-A3))
        if N < lambd*C1/((A1-A2)):
            # On n'a pas le temps de passer à l'état VEI
            return A1*N
        if N < (C2-C1)/(lambd*(A2-A3)):
            # On a le temps de passer à l'état VEI mais pas à l'état OFF
            return A1 * (lambd*C1/(A1-A2-1)) + A2 * (N - lambd*C1/((A1-A2)) + 1) + C1
        # On a le temps pour passer à l'état VEI puis à l'état OFF
        return A1*(lambd*C1/(A1-A2-1)) + A2*((C2-C1)/(lambd*(A2-A3))-lambd*C1/(A1-A2)) + A3*(N - (C2-C1)/(lambd*(A2-A3)) + 1) + C2

    # Sinon, c'est-à-dire si Npred >= (C2-C1)/(A2-A3), passer dans l'état VEI au temps 
    # lambda*C1/(A1-A2) puis dans l'état OFF au temps lambda(C2-C1)/A2-A3)
    if N < lambd*C1/(A1-A2):
        # On n'a pas le temps de passer à l'état VEI
        return A1*N
    if N < lambd*(C2-C1)/(A2-A3):
        # On a le temps de passer à l'état VEI mais pas à l'état OFF
        return A1 * (lambd*C1/((A1-A2))-1) + A2 * (N - lambd*C1/((A1-A2)) + 1) + C1
    # On a le temps pour passer à l'état VEI puis à l'état OFF
    return A1*(lambd*C1/(A1-A2-1) + A2*(lambd*(C2-C1)/(A2-A3))-lambd*C1/(A1-A2)) + A3*(N - lambd*(C2-C1)/(A2-A3) + 1) + C2

def erreur_prediction(Npred, N):
    """ 
    Calcule l'erreur de prédiction
    
    Parameters
    ----------
    Npred : int
           La prédiction de la durée entre les tâches
           
    N : int
           La vraie durée séparant les deux tâches 
           
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
    Npred : int
            Le nombre prédit d'instants entre les tâches.
            
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
            Le vrai nombre d'instants entre les tâches.

    Returns
    -------
    int
        C PB

    """
    # Tirage aléatoire de N et de Npred

    nbExemples = 15000 #Nombre d'exemples du problème
    Nmax = 30 #Nombre d'instant entre les tâches maximum
    
    Ns = [randint(1, Nmax) for i in range(nbExemples)]
    Npreds = [Ns[i] + randint(0, Nmax-Ns[i]) - randint(0, Ns[i]-1)  for i in range(nbExemples)]
    #print("Ns = ", Ns)
    #print("Npreds = ", Npreds)
    
    Ns_Npred = [(Ns[i], Npreds[i]) for i in range(nbExemples)]
    Ns_Npred.sort(key=lambda vpa: erreur_prediction(vpa[1], vpa[0]))
    
    lambd = 0.5 #PB a faire varier
    # Définition des points du graphique
    X = [erreur_prediction(p, v) for (v, p) in Ns_Npred]
    #Ratio de compétitibité déterminé par algo / aveugle(N,N) puisque aveugle(N,N) est l'optimum
    Yaveugle = [aveugle(Npred, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons05 = [simpleRobCons(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons0 = [simpleRobCons(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons1 = [simpleRobCons(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    
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
    # Algorithme aveugle
    resA.append(locale(X, Yaveugle, "Aveugle"))
    # Algorithme simple robuste et consistant pour un lambda de 0.5
    resA.append(locale(X, YsimpleRobCons05, "Simple Robuste Consistant lamb = 0.5"))
    # Algorithme simple robuste et consistant pour un lambda de 0
    resA.append(locale(X, YsimpleRobCons0, "Simple Robuste Consistant lamb = 0"))
    # Algorithme simple robuste et consistant pour un lambda de 1
    resA.append(locale(X, YsimpleRobCons1, "Simple Robuste Consistant lamb = 1"))
    
    # affichage du graphique
    show_multiple_sims(resA)
    


simulations()