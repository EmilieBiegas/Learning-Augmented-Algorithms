from random import randint
import matplotlib.pyplot as plt

k = 3
A = [2*(k-i) for i in range(k)] #On a bien A[0]>A[1]>...>A[k-1]
C = [5*i for i in range(k)] #On a bien C[0]=0 puis C[1]<C[2]<...<C[k-1]
#/!\ C[j] équivaut à C(j+1) dans le rapport /!\

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
    if Npred < C[1]/(A[0]-A[1]): #Aussi exprimé dans la boucle for suivante
        # Rester dans l'état 1 jusqu'à l'arrivée d'une nouvelle tâche pour un coût de A_1.N
        return A[0]*N
    
    for i in range(1, k):
        if Npred < (C[i]-C[i-1])/(A[i-1]-A[i]):
            # Passer dans l'état i dès le premier instant pour un coût de A_i.N + C_i
            return A[i-1]*N + C[i-1]
    # Sinon, c'est-à-dire si Npred >= (Ck-C(k-1))/(A(k-1)-Ak), passer dans l'état k dès le 
    # premier instant pour un coût de A_k.N + C_k
    return A[k-1]*N +C[k-1]

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
    if Npred < C[1]/(A[0]-A[1]):
        # Passer dans l'état 2 au temps C1/(lambda(A1-A2)) puis dans l'état i au temps 
        # (Ci-C(i-1))/(lambda(A(i-1)-Ai))
        if lambd==0 or N < C[1]/(lambd*(A[0]-A[1])): #Aussi exprimé dans la boucle for suivante
            # On n'a pas le temps de passer à l'état 2
            return A[0]*N
        
        for i in range(2, k+1):
            if N < (C[i-1]-C[i-2])/(lambd*(A[i-2]-A[i-1])):
                # On n'a pas le temps de passer à l'état i
                res = 0
                tmp = 0
                for j in range(1, i-1):
                    res += A[j-1]*((C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1-tmp)
                    tmp = (C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1  
                return res + A[i-2]*(N-tmp) + C[i-2]
        
        # Ici, on a le temps de passer dans tous les états
        res = 0
        tmp = 0
        for j in range(1, k):
            res += A[j-1]*((C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1-tmp)
            tmp = (C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1  
        return res + A[k-1]*(N-tmp) + C[k-1]
            
    for i in range(2, k): #PB si i=1, a-t-on comme au dessus? OK
        if Npred < (C[i]-C[i-1])/(A[i-1]-A[i]):
        # Pour tout l<i, passer dans l'état l au temps lambda*(Cl-C(l-1))/((A(l-1)-Al)), puis pour
        # tout r>=i, passer dans l'état r au temps (Cr-C(r-1))/(lambda(A(r-1)-Ar))
            for l in range(2, i):
                if N < lambd*(C[l-1]-C[l-2])/((A[l-2]-A[l-1])):
                    # On n'a pas le temps de passer à l'état l
                    res = 0
                    tmp = 0
                    for j in range(1, l-1):
                        res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
                        tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1  
                    return res + A[l-2]*(N-tmp) + C[l-2]
              
            # Si on arrive ici, on prend déjà en compte les coûts pour les changement d'états l<i    
            res = 0
            tmp = 0
            for j in range(1, i):
                res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
                tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1 
                
            for r in range(i, k+1): 
                if lambd==0 or N < (C[r-1]-C[r-2])/(lambd*(A[r-2]-A[r-1])):
                    # On n'a pas le temps de passer à l'état r
                    for j in range(i, r-1):
                        res += A[j-1]*((C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1-tmp)
                        tmp = (C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1  
                    return res + A[r-2]*(N-tmp) + C[r-2]
            # Ici, on a le temps de passer dans tous les états   
            for j in range(i, k):
                res += A[j-1]*((C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1-tmp)
                tmp = (C[j]-C[j-1])/(lambd*(A[j-1]-A[j]))-1  
            return res + A[k-1]*(N-tmp) + C[k-1]
            
    # Sinon, c'est-à-dire si Npred >= (Ck-C(k-1))/(A(k-1)-Ak), passer dans l'état i au temps 
    # lambda(Ci-C(i-1))/(A(i-1)-Ai)
    for i in range(2, k+1): 
        if N < lambd*(C[i-1]-C[i-2])/(A[i-2]-A[i-1]):
            # On n'a pas le temps de passer à l'état i
            res = 0
            tmp = 0
            for j in range(1, i-1):
                res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
                tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1  
            return res + A[i-2]*(N-tmp) + C[i-2] 
        
    # Sinon, c'est-à-dire si N >= lambda(Ck-C(k-1))/(A(k-1)-Ak)
    # Ici, on a le temps de passer dans tous les états
    res = 0
    tmp = 0
    for j in range(1, k):
        res += A[j-1]*(lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1-tmp)
        tmp = lambd*(C[j]-C[j-1])/(A[j-1]-A[j])-1     
    return res + A[k-1]*(N-tmp) + C[k-1]
        
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
    None

    """
    # Tirage aléatoire de N et de Npred

    nbExemples = 5000*k #Nombre d'exemples du problème
    Nmax = 10*k #Nombre d'instant entre les tâches maximum
    # En fonction du nombre d'état pour avoir potentiellement le temps de parcourir tous les états
    
    Ns = [randint(1, Nmax) for i in range(nbExemples)]
    Npreds = [Ns[i] + randint(0, Nmax-Ns[i]) - randint(0, Ns[i]-1)  for i in range(nbExemples)]
    #print("Ns = ", Ns)
    #print("Npreds = ", Npreds)
    
    Ns_Npred = [(Ns[i], Npreds[i]) for i in range(nbExemples)]
    Ns_Npred.sort(key=lambda vpa: erreur_prediction(vpa[1], vpa[0]))
    
    # Définition des points du graphique
    X = [erreur_prediction(p, v) for (v, p) in Ns_Npred]
    #Ratio de compétitibité déterminé par algo / aveugle(N,N) puisque aveugle(N,N) est l'optimum
    Yaveugle = [aveugle(Npred, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons05 = [simpleRobCons(Npred, 0.5, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons0 = [simpleRobCons(Npred, 0.1, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons1 = [simpleRobCons(Npred, 1, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    #YprimalDualSansPred = [primalDualSansPred(N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    #YprimalDualAvecPred = [primalDualAvecPred(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    
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
    # Algorithme primal dual sans prédictions
    #resA.append(locale(X, YprimalDualSansPred, "Primal-Dual sans Pred "))
    # Algorithme primal dual avec prédictions
    #resA.append(locale(X, YprimalDualAvecPred, "Primal-Dual avec Pred lamb = "+str(lambd)))
    
    # affichage du graphique
    show_multiple_sims(resA)
    


#simulations()

# -------------------------------------------------------------------------------------------
# ---------------------------------TEST PRIMAL DUAL PB --------------------------------------
# -------------------------------------------------------------------------------------------

def primalDualSansPred(N):
    """
    Algorithme primal-dual sans prédictions
    
    Parameters
    ----------
    N : int
            Le nombre d'instants réél entre les deux tâches.

    Returns
    -------
    int
         Coût en terme d'énergie dépensée entre les deux tâches.

    """
    
    # Definition des variables primales
    listET = [[0 for i in range(N)] for j in range(k)]
    # On n'a pas besoin d'exprimer les xi puisqu'on a accès à l'indice N-1 des sous-tableaux ci-dessus
    
    # Initialisation
    listET[0][0] = 1
    #for j in range(1, k): #Déjà initialisé à 0
        #listET[j][0] = 0
    
    # Posons ça pour voir PB : à trouver les bonnes valeurs en fct des données peut être
    c = A 
    
    for j in range(1, N): #Pour chaque nouvelle unité de temps
        #PB je ne sais pas quoi faire ici ??
        # Peut etre passer vers l'état supérieur d'un petit peu ?
        for i in range(k-1): #Pour chaque état
            if listET[i][j-1]>0:
                listET[i][j] = (1-1/c[i])*listET[i][j-1]
                listET[i+1][j] = listET[i][j-1]/c[i]
                # Ce qui somme bien à 1 et on arrête de s'occuper de cette unité de temps
                break
        somme = sum([listET[i][j] for i in range(k)])
        if somme < 1: #Si on n'a pas atteint une somme à 1 c'est qu'il manque des unités dans le dernier état
            listET[k-1][j] = 1 - somme
    
    # Regardons dans quel état est la machine lors de l'arrivée de la tâche
    for i in range(k):
        if listET[i][N-1]>1/k: # La machine est considéré dans l'état i lors de l'arrivée de la tâche
            res = C[i]   
    
    return res + sum([A[i]*sum([listET[i][j] for j in range(N)]) for i in range(k)])
#PB ce ne sont pas des entiers donc pas un seul état de la machine à un instant donné
# Pas grand chose à voir avec le primal-dual
            
def primalDualAvecPred(Npred, lambd, N):
    """
    Algorithme primal-dual avec prédictions
    
    Parameters
    ----------
    Npred : int
            Le nombre d'instants prédit entre les deux tâches.
            
    lambd : float
            L'hyperparamètre de robustesse.
            
    N : int
            Le nombre d'instants réél entre les deux tâches.

    Returns
    -------
    int
         Coût en terme d'énergie dépensée entre les deux tâches.

    """  
    
    # Definition des variables primales
    listET = [[0 for i in range(N)] for j in range(k)]
    # On n'a pas besoin d'exprimer les xi puisqu'on a accès à l'indice N-1 des sous-tableaux ci-dessus
    
    # Initialisation
    listET[0][0] = 1
    #for j in range(1, k): #Déjà initialisé à 0
        #listET[j][0] = 0
    
    # Posons ça pour voir PB : à trouver les bonnes valeurs en fct des données peut être
    cEleve = 100
    cFaible = 2
    
    for i in range(1, k):        
        if Npred < (C[i]-C[i-1])/(A[i-1]-A[i]):
            # Pour tout l<i, passer dans l'état l rapidement, puis pour
            # tout r>=i, passer dans l'état r moins vite
            c = [cEleve for l in range(i-1)] + [cFaible for r in range(i-1,k)]
            break
        
    if Npred >= (C[k-1]-C[k-2])/(A[k-2]-A[k-1]): # Si on n'est entré dans aucun if et que c n'est pas initialisé
            # Pour tout l<k, passer dans l'état l rapidement
            c = [cEleve for l in range(k)]
    
    for j in range(1, N): #Pour chaque nouvelle unité de temps
        #PB je ne sais pas quoi faire ici ??
        # Peut etre passer vers l'état supérieur d'un petit peu ?
        for i in range(k-1): #Pour chaque état
            if listET[i][j-1]>0:
                listET[i][j] = (1-1/c[i])*listET[i][j-1]
                listET[i+1][j] = listET[i][j-1]/c[i]
                # Ce qui somme bien à 1 et on arrête de s'occuper de cette unité de temps
                break
        somme = sum([listET[i][j] for i in range(k)])
        if somme < 1: #Si on n'a pas atteint une somme à 1 c'est qu'il manque des unités dans le dernier état
            listET[k-1][j] = 1 - somme
    
    # Regardons dans quel état est la machine lors de l'arrivée de la tâche
    for i in range(k):
        if listET[i][N-1]>1/k: # La machine est considéré dans l'état i lors de l'arrivée de la tâche
            res = C[i]   
    
    return res + sum([A[i]*sum([listET[i][j] for j in range(N)]) for i in range(k)])
#PB ce ne sont pas des entiers donc pas un seul état de la machine à un instant donné
# Pas grand chose à voir avec le primal-dual



# ------------------------------- PB A ENLEVER CE QUI SUIT : -----------------------------
def simulations2():
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
    None

    """
    # Tirage aléatoire de N et de Npred

    nbExemples = 5000*k #Nombre d'exemples du problème
    Nmax = 10*k #Nombre d'instant entre les tâches maximum
    # En fonction du nombre d'état pour avoir potentiellement le temps de parcourir tous les états
    
    Ns = [randint(1, Nmax) for i in range(nbExemples)]
    Npreds = [Ns[i] + randint(0, Nmax-Ns[i]) - randint(0, Ns[i]-1)  for i in range(nbExemples)]
    #print("Ns = ", Ns)
    #print("Npreds = ", Npreds)
    
    Ns_Npred = [(Ns[i], Npreds[i]) for i in range(nbExemples)]
    Ns_Npred.sort(key=lambda vpa: erreur_prediction(vpa[1], vpa[0]))
    
    lambd = 0.5
    
    # Définition des points du graphique
    X = [erreur_prediction(p, v) for (v, p) in Ns_Npred]
    #Ratio de compétitibité déterminé par algo / aveugle(N,N) puisque aveugle(N,N) est l'optimum
    Yaveugle = [aveugle(Npred, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons05 = [simpleRobCons(Npred, 0.5, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons0 = [simpleRobCons(Npred, 0.1, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YsimpleRobCons1 = [simpleRobCons(Npred, 1, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YprimalDualSansPred = [primalDualSansPred(N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    YprimalDualAvecPred = [primalDualAvecPred(Npred, lambd, N) / aveugle(N, N) for (N, Npred) in Ns_Npred]
    
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
    # Algorithme primal dual sans prédictions
    resA.append(locale(X, YprimalDualSansPred, "Primal-Dual sans Pred "))
    # Algorithme primal dual avec prédictions
    resA.append(locale(X, YprimalDualAvecPred, "Primal-Dual avec Pred lamb = "+str(lambd)))
    
    # affichage du graphique
    show_multiple_sims(resA)
    


simulations2()