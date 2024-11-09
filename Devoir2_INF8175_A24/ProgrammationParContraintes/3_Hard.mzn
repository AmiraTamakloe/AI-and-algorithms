% Jacob Ducas (2138758)
% Amira-Fatim Tamakloe (2131198)
% Nour Zahreddine (2167661)

include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de villes à visiter + point de départ de la tournée
int: n;
 % Distance entre chaque villes (symmétrique)
array[1..n,1..n] of int: distance;
 % Distance minimale possible entre deux villes différentes
int: min_val = min([distance[i,j] | i,j in 1..n where distance[i,j] > 0]);
% Distance maximale possible entre deux villes différentes
int: max_val = max([distance[i,j] | i,j in 1..n]);

% Fenêtre de temps (1 = début, 2 = fin) durant laquelle il est possible de jouer pour chaque ville
array[1..n,1..2] of int: fenetre_arrivee;
% Temps maximum pour tout le trajet
int: max_trajet = max([fenetre_arrivee[i,2] | i in 1..n]);
% Fenêtre de temps la plus grande du trajet
int: max_fenetre = max([fenetre_arrivee[i,2] - fenetre_arrivee[i,1]| i in 2..n]);

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Ordre des villes visitées par la tournée
array[1..n] of var 1..n: chemin_tournee;
% Circuit de la tournée, circuit_tournee[i] = j => j est le successeur de i
array[1..n] of var 1..n: circuit_tournee;
% Temps cumulé lors de l'arrivé au point i de la tournée
array[1..n] of var 0..max_trajet: cumul_temps;
% Temps d'attente avant le spectacle en ville i
array[1..n] of var 0..max_fenetre: temps_attente;
% Temps d'attente total 
var 0..max_trajet: total_temps_attentes = sum(temps_attente);
var 1..n: index;

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Temps de trajet à minimiser
var int: temps_trajet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
% Le chemin indique la succession de ville à partir de la première
constraint chemin_tournee[n] = 1;
constraint circuit_tournee[1] = chemin_tournee[1]; 
 
% S'assurer d'avoir un circuit valide
constraint circuit(circuit_tournee);

% S'assurer que l'on visite toute les villes une seule fois
constraint alldifferent(chemin_tournee);
constraint alldifferent(circuit_tournee);

% Liaison entre chemin_tournee et circuit_tournee
constraint forall(i in 1..n-1) (
   circuit_tournee[chemin_tournee[i]] = chemin_tournee[i + 1]
 );

% On ne peut pas gagner du temps durant le parcours et si cumul_temps[i] == cumul_temps[i-1], on n'aurait pas changé de ville
constraint forall(i in 1..n-1) (
  cumul_temps[i] < cumul_temps[i+1]
);

% La première ville du chemin est la ville après 1
constraint temps_attente[chemin_tournee[1]] == max(0, fenetre_arrivee[chemin_tournee[1],1] - distance[1, chemin_tournee[1]]);

% Le temps cumulé au temps 1 va être équivalent à la distance parcourue si on arrive dans fenetre sinon il faudra attentre l'ouverture de la ville
constraint if distance[1, chemin_tournee[1]] >= fenetre_arrivee[chemin_tournee[1],1] /\ distance[1, chemin_tournee[1]] <= fenetre_arrivee[chemin_tournee[1],2] then cumul_temps[1] = distance[1, chemin_tournee[1]] else cumul_temps[1] = fenetre_arrivee[chemin_tournee[1],1] endif;

% Ne pas rentrer dans une ville hors de sa fenêtre
constraint forall(i in 2..n) (
  cumul_temps[i] >= fenetre_arrivee[chemin_tournee[i],1] /\
  cumul_temps[i] <= fenetre_arrivee[chemin_tournee[i],2]
);

% Le temps cumul suivant va être cumul_temps[precedent] + distance si dans fenetre sinon fenetre[curr, 1]
constraint forall(i in 2..n) (
if distance[chemin_tournee[i], chemin_tournee[i-1]] + cumul_temps[i-1] >= fenetre_arrivee[chemin_tournee[i],1] then 
  cumul_temps[i] = distance[chemin_tournee[i], chemin_tournee[i-1]] + cumul_temps[i-1] 
  else 
    if distance[chemin_tournee[i], chemin_tournee[i-1]] + cumul_temps[i-1] < fenetre_arrivee[chemin_tournee[i],1] then
      cumul_temps[i] = fenetre_arrivee[chemin_tournee[i],1] endif
    endif
);

% Si on arrive avant la fenêtre, il faut considérer ce temps
constraint forall(i in 2..n) (
  if cumul_temps[i-1] + distance[chemin_tournee[i], chemin_tournee[i-1]] < cumul_temps[i] 
    then temps_attente[chemin_tournee[i]] = cumul_temps[i] - cumul_temps[i-1] - distance[chemin_tournee[i], chemin_tournee[i-1]] 
  else 
    temps_attente[chemin_tournee[i]] = 0 endif
);

% Identifier le temps total du trajet à la fin
constraint temps_trajet = cumul_temps[n];

% Objectif
solve minimize temps_trajet;
%-----------------------------------------------------------------------------%

output [
  "Chemin de la tournée                   : \(chemin_tournee)\n",
  "Circuit de la tournée                  : \(circuit_tournee)\n",
  "Temps cumulés avant d'entrer en ville  : \(cumul_temps)\n",
  "Temps d'attente avant l'entrée en ville: \(temps_attente)\n",
  "Temps d'attente total                  : \(total_temps_attentes)\n",
  "Temps de trajet total                  : \(temps_trajet)\n",
];
