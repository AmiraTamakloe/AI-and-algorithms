% Jacob Ducas (2138758)
% Amira-Fatim Tamakloe (2131198)
% Nour Zahreddine (2167661)

include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de tâches
int: n;
% Durée maximal du projet
int: m;
% Limite de ressources pouvant être partagées entre les tâches en cours
int: limite;

% Durée des tâches
array[1..n] of 1..m: duree_tache;
% Ressources requises par les tâches
array[1..n] of 1..m: ressource_tache;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

array[1..n] of var 1..m: debut_tache; % Début des tâches
array[1..n] of var 1..m: fin_tache; % Fin des tâches

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Fin du projet
var int: fin_projet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
% Temps de fin correct
constraint forall(i in 1..n) (
    fin_tache[i] = debut_tache[i] + duree_tache[i]
);

% Max deux tâches commencées en même temps
constraint forall(t in 1..m) (
    sum([if debut_tache[i] == t then 1 else 0 endif | i in 1..n]) <= 2
);

% Ne pas dépasser la limite
constraint cumulative(debut_tache, duree_tache, ressource_tache, limite);

% Minimiser fin_projet
constraint fin_projet = max([fin_tache[i] | i in 1..n]);
solve minimize fin_projet;

%-----------------------------------------------------------------------------%

output [
  "Envergure du projet  : ", show(fin_projet), "\n",
  "Début des tâches     : ", show(debut_tache), "\n",
  "Fin des tâches       : ", show(fin_tache), "\n",
];