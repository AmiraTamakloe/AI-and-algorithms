% Jacob Ducas (2138758)
% Amira-Fatim Tamakloe (2131198)
% Nour Zahreddine (2167661)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre d'acteurs et de rôles
int: n;
% Couleurs des costumes
enum costumes; 
% Costume démandé pour un rôle i (première dimension) par un acteur j (seconde dimension)
array[1..n,1..n] of costumes: choix_costume;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Variables de décision pour assigner un acteur à un rôle
array [1..n] of var 1..n: assignation_role;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% Tous les rôles sont différents
constraint alldifferent(assignation_role);

% Les rôles ne sont pas adjacents
constraint forall(i in 1..n-1) (
  abs(assignation_role[i] - assignation_role[i + 1]) > 1
);

% Tous les costumes sont différents
constraint all_different([choix_costume[assignation_role[i], i] | i in 1..n]);

solve satisfy;
%-----------------------------------------------------------------------------%

output	
  ["L'acteur " ++ format(i) ++ 
   " prendra le rôle " ++ format(fix(assignation_role[i])) ++ 
   " (costume " ++ format(choix_costume[fix(assignation_role[i]), i]) ++ ")" ++"\n" 
  | i in 1..n];