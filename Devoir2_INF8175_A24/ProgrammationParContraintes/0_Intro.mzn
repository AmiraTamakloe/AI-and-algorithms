% Jacob Ducas (2138758)
% Amira-Fatim Tamakloe (2131198)
% Nour Zahreddine (2167661)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Nombre devant satisfaire les critères
var 1000..9999: nombre;

% Décomposition du nombre en un ensemble de chiffres
array[1..4] of var 1..9: chiffres;

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Variable à minimiser
var int: p;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

% C’est un nombre pair
constraint chiffres[4] mod 2 == 0; 

% Pas de zéro
constraint at_most(0, chiffres, 0);

% Les 4 chiffres sont différents
constraint alldifferent(chiffres);

% Le chiffre à la position des milliers est supérieur à celui à la position des centaines
constraint chiffres[1] > chiffres[2];

% Le chiffre à la position des dizaines est inférieur à celui à la position des unités
constraint chiffres[3] < chiffres[4];

% Le chiffre à la position des centaines est supérieur à celui à la position des unités
constraint chiffres[2] > chiffres[4];

% La somme des 4 chiffres est supérieure à 15 
constraint sum(chiffres) > 15; 

% Le produit des 3 derniers chiffres (chiffre à la position des centaines×chiffre à la position des dizaines × chiffre à la position des unités) doit être minimisé.
p = chiffres[2] * chiffres[3] * chiffres[4]; % Produit des 3 derniers chiffres
solve minimize p;

constraint nombre = 1000 * chiffres[1] + 100 * chiffres[2] + 10 * chiffres[3] + chiffres[4];
%-----------------------------------------------------------------------------%
output [
  show(nombre)
];



