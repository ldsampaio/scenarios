Multicell Multiuser MultiBand Telecommunications Scenario Creation Tool
===============

This scripts provides tools for different scenarios creation in resource allocation of telecommunications schemes.

Usage
-------------

Import scenarios to your py file and use it through the method call:

create_scenario(C,R,K,Fmin,Fmax,S,gamma,W,Bmax,d0) where:

* C is the cluster size (check Centered Hexagon Number Wikipedia for further reference)
* R is the cell radius in meters
* K is the number of users per cell
* Fmin is the smallest carrier frequency possible
* Fmax is the largest carrier frequency possible
* S is the Shadowing variance in dB
* gamma is the path loss exponent (for Simplified Path Loss Model)
* W is the number of spectrum wholes available
* Bmax is the maximum bandwidth per whole
* d0 is the reference distance (for Simplified Path Loss Model) 

Returns:

* k_x is the x-axis position of each user in the system
* k_y is the y-axis position of each user in the system
* c_x is the x-axis position of each base station in the system
* c_y is the y-axis position of each base station in the system
* d is the distance hipermatrix with user, origin cell and destination cell indexers
* tmp is the shadowing effect hipermatrix  with user, origin cell, destination cell and carrier frequency indexers
* beta is the channel gain hipermatrix with user, origin cell, destination cell and carrier frequency indexers


To do List
-------------

1. Incorporate other path loss models
2. Incorporate fading models (Rice, Rayleigh and Nakagami-m)
3. Extend cluster size to C > 4

Disclaimer
----------

**Only works** for clustersize (C <= 4)
