function [A_dis,B_dis] = linDiscreteDyn(L_des,dt)

    L = L_des;
    
    %Cont. Dynamics, function of L
    A = [[0,    -98.1, 0, 0]
         [0, 107.91/L, 0, 0]];
    A = [[zeros(2,2) eye(2)]; A];
    
    B = [1.0;
        -1.0/L];
    B = [zeros(2,1); B];
    
    %Discretization (ZOH method)
    A_dis = expm((A*dt));
    B_dis  = pinv(A)*(A_dis - eye(4))*B;
end 