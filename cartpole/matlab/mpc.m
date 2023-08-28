function u = MPC(q,q_des,L_des,N,dtMPC)
   
    states  =  q;
    
    %Discretize dynamics 
    [A,B] = linDiscreteDyn(L_des,dtMPC);
    
    %Setup with horizon N
    dim_s = 4; % cart-pole # states
    decayRate = 1.0;
    dim_q = (4*(N+1))+ N; % dim(state) = 4
    dim_u = N;
    Q = [];
    f = [];
    
    %Cost function penalization
%     diagCost = [0; 100; 0; 1];
    diagCost = [10^-6; 100; 10^-6; 1^-1];
    Qq = diag(diagCost);

    %Build cost function for optimization vars
    for i = 1:(N+1)
        Qq = Qq * decayRate^(i-1);
        Q = blkdiag(Q,Qq);
        q_des_cost = -2.0*q_des.*diagCost;
        f = [f q_des_cost'];
    end 
    R = 0.01*eye(N);
%     R = 0*eye(N);
    for i =1:N
        R(i,i) = R(i,i)*decayRate^(i-1);
    end 
    H = blkdiag(Q,R);
    f = [f zeros(1,N)];
    
    %Inforcing dynamic constraints
    Aeq = zeros(dim_s*N,(dim_s*(N+1))+N);
    Atop = [eye(dim_s) zeros(dim_s,dim_q-dim_s)]; %n=0
    beq = [];
    j = 1;
    for i = 1:dim_s:dim_s*N
        Aeq(i:i+(dim_s-1),i:i+(dim_s-1)) = -A;
        Aeq(i:i+(dim_s-1),i+dim_s:i+(2*dim_s-1)) = eye(dim_s);
        Aeq(i:i+(dim_s-1),dim_q-dim_u+j) = -B;
        j = j+1;
    end 
    Aeq = [Atop;Aeq];
    beq = [states;zeros(4*N,1)];

    
    %Manual expansion of dynamic constraint
%     Aeq = [];  
%     Aeq = [eye(4) zeros(4,dim_q-4)]; %n=0
%     Aeq = [Aeq; [-A eye(4) zeros(4,dim_q - 8 - dim_u) -B zeros(4,dim_u-1)]]; %n=1
%     Aeq = [Aeq; [-A^2 zeros(4,4) eye(4) zeros(4,dim_q - 12 - dim_u) -A*B -B zeros(4,dim_u-2)]]; %n=2
%     Aeq = [Aeq; [-A^3 zeros(4,8) eye(4) zeros(4,dim_q - 16 - dim_u) -A^2*B -A*B -B zeros(4,dim_u-3)]]; %n=3
%     Aeq = [Aeq; [-A^4 zeros(4,12) eye(4) zeros(4,dim_q - 20 - dim_u) -A^3*B -A^2*B -A*B -B zeros(4,dim_u-4)]]; %n=4
%     Aeq = [Aeq; [-A^5 zeros(4,16) eye(4) zeros(4,dim_q - 24 - dim_u) -A^4*B -A^3*B -A^2*B -A*B -B zeros(4,dim_u-5)]]; %n=5
%     Aeq = [Aeq; [-A^6 zeros(4,20) eye(4) zeros(4,dim_q - 28 - dim_u) -A^5*B -A^4*B -A^3*B -A^2*B -A*B -B zeros(4,dim_u-6)]]; %n=6
%     Aeq = [Aeq; [-A^7 zeros(4,24) eye(4) zeros(4,dim_q - 32 - dim_u) -A^6*B -A^5*B -A^4*B -A^3*B -A^2*B -A*B -B zeros(4,dim_u-7)]]; %n=7
%     Aeq = [Aeq; [-A^8 zeros(4,28) eye(4) zeros(4,dim_q - 36 - dim_u) -A^7*B -A^6*B -A^5*B -A^4*B -A^3*B -A^2*B -A*B -B zeros(4,dim_u-8)]]; %n=8
%     Aeq = [Aeq; [-A^9 zeros(4,32) eye(4) zeros(4,dim_q - 40 - dim_u) -A^8*B -A^7*B -A^6*B -A^5*B -A^4*B -A^3*B -A^2*B -A*B -B zeros(4,dim_u-9)]]; %n=9
%     Aeq = [Aeq; [-A^10 zeros(4,36) eye(4) zeros(4,dim_q - 44 - dim_u) -A^9*B -A^8*B -A^7*B -A^6*B -A^5*B -A^4*B -A^3*B -A^2*B -A*B -B]]; %n=10
%     beq = [states;zeros(4*N,1)];

    %Inforcing Inequality Constr. on rate of change on input
    A_ineq = zeros(N-1,dim_q);
    row = 1;
    for i = (dim_q-dim_u+1):dim_q-1
        A_ineq(row,i:i+1) = [-1 1];
        row = row +1;
    end 
    b_ineq = 40*ones(dim_u-1,1);
    
    lb_motor = -75*ones(dim_u,1);
    ub_motor = 75*ones(dim_u,1);
    lb = [-Inf(dim_q-dim_u,1);lb_motor]; % Lower bound
    ub = [ Inf(dim_q-dim_u,1);ub_motor];   % Upper bound
    
%     u = quadprog(H,f,A_ineq,b_ineq,Aeq,beq,lb,ub);    
    u = quadprog(H,f,[],[],Aeq,beq);    

end 