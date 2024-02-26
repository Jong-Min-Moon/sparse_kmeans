rho_vec = [0.05, 0.15, 0.2, 0.35, 0.4, 0.45];
p_vec = [100,200,300,400,500,600,700,800,900,1000];
for rho = rho_vec
    for p = p_vec
        Omega = zeros([p,p]);

        for j=1:p
            for l = 1:p
                if j==l
                    Omega(j,l) = 1;
                elseif abs(j-l) ==1
                    Omega(j,l) = rho;
                end
            end
        end

        try chol(Omega);
            %disp('Matrix is symmetric positive definite.')
        catch ME
            disp('Matrix is not symmetric positive definite')
            dist('rho_vec')
            dsip('p_vec')
        end
    end
end
