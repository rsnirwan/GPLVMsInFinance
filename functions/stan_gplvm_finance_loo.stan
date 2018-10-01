functions {
    matrix cov_linear(vector[] X1, vector[] X2, real sigma){
        int N = size(X1);
        int M = size(X2);
        int Q = num_elements(X1[1]);
        matrix[N,M] K;
        {
            matrix[N,Q] x1;
            matrix[M,Q] x2;
            for (n in 1:N)
                x1[n,] = X1[n]';
            for (m in 1:M)
                x2[m,] = X2[m]';
            K = x1*x2';
        }
        return square(sigma)*K;
    }
    
    matrix cov_matern32(vector[] X1, vector[] X2, real sigma, real l){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        for (n in 1:N)
            for (m in 1:M){
                dist = sqrt(squared_distance(X1[n], X2[m]) + 1e-14);
                K[n,m] = square(sigma)*(1+sqrt(3)*dist/l)*exp(-sqrt(3)*dist/l);
            }
        return K;
    }
    
    matrix cov_matern52(vector[] X1, vector[] X2, real sigma, real l){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        for (n in 1:N)
            for (m in 1:N){
                dist = sqrt(squared_distance(X1[n], X2[m]) + 1e-14);
                K[n,m] = square(sigma)*(1+sqrt(5)*dist/l+5*square(dist)/(3*square(l)))*exp(-sqrt(5)*dist/l);
            }
        return K;
    }
    
    matrix cov_exp_l2(vector[] X1, vector[] X2, real sigma, real l){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        for (n in 1:N)
            for (m in 1:M){
                dist = sqrt(squared_distance(X1[n], X2[m]) + 1e-14);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
    
    matrix cov_exp(vector[] X1, vector[] X2, real sigma, real l){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        int Q = rows(X1[1]);
        for (n in 1:N)
            for (m in 1:M){
                dist = 0;  //sqrt(squared_distance(X1[n], X2[m]) + 1e-14);
                for (i in 1:Q)
                    dist = dist + fabs(X1[n,i] - X2[m,i]);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
    
    matrix multipy_vec_mat_vec(vector a, matrix B, vector c){
        //int N = size(a);
        //int M = size(b);
        //matrix[N, M] D;
        return diag_post_multiply( diag_pre_multiply(a, B), c );
    }
    
    matrix kernel_f(vector[] X1, vector[] X2, real sigma, real l, 
                    real a, int model_number, vector diag_stds1, vector diag_stds2){
        // X:latent space, sigma:kernel_std, l:lengthscale, a:alpha (for linear kernel)
        // diag_stds: variances of every datapoints
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        if (model_number==0){
            K = cov_linear(X1, X2, a);           // K = a^2*X1*X2.T
        }
        else if (model_number==1){
            //print("cov_exp_quad");
            K = cov_exp_quad(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2);
        }
        else if (model_number==2){
            //print("cov_exp");
            K = cov_exp(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2);
        }
        else if (model_number==3){
            K = cov_matern32(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2);
        }
        else if (model_number==4){
            K = cov_matern52(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2);
        }
        else if (model_number==5){
            K = cov_exp_quad(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2).*cov_linear(X1, X2, a);
        }
        else if (model_number==6){
            K = cov_exp(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2).*cov_linear(X1, X2, a);
        }
        else if (model_number==7){
            K = cov_exp_quad(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2) + cov_linear(X1, X2, a);
        }
        else {
            K = cov_exp(X1, X2, sigma, l);
            K = multipy_vec_mat_vec(diag_stds1, K, diag_stds2) + cov_linear(X1, X2, a);
        }
        return K;
    }
    matrix gp_pred_loo(vector[] X, matrix Y, vector noise_stds,
                        real kernel_lengthscale, vector diag_stds, 
                        real alpha, int model_number){
        // Fits a GP to each column d of Y. Fits each column N times, where in n-th run 
        // n-th element of Y[:,d] is missing. (See leaving-one-out cross-validation)
        
        int N = size(X);
        int Q = num_elements(X[1]);
        int D = num_elements(Y[1]);
        matrix[N, D] Y_hat; 
        {
            int N_pred = 1;
            matrix[N-1,N-1] K;
            matrix[N-1,N-1] L;
            matrix[N_pred,N-1] K_x_pred_x;
            matrix[N_pred,N_pred] K_x_pred_x_pred;
            matrix[N-1, N_pred] L_div_K_x_x_pred;
            vector[N-1] K_div_y;            

            vector[Q] X_given_stocks[N-1];
            vector[Q] X_missing_stock[1];
            vector[N-1] diag_stds_given;
            vector[1] diag_stds_missing;
            vector[N-1] noise_stds_given;
            vector[1] noise_stds_missing;

            vector[N-1] Y_m_n_d;


            for (n in 1:N){
                X_given_stocks[1:n-1,] = X[1:n-1,];
                X_given_stocks[n:,] = X[n+1:,];
                X_missing_stock[1,:] = X[n,:];

                diag_stds_given[1:n-1] = diag_stds[1:n-1];
                diag_stds_given[n:] = diag_stds[n+1:];
                diag_stds_missing[1] = diag_stds[n];
                
                noise_stds_given[1:n-1] = noise_stds[1:n-1];
                noise_stds_given[n:] = noise_stds[n+1:];
                noise_stds_missing[1] = noise_stds[n];
                
                K = kernel_f(X_given_stocks, X_given_stocks, 1., kernel_lengthscale, 
                            alpha, model_number, diag_stds_given, diag_stds_given);
                K_x_pred_x = kernel_f(X_missing_stock, X_given_stocks, 1., 
                                        kernel_lengthscale, alpha, model_number, 
                                        diag_stds_missing, diag_stds_given);
                for (n_ in 1:N-1)
                    K[n_,n_] = K[n_,n_] + square(noise_stds_given[n_]) + 1e-14;
                L = cholesky_decompose(K);

                for (d in 1:D){
                    Y_m_n_d[1:n-1] = Y[1:n-1,d];
                    Y_m_n_d[n:] = Y[n+1:,d];

                    K_div_y = mdivide_left_tri_low(L, Y_m_n_d);
                    K_div_y = mdivide_right_tri_low(K_div_y', L)';
                    Y_hat[n,d] = (K_x_pred_x * K_div_y)[1];
                }
            }
        }
        return Y_hat;
    }
}
data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> Q;
    matrix[N,D] Y;
    int<lower=1,upper=D> D_pred;          // how many days we want to predict
    int<lower=0,upper=8> model_number;    // used by function 'kernel_f()' for model choice
}
transformed data {
    int<lower=1,upper=D> D_train = D-D_pred; 
    vector[N] mu = rep_vector(0, N);
    matrix[N,D_train] Y_train = Y[,:D_train];
    matrix[N,D_pred] Y_pred = Y[,D_train+1:];
}
parameters {
    //vector[N] mu;
    vector[Q] X[N];                       // latent space
    real<lower=0> kernel_lengthscale;     // kernel lengthscale
    vector<lower=0>[N] diag_stds;         // std for each stock
    vector<lower=0>[N] noise_stds;         // observation noise ... non isotropic a la factor model
    real<lower=0> alpha;                  // kernel std for linear kernel
}
transformed parameters {
    matrix[N,N] L;
    real R2 = 0;
    {
        // we set kernel_std to 1. and model different stds for different points by diag_stds
        matrix[N,N] K = kernel_f(X, X, 1., kernel_lengthscale, 
                                    alpha, model_number, diag_stds, diag_stds);
        
        for (n in 1:N)
            K[n,n] = K[n,n] + pow(noise_stds[n], 2) + 1e-14;
        L = cholesky_decompose(K);
        
        R2 = sum(1 - square(noise_stds) ./diagonal(K) )/N;
    }
}
model {
    for (n in 1:N)
        //X[n] ~ cauchy(0, 1);
        X[n] ~ normal(0, 1);
        
    //mu ~ normal(0, .5);
    diag_stds ~ normal(0, .5);
    noise_stds ~ normal(0, .5);
    kernel_lengthscale ~ inv_gamma(3.0,1.0);// inv_gamma for zero-avoiding prop
    alpha ~ inv_gamma(3.0,1.0);             // kernel std for linear kernel
    
    for (d in 1:D_train) 
        col(Y_train,d) ~ multi_normal_cholesky(mu, L);
}
generated quantities {
    matrix[N, D_pred] Y_hat;
    vector[N] R2_N_pred;
    vector[N] R2_N;
    real R2_hat_pred = 0;
    real R2_hat = 0;
    real mean_squared_error = 0;
    real mean_abs_error = 0;
    
    
    {
        matrix[N, D_pred] resid_pred;
        
        Y_hat = gp_pred_loo(X, Y_pred, noise_stds, kernel_lengthscale, 
                            diag_stds, alpha, model_number);
        resid_pred = Y_pred - Y_hat;
        
        for (n in 1:N)
            R2_N_pred[n] = 1 - sum( square(row(resid_pred,n)) )/ 
                                    sum( square(row(Y_pred,n)-mean(row(Y_pred,n))) );
        
        R2_hat_pred = mean(R2_N_pred);
        mean_squared_error = sum(square(resid_pred))/(N*D_pred);
        mean_abs_error = sum(fabs(resid_pred))/(N*D_pred);
    }
      
        
    {
        matrix[N,N] K = kernel_f(X, X, 1., kernel_lengthscale, alpha, 
                                model_number, diag_stds, diag_stds);
        matrix[N,N] K_noise = K;
        matrix[N,D] Y_hat_in_sample;
        matrix[N,D] resid_in_sample;
        
        for (n in 1:N)
            K_noise[n,n] = K_noise[n,n] + pow(noise_stds[n], 2) + 1e-14;
        Y_hat_in_sample = K * mdivide_left_spd(K_noise, Y);
        resid_in_sample = Y - Y_hat_in_sample;
        
        for (n in 1:N)
            R2_N[n] = 1 - sum( square(row(resid_in_sample,n)) )/ 
                                    sum( square(row(Y,n)-mean(row(Y,n))) );
        
        R2_hat = mean(R2_N);
        K = K_noise;           //includes non-isotropic noise to the output K 
    }
}






