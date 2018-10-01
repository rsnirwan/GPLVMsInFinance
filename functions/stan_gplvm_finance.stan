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
                    dist = dist + abs(X1[n,i] - X2[m,i]);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
    
    matrix kernel_f(vector[] X1, vector[] X2, real sigma, real l, 
                    real a, int model_number, vector diag_stds){
        // X:latent space, sigma:kernel_std, l:lengthscale, f:frequency, a:alpha
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        if (model_number==0)
            K = cov_linear(X1, X2, a);               // K = a^2*X1*X2.T
        else if (model_number==1){
            K = cov_exp_quad(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds);
        }
        else if (model_number==2){
            K = cov_exp(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds);
        }
        else if (model_number==3){
            K = cov_matern32(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds);
        }
        else if (model_number==4){
            K = cov_matern52(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds);
        }
        else if (model_number==5){
            K = cov_exp_quad(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds).*cov_linear(X1, X2, a);
        }
        else if (model_number==6){
            K = cov_exp(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds).*cov_linear(X1, X2, a);
        }
        else if (model_number==7){
            K = cov_exp_quad(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds) + cov_linear(X1, X2, a);
        }
        else {
            K = cov_exp(X1, X2, sigma, l);
            K = quad_form_diag(K, diag_stds) + cov_linear(X1, X2, a);
        }
        return K;
    }
}
data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> Q;
    matrix[N,D] Y;
    int<lower=0,upper=8> model_number;    // used by function 'kernel_f()' for model choice
}
transformed data {
    vector[N] mu = rep_vector(0, N);
}
parameters {
    //vector[N] mu;
    vector[Q] X[N];                       // latent space
    real<lower=0> kernel_lengthscale;     // kernel lengthscale
    vector<lower=0>[N] diag_stds;         // std for each stock
    vector<lower=0>[N] noise_std;         // observation noise ... non isotropic a la factor model
    real<lower=0> alpha;                  // kernel std for linear kernel
}
transformed parameters {
    matrix[N,N] L;
    real R2 = 0;
    {
        // we set kernel_std to 1. and model different stds for different points by diag_stds
        matrix[N,N] K = kernel_f(X, X, 1., kernel_lengthscale, 
                                 alpha, model_number, diag_stds);
        
        for (n in 1:N)
            K[n,n] = K[n,n] + pow(noise_std[n], 2) + 1e-14;
        L = cholesky_decompose(K);
        
        R2 = sum(1 - square(noise_std) ./diagonal(K) )/N;
    }
}
model {
    for (n in 1:N)
        //X[n] ~ cauchy(0, 1);
        X[n] ~ normal(0, 1);
        
    //mu ~ normal(0, .5);
    diag_stds ~ normal(0, .5);
    noise_std ~ normal(0, .5);
    kernel_lengthscale ~ inv_gamma(3.0,1.0);// inv_gamma for zero-avoiding prop
    alpha ~ inv_gamma(3.0,1.0);             // kernel std for linear kernel
    
    for (d in 1:D) 
        col(Y,d) ~ multi_normal_cholesky(mu, L);
}
generated quantities {
    real log_likelihood = 0;   //just the log_likelihood values. without log_prior
    real R2_hat_N = 0;
    vector[N] R2_hat_vec_N;
    //matrix[N,D] Y_hat;
    
    matrix[N,N] K = kernel_f(X, X, 1., kernel_lengthscale, 
                             alpha, model_number, diag_stds);
    
    for (d in 1:D)
        log_likelihood = log_likelihood + multi_normal_cholesky_lpdf(col(Y,d) | mu, L);
        
    {
        matrix[N,N] K_noise = K;
        matrix[N,D] Y_hat;
        matrix[N,D] resid;
        
        for (n in 1:N)
            K_noise[n,n] = K_noise[n,n] + pow(noise_std[n], 2) + 1e-14;
        Y_hat = K * mdivide_left_spd(K_noise, Y);
        resid = Y - Y_hat;
        for (n in 1:N)
            R2_hat_vec_N[n] = 1 - sum( square(row(resid,n)) )/ sum( square(row(Y,n)-mean(row(Y,n))) );
        
        R2_hat_N = mean(R2_hat_vec_N);
        K = K_noise;           //includes non-isotropic noise to the output K 
    }
    //print(R2_hat_N);
}
