functions {
  vector lp_reduce( vector beta , vector theta , real[] xr , int[] xi ) {
    int n = size(xr);
    int y[n] = xi[1:n];
    int m[n] = xi[(n+1):(2*n)];
    real lp = binomial_logit_lpmf( y | m , beta[1] + to_vector(xr) * beta[2] );
    return [lp]';
  }
} 
data {
  int N;
  int n_redcards[N];
  int n_games[N];
  real rating[N];
}
transformed data {
  // 7 shards
  // M = N/7 = 124621/7 = 17803
  int n_shards = 7;
  int M = N/n_shards;
  int xi[n_shards, 2*M];  // 2M because two variables, and they get stacked in array
  real xr[n_shards, M];
  // an empty set of per-shard parameters
  vector[0] theta[n_shards];
  // split into shards
  for ( i in 1:n_shards ) {
    int j = 1 + (i-1)*M;
    int k = i*M;
    xi[i,1:M] = n_redcards[ j:k ];
    xi[i,(M+1):(2*M)] = n_games[ j:k ];
    xr[i] = rating[ j:k ];
  }
}
parameters {
  vector[2] beta;
}
model {
  beta ~ normal(0,1);
  target += sum( map_rect( lp_reduce , beta , theta , xr , xi ) );
}
