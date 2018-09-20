data {
  int N;
  int n_redcards[N];
  int n_games[N];
  real rating[N];
}
parameters {
  vector[2] beta;
}
model {
  beta ~ normal(0,1);
  n_redcards ~ binomial_logit( n_games , beta[1] + beta[2] * to_vector(rating) );
}
